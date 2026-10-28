"""Only missing direct cumulative latents; no tokenizer or old VAR completions."""
from residual_runtime import *
import torch,signal,traceback,collections
stop=False
def on_stop(*args):
    global stop
    stop=True
def main(path,expected):
    assert sha(path)==expected;r=read(path)
    for p in r['code']:checked(p)
    out=pathlib.Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic();signal.signal(signal.SIGTERM,on_stop);signal.signal(signal.SIGINT,on_stop)
    try:
        n,dc,statehash=ledger.call('model_load','once',lambda:load_frozen(r,out));codec=FitCodec(R,native=n)
        train=read(checked(r['train']))['sources'];dev=read(checked(r['dev']))['sources'];assert len(train)==19968 and len(dev)==200 and not({s['source_id']for s in train}&{s['source_id']for s in dev})
        prior={x['source']['source_id']:x for x in read(checked(r['b0_cache']))};result=[];groups=collections.defaultdict(list);verified=set()
        for split,ss in [('train',train),('dev',dev)]:
            for s in ss:groups[(split,s['old_latent_shard']['path'])].append(s)
        def load(d):
            p=d['path']
            if p not in verified:checked(d);verified.add(p)
            return torch.load(p,map_location='cpu',weights_only=True)
        for gi,((split,_),ss)in enumerate(groups.items()):
            if stop or(L/'STOP').exists()or time.monotonic()-begin>r['seconds']-120:
                save(out/'partial_index.json',result);save(out/'completion.json',dict(status='SAFE_BOUNDARY_STOP',sources=len(result),ledger=ledger.summary()));return
            old=load(ss[0]['old_latent_shard']);rgb=load(ss[0]['rgb_shard'])
            for s in ss:
                i=s['within_shard'];assert old['image_ids'][i]==rgb['image_ids'][i]==s['source_id']
            for off in range(0,len(ss),16):
                bb=ss[off:off+16];todo=[s for s in bb if s['source_id']not in prior];dest=out/(f'{split}_{gi:04d}_{off:03d}');dest.mkdir()
                if todo:
                    tokens=torch.stack([old['full_tokens'][s['within_shard']].long()for s in todo])
                    def make():
                        full,fc=direct_latents(n,tokens.numpy());full=full.cpu();fc=fc.cpu();torch.save(dict(full=full,coarse=fc,tokens=tokens,source_ids=[s['source_id']for s in todo]),dest/'latents.pt');return pin(dest/'latents.pt')
                    cached=ledger.call('latent_batch',f'{split}_{gi}_{off}',make)
                for s in bb:
                    i=s['within_shard'];sid=s['source_id'];d=dict(source_id=sid,source_index=s['source_index'],split=split,rgb_shard=s['rgb_shard'],within_rgb=i)
                    if sid in prior:
                        x=prior[sid];d.update(latents=x['cache'],within_latent=0,layout='single',reused=True,old_Fq_max_abs_difference=x['old_Fq_max_abs_difference'])
                    else:
                        j=todo.index(s);d.update(latents=cached,within_latent=j,layout='batch',reused=False)
                    if split=='dev':
                        tokens=old['full_tokens'][i].numpy().astype(np.int64)
                        def coarse_wire():
                            e=codec.Encoder();tables={}
                            with codec._inference('EC_VAR_WHOLE'),codec._context('EC_VAR_WHOLE')as p:
                                for scale in range(4):
                                    tab=codec._table(p,scale,[]);tables['cdf'+str(scale)]=tab;v=tokens[OFFSETS[scale]:OFFSETS[scale+1]];e.encode(v,tab);p.advance(v)
                            bits=np.asarray(e.finish(),np.uint8);file=dest/(str(s['source_index'])+'_cdf.npz')
                            with file.open('xb')as f:np.savez(f,**tables,bits=bits)
                            return pin(file)
                        d['coarse_stream']=ledger.call('coarse_TX',sid,coarse_wire)
                        def check_rx():
                            with np.load(checked(d['coarse_stream']),allow_pickle=False)as z:bits=z['bits'].copy()
                            rx=codec.receive_single(bits,4,0);fc=coarse_latent(n,rx['C']).cpu();z=load(d['latents']);reference=z['coarse'][d['within_latent']:d['within_latent']+1]
                            assert rx['C']==tokens[:OFFSETS[4]].tolist()and torch.equal(fc,reference)
                            return dict(source_id=sid,actual_coarse_bits=len(bits),coarse_token_exact=True,Fc_exact=True,TX_CDF_used=False)
                        d['coarse_RX_proof']=ledger.call('coarse_RX',sid,check_rx)
                    result.append(d)
                save(dest/'commit.json',result[-len(bb):])
            del old,rgb
            if gi%10==0:print('RESIDUAL_CACHE',len(result),20168,flush=True)
        assert len(result)==20168 and len({s['source_id']for s in result})==20168
        save(out/'index.json',dict(sources=result,train_sources=19968,dev_sources=200,dev_is_calibration_subset=True,frozen_models=n.loaded['identity'],reused_b0=4))
        save(out/'completion.json',dict(status='CACHE_COMPLETE',index=pin(out/'index.json'),sources=len(result),new_latents=20164,reused_latents=4,dev_TX=200,dev_RX=200,ledger=ledger.summary(),seconds=time.monotonic()-begin))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
