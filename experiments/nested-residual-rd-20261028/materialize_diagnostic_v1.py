"""Fixed diagnostic anchors; immutable prior scores, independently decoded streams."""
from residual_runtime import *
from residual_model import ResidualCodec
from train_residual_v1 import Data
from mixed_arithmetic_v1 import classes
from nested_entropy import encode_candidates,decode_residual
from fit_common import csvsave
import torch,traceback

def main(path,expected):
    assert sha(path)==expected;r=read(path);[checked(v)for v in r['code']]
    out=pathlib.Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic()
    try:
        n,dc,statehash=ledger.call('model_load','once',lambda:load_frozen(r,out));lp=ledger.call('lpips_load','once',lambda:load_lpips(r))
        cp=torch.load(checked(r['checkpoint']),map_location='cpu',weights_only=False);model=ResidualCodec(r['order']).cuda().eval();model.load_state_dict(cp['model']);assert statehash(model)==cp['model_state_sha256']
        plan=read(checked(r['plan']));data=Data(r['cache_index']);sources=[data.by[s['source_id']]for s in plan['sources']];Encoder,Decoder=classes(R/'src/var_comm/whole_entropy.py')
        oldrows=read(checked(r['old_summary']))['rows'];old={(v['source_id'],v['budget']):v for v in oldrows};memo={}
        for s in sources:
            tokens=torch.load(checked(s['rx_coarse']),map_location='cpu',weights_only=True)['tokens']
            with np.load(checked(s['rx_cdfs']),allow_pickle=False)as z:
                for q in range(4):
                    key=(q,tuple(tokens[:OFFSETS[q]]));t=z['cdf'+str(q)]
                    if key in memo:assert np.array_equal(memo[key],t)
                    else:memo[key]=t
        def receive(bits,j):
            d=Decoder(bits);e=Encoder();tokens=[]
            for q in range(4):
                key=(q,tuple(tokens));assert key in memo,'unknown RX context'
                cdf=memo[key];v=d.decode(cdf);e.encode(v,cdf);tokens+=v.tolist()
            fc=coarse_latent(n,tokens);pr=model.probabilities(fc)[0].cpu().numpy();values,depth=decode_residual(d,e,j,pr,r['order'])
            assert np.array_equal(e.finish(),bits)and d.position-len(bits)==30
            return fc,torch.from_numpy(values)[None].cuda(),torch.from_numpy(depth)[None].cuda()
        rows=[];bitsrows=[];native=[];refs=[];reused=0
        with torch.no_grad():
            for s in sources:
                assert time.monotonic()-begin<r['seconds']and not(L/'STOP').exists();sid=s['source_id'];folder=out/str(s['source_index']);folder.mkdir()
                full,fc,target,tokens=data.sample(s);full=full[None].cuda();fc=fc[None].cuda();target=target[None].cuda()
                ref=(target[0].cpu().numpy()*255).round().astype(np.uint8);rp=folder/'reference.npz'
                with rp.open('xb')as f:np.savez(f,pixels=ref)
                refs.append(dict(source_id=sid,source_index=s['source_index'],archive=pin(rp),reference_sha256=asha(ref.astype(np.float32)/np.float32(255))))
                def encode():
                    u=model.encode(full,fc);pr=model.probabilities(fc);v=torch.floor((u+1)*8).clamp(0,15).long()[0].cpu().numpy()
                    with np.load(checked(s['coarse_stream']),allow_pickle=False)as z:tables=[z['cdf'+str(q)]for q in range(4)]
                    streams,emitted=encode_candidates(Encoder,[tokens[OFFSETS[q]:OFFSETS[q+1]].numpy()for q in range(4)],tables,v,pr[0].cpu().numpy(),r['order'])
                    predicted=model.estimated_bits(u,pr,256,all_prefixes=True)[0].cpu().numpy()
                    with(folder/'streams.npz').open('xb')as f:np.savez(f,**{str(j):v for j,v in streams.items()})
                    save(folder/'code_diagnostic.json',dict(source_id=sid,indices=v.tolist(),index_sha256=asha(v),unique_scalar_values=int(len(np.unique(v))),conditional_argmax_fraction=float(np.mean(v==pr[0].argmax(-1).cpu().numpy())),predicted_full_residual_bits=float(predicted[256]),actual_full_minus_coarse_bits=len(streams[256])-len(streams[0])))
                    return streams,emitted,predicted
                streams,emitted,predicted=ledger.call('source_encode',sid,encode)
                chosen={b:max((j for j,w in streams.items()if len(w)<=b),default=None)for b in plan['budgets']}
                for b,j in chosen.items():assert j==old[(sid,b)]['j']and (len(streams[j])if j is not None else 0)==old[(sid,b)]['actual_bits']
                for j in range(257):bitsrows.append(dict(source_id=sid,source_index=s['source_index'],order=r['order'],step=cp['step'],j=j,actual_total_bits=len(streams[j]),predicted_residual_bits=float(predicted[j]),actual_delta_finished_bits=len(streams[j])-len(streams[0]),emitted_bits=emitted[j],termination_bits=len(streams[j])-emitted[j],previous_endpoint_gap=0 if j==0 else len(streams[j])-len(streams[j-1])))
                scores={}
                for j in dict.fromkeys(plan['anchors']+list(chosen.values())):
                    def render_one():
                        if j is None:im=torch.full_like(target,.5)
                        else:
                            rxfc,q,depth=receive(streams[j],j);im=dc(model.decode(q,depth,rxfc))
                        a=im[0].cpu().numpy();assert a.dtype==np.float32 and np.isfinite(a).all()and a.min()>=0 and a.max()<=1
                        ip=folder/('j'+str(j)+'.npz')
                        with ip.open('xb')as f:np.savez(f,image=a)
                        mse=float((im.double()-target.double()).square().mean())
                        return a,im,mse,dict(source_id=sid,source_index=s['source_index'],order=r['order'],j=j,image=pin(ip),pixel_sha256=asha(a))
                    a,im,mse,entry=ledger.call('receive_materialize',sid+'|'+str(j),render_one);native.append(entry)
                    existing=next((old[(sid,b)]for b,v in chosen.items()if v==j),None)
                    if existing is not None:
                        assert mse==existing['mse'],'materialized pixels do not match saved-score MSE';pl=existing['lpips_alex'];reused+=1
                    else:pl=ledger.call('anchor_lpips',sid+'|'+str(j),lambda:float(lp(im*2-1,target*2-1).mean()))
                    scores[j]=dict(mse=mse,psnr_db=float(-10*np.log10(mse)),lpips_alex=pl,D=mse+.1*pl,actual_bits=len(streams[j])if j is not None else 0,pixel_sha256=entry['pixel_sha256'],image=entry['image']['path'],status='PAID_NULL'if j is None else'LOSSLESS_RX',lpips_reused=existing is not None)
                for j in plan['anchors']:rows.append(dict(source_id=sid,source_index=s['source_index'],order=r['order'],step=cp['step'],kind='anchor',budget=0,j=j,**scores[j]))
                for b,j in chosen.items():rows.append(dict(source_id=sid,source_index=s['source_index'],order=r['order'],step=cp['step'],kind='budget',budget=b,j=j,**scores[j]))
                save(folder/'commit.json',dict(source_id=sid,rows=[x for x in rows if x['source_id']==sid]));print('DIAGNOSTIC_NATIVE',r['order'],s['source_index'],flush=True)
        csvsave(out/'source_rd.csv',rows);csvsave(out/'predicted_bits_vs_actual_bits.csv',bitsrows);save(out/'native_index.json',native);save(out/'references.json',refs)
        assert statehash(dc)==n.loaded['identity']['models']['decoder']
        save(out/'completion.json',dict(status='COMPLETE_DIAGNOSTIC_MATERIALIZATION',checkpoint=r['checkpoint'],step=cp['step'],order=r['order'],rows=pin(out/'source_rd.csv'),bits=pin(out/'predicted_bits_vs_actual_bits.csv'),native=pin(out/'native_index.json'),references=pin(out/'references.json'),saved_lpips_reused=reused,ledger=ledger.summary(),seconds=time.monotonic()-begin,wireless_frames=0))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
