"""Reuse completed train/full caches; canonical batch-one Fc on both codec ends."""
from residual_runtime import *
import torch,traceback,collections
def main(path,expected):
    assert sha(path)==expected;r=read(path);[checked(x)for x in r['code']];out=pathlib.Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic()
    try:
        n,dc,statehash=ledger.call('model_load','once',lambda:load_frozen(r,out));codec=FitCodec(R,native=n);train=read(checked(r['train_records']));dev=read(checked(r['dev_manifest']))['sources'];prior=read(checked(r['dev_full_reuse']));diag=read(checked(r['diagnosis']));rows=[];loaded={}
        def load(d):
            if d['path']not in loaded:loaded[d['path']]=torch.load(checked(d),map_location='cpu',weights_only=True)
            return loaded[d['path']]
        for s in dev:
            assert time.monotonic()-begin<r['seconds']and not(L/'STOP').exists();sid=s['source_id'];i=s['within_shard'];old=load(s['old_latent_shard']);assert old['image_ids'][i]==sid;tokens=old['full_tokens'][i].numpy().astype(np.int64);dest=out/str(s['source_index']);dest.mkdir()
            if sid in prior:
                pp=prior[sid];full=load(pp['cache'])['full'][pp['row']:pp['row']+1].cuda()
                if sid==diag['source_id']:fc=load(diag['rx_coarse'])['coarse'].cuda()
                else:fc=ledger.call('canonical_TX_Fc',sid,lambda:coarse_latent(n,tokens[:OFFSETS[4]]))
            else:full,fc=ledger.call('new_direct_full_Fc',sid,lambda:direct_latents(n,tokens))
            if sid==diag['source_id']:wire=r['first_coarse_stream']
            else:
                def encode():
                    e=codec.Encoder();tables={}
                    with codec._inference('EC_VAR_WHOLE'),codec._context('EC_VAR_WHOLE')as p:
                        for scale in range(4):
                            table=codec._table(p,scale,[]);tables['cdf'+str(scale)]=table;v=tokens[OFFSETS[scale]:OFFSETS[scale+1]];e.encode(v,table);p.advance(v)
                    with(dest/'coarse_tx.npz').open('xb')as f:np.savez(f,**tables,bits=e.finish())
                    return pin(dest/'coarse_tx.npz')
                wire=ledger.call('coarse_TX',sid,encode)
            if sid==diag['source_id']:rxc=diag['rx_coarse'];rxcd=diag['rx_cdfs'];recovered=load(rxc)['tokens'];rxfc=load(rxc)['coarse']
            else:
                def receive():
                    with np.load(checked(wire),allow_pickle=False)as z:bits=z['bits'].copy()
                    d=codec._decoder(bits);e=codec.Encoder();v=[];tables={}
                    with codec._inference('EC_VAR_WHOLE'),codec._context('EC_VAR_WHOLE')as p:
                        for scale in range(4):
                            table=codec._table(p,scale,[]);tables['cdf'+str(scale)]=table;x=d.decode(table);e.encode(x,table);v+=x.tolist();p.advance(x)
                    assert np.array_equal(e.finish(),bits)and d.position-len(bits)==30
                    rxfc=coarse_latent(n,v).cpu();torch.save(dict(coarse=rxfc,tokens=v),dest/'rx_coarse.pt')
                    with(dest/'rx_cdfs.npz').open('xb')as f:np.savez(f,**tables)
                    return v,rxfc,pin(dest/'rx_coarse.pt'),pin(dest/'rx_cdfs.npz')
                recovered,rxfc,rxc,rxcd=ledger.call('independent_coarse_RX',sid,receive)
            assert recovered==tokens[:OFFSETS[4]].tolist()and torch.equal(rxfc,fc.cpu())
            torch.save(dict(full=full.cpu(),coarse=rxfc,tokens=torch.as_tensor(tokens)[None],source_ids=[sid]),dest/'latents.pt')
            row=dict(source_id=sid,source_index=s['source_index'],split='dev',rgb_shard=s['rgb_shard'],within_rgb=i,latents=pin(dest/'latents.pt'),within_latent=0,layout='batch',coarse_stream=wire,rx_coarse=rxc,rx_cdfs=rxcd,canonical_batch_one_Fc=True,full_reused=sid in prior)
            save(dest/'commit.json',row);rows.append(row)
            if len(rows)%20==0:print('CANONICAL_DEV_CACHE',len(rows),200,flush=True)
        assert len(train)==19968 and len(rows)==200
        save(out/'index.json',dict(sources=train+rows,train_sources=19968,dev_sources=200,training_Fc='Frozen batch-cache FP32; no wire encoding during optimization',evaluation_Fc='Canonical batch-one frozen operator; independently received tokens; exact TX/RX',revision=r['revision']))
        save(out/'completion.json',dict(status='CANONICAL_CACHE_COMPLETE',index=pin(out/'index.json'),training_cache_reused=19968,dev_sources=200,dev_full_reused=len(prior),ledger=ledger.summary(),seconds=time.monotonic()-begin))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
