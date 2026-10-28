"""Bounded failed-cache diagnosis; preserve batch tensor, derive RX independently."""
from residual_runtime import *
import torch,traceback
def main(path,expected):
    assert sha(path)==expected;r=read(path);[checked(d)for d in r['code']];out=pathlib.Path(r['output']);out.mkdir(exist_ok=False);ledger=Ledger(out/'calls.sqlite',r['caps']);os.sched_setaffinity(0,set(r['cpu_affinity']))
    try:
        n,dc,statehash=ledger.call('model_load','once',lambda:load_frozen(r,out));codec=FitCodec(R,native=n)
        with np.load(checked(r['coarse_stream']),allow_pickle=False)as z:bits=z['bits'].copy()
        def decode():
            d=codec._decoder(bits);e=codec.Encoder();tokens=[];tables={}
            with codec._inference('EC_VAR_WHOLE'),codec._context('EC_VAR_WHOLE')as p:
                for scale in range(4):
                    table=codec._table(p,scale,[]);tables['cdf'+str(scale)]=table;v=d.decode(table);e.encode(v,table);tokens+=v.tolist();p.advance(v)
            assert np.array_equal(e.finish(),bits)and d.position-len(bits)==30
            fc=coarse_latent(n,tokens).cpu();torch.save(dict(coarse=fc,tokens=tokens),out/'rx_coarse.pt')
            with(out/'rx_cdfs.npz').open('xb')as f:np.savez(f,**tables)
            return fc,tokens
        fc,tokens=ledger.call('independent_coarse_RX_diagnostic','first_failed_source',decode)
        cached=torch.load(checked(r['batch_cache']),map_location='cpu',weights_only=True);i=r['within_batch'];assert tokens==cached['tokens'][i,:OFFSETS[4]].tolist()
        delta=float((fc-cached['coarse'][i:i+1]).abs().max());assert torch.isfinite(fc).all()
        single=ledger.call('single_TX_coarse_diagnostic','first_failed_source',lambda:coarse_latent(n,cached['tokens'][i,:OFFSETS[4]].tolist()).cpu())
        save(out/'diagnosis.json',dict(source_id=cached['source_ids'][i],tokens_exact=True,batch_vs_single_max_abs=delta,single_TX_RX_exact=torch.equal(fc,single),batch_cache=r['batch_cache'],rx_coarse=pin(out/'rx_coarse.pt'),rx_cdfs=pin(out/'rx_cdfs.npz'),new_training=0))
        assert torch.equal(fc,single)
        save(out/'completion.json',dict(status='NUMERIC_BATCH_DIAGNOSIS_COMPLETE',diagnosis=pin(out/'diagnosis.json'),ledger=ledger.summary()))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
