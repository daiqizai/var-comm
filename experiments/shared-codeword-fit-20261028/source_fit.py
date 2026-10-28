"""32-source new fine endpoints; preserved per-source commit, no retries."""
from fit_common import *
from fit_protocol import endpoints,pack_compact,parse_compact
from fit_codec import FitCodec,OFFSETS
import traceback,tempfile
def main(path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    out=Path(r['output']);out.mkdir(exist_ok=False);(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
    os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic()
    def boundary():assert time.monotonic()-begin<r['seconds'] and not(L/'STOP').exists()
    try:
        sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'))
        from kn_native import load
        n,_=ledger.call('model_load','once',lambda:load(cr(r['model_routes']),ledger,boundary,out,r['pci']))
        codec=FitCodec(R,native=n);rows=[];accounting=[]
        for s in cr(r['inputs']):
            boundary();dest=out/('%04d'%s['source_index']);dest.mkdir()
            with np.load(checked(s['archive']),allow_pickle=False)as z:tokens=z['tokens'].copy()
            def encode():
                streams,tables,trace=codec.encode_fit(tokens)
                with(dest/'streams.npz').open('xb')as f:np.savez(f,**streams)
                with(dest/'tx_cdfs.npz').open('xb')as f:np.savez(f,**tables)
                save(dest/'trace.json',trace)
                with np.load(checked(s['nest_streams']),allow_pickle=False)as old:
                    for key in set(old.files)&set(streams):assert np.array_equal(streams[key],old[key]),key
                item=dict(source_id=s['source_id'],source_index=s['source_index'],streams=pin(dest/'streams.npz'),tx_cdfs=pin(dest/'tx_cdfs.npz'),trace=pin(dest/'trace.json'),lengths={k:len(v)for k,v in streams.items()},tokenizer_calls=0)
                save(dest/'source_commit.json',item);return item
            row=ledger.call('source_probability_pass',s['source_id'],encode);rows.append(row)
            lengths=row['lengths'];fitting=[(m,K)for m,K in endpoints()if lengths[f'm{m}_K{K}']<=608]
            selected=max(fitting,key=lambda x:OFFSETS[x[0]]+x[1])if fitting else None
            fits=[K for K in range(26)if lengths['m4_K0']+lengths[f'K{K}_A']+lengths[f'K{K}_B']<=542]
            compact=max(fits)if fits else None
            for mode,target in [('S_FILL',selected),('COMPACT_TOKEN',None if compact is None else(4,compact)),('COMPACT_FIXED',(4,12))]:
                if target is None:accounting.append(dict(source_id=s['source_id'],source_index=s['source_index'],mode=mode,m=4,K=0,source_bits=0,fields=94 if mode.startswith('COMPACT')else 29,padding=0,status='PAID_NULL'));continue
                m,K=target;b=lengths[f'm{m}_K{K}']if mode=='S_FILL'else lengths['m4_K0']+lengths[f'K{K}_A']+lengths[f'K{K}_B']
                fields=29 if mode=='S_FILL'else 94;k=637 if mode=='S_FILL'else 636
                accounting.append(dict(source_id=s['source_id'],source_index=s['source_index'],mode=mode,m=m,K=K,source_bits=b,fields=fields,padding=k-fields-b,status='FITS'if b+fields<=k else'PAID_NULL'))
            print('FINE_SOURCE_COMMITTED',s['source_index'],len(rows),flush=True)
        save(out/'sources.json',rows);csvsave(out/'source_accounting.csv',accounting)
        save(out/'completion.json',dict(status='SOURCE_LENGTHS_COMPLETE_RX_PENDING',sources=pin(out/'sources.json'),accounting=pin(out/'source_accounting.csv'),ledger=ledger.summary(),seconds=time.monotonic()-begin))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
