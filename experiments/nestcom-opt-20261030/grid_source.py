"""One incremental source traversal, all independently terminated legal endpoints."""
from opt_common import *
from grid_codec import codec,endpoints
import tempfile,traceback
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];sources=cr(r['sources']);routes=cr(r['model_routes'])
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'streams').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp');os.sched_setaffinity(0,set(r['cpu_affinity']))
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',dict(model_load=1,source_traversal=len(sources),prior_scale=10*len(sources),decoder_forward=0));save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('D source traversal limit')
 try:
  sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'));from kn_native import load
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));c=codec(n);rows=[];allends=endpoints('token');assert len(allends)==651
  for s in sources:
   boundary()
   with np.load(checked(s['archive']),allow_pickle=False)as z:tokens=z['tokens'].astype(np.int64)
   with np.load(checked(s['nest_streams']),allow_pickle=False)as z:old={k:z[k].copy()for k in z.files}
   beg=time.monotonic();v=ledger.call('source_traversal',s['source_id'],lambda:c.encode_endpoints(tokens,allends));seconds=time.monotonic()-beg
   streams={'m%d_K%d'%e:d['bits']for e,d in v.items()}
   assert len(streams)==651
   for key,bits in old.items():assert np.array_equal(streams[key],bits),('Original independently finished stream differs',s['source_id'],key)
   path=out/'streams'/('i%04d.npz'%s['source_index'])
   with path.open('xb')as f:np.savez_compressed(f,**streams)
   row=dict(source_id=s['source_id'],source_index=s['source_index'],source_pixels_sha256=s['preprocessing_id'],streams=pin(path),lengths={k:len(b)for k,b in streams.items()},array_sha256={k:asha(b)for k,b in streams.items()},independent_finish_count=651,source_seconds=seconds,original_streams=s['nest_streams'],original_all25_exact=True)
   save(path.with_suffix('.json'),row);rows.append(row)
   if len(rows)%10==0:print('D_SOURCE',len(rows),len(sources),flush=True)
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE',sources=len(rows),new_source_traversals=len(rows),independent_finish_calls=651*len(rows),original_quarter_streams_exact=True,ledger=ledger.summary(),seconds=time.monotonic()-start,new_tokenizer=0,new_PHY=0))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
