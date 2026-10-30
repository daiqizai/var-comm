"""One deterministic original receiver per input; TX images cannot be opened."""
from opt_common import *
from receiver_variants import completed_latent,decode_image
import traceback,tempfile

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 jobs=cr(r['jobs']);reuse=cr(r['reuse']);routes=cr(r['model_routes']);static=r['static']
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'));os.sched_setaffinity(0,set(r['cpu_affinity']))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('Original receiver stage cap')
 try:
  sys.path[:0]=[str(R/'src'),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
  from runtime_boundary import firewall
  allowed=[v['path']for v in routes['files']+routes['author_files']]+[routes['dist']['path'],static['table']['path'],static['completion']['path']]+[x['image']['path']for x in reuse.values()if x['image']is not None]
  firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  from kn_native import load,receiver_codec
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']))
  if r.get('expanded_grid'):
   from grid_codec import codec as make_codec
   codec=make_codec(n)
  else:codec=receiver_codec(n)
  from h800_ep48_link_core_v1 import bind_static
  from t1_codec_runtime import InvalidSourceStream
  bind_static(codec,static);rows=[]
  with n.torch.no_grad():
   for j in jobs:
    boundary();key=j['key'];p=j['projection'];v=reuse.get(key)
    if v is not None:
     if v['image']is not None:
      with np.load(checked(v['image']),allow_pickle=False)as z:image=z['image'].copy();tokens=z['received_tokens'].copy()
      assert asha(image)==v['pixel_sha256']and asha(tokens)==v['received_tokens_sha256']
      assert p['status']=='SOURCE_INPUT'and tokens.shape==(OFFSETS[p['m']]+p['K'],)
     rows.append(dict(v,variant='original',origin='EXACT_CACHED_RECEIVER'));continue
    tokens=np.empty(0,np.int64);reason=p.get('reason');parsed=None
    if p['status']=='SOURCE_INPUT':
     assert p['codec']=='NEST'
     def decode():
      try:return codec.decode(p['family'],np.asarray(p['bits'],np.uint8),p['m'],p['K'])
      except InvalidSourceStream as exc:return dict(rejected=True,reason=str(exc))
     parsed=ledger.call('source_decode',key,decode)
     if parsed.get('rejected'):reason='SOURCE_DECODE_REJECT: '+parsed['reason']
     else:
      assert parsed['canonical']and not parsed['receiver_truth_used']and not parsed['transmitter_probability_table_used']
      tokens=np.asarray(parsed['received_tokens'],np.int64)
    if p['status']!='SOURCE_INPUT'or parsed.get('rejected'):
     image=np.full((3,256,256),.5,np.float32);row=dict(key=key,variant='original',image=None,pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),status='FIXED_GRAY',reason=reason,origin='FIXED_GRAY')
    else:
     lat,meta=ledger.call('completion_forward',key,lambda:completed_latent(n,tokens,p['m'],p['K']))
     image=ledger.call('render',key,lambda:decode_image(n,lat));path=out/'images'/(key+'.npz')
     with path.open('xb')as f:np.savez(f,image=image,received_tokens=tokens)
     row=dict(key=key,variant='original',image=pin(path),pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),status='ACCEPTED_SOURCE',origin='NEW_BASELINE',**meta)
    rows.append(row)
    if len(rows)%50==0:print('ORIGINAL_RX',len(rows),len(jobs),flush=True)
  assert len(rows)==len(jobs)and ledger.summary()['unresolved']==0
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE',unique_inputs=len(jobs),reused=sum(v['origin']=='EXACT_CACHED_RECEIVER'for v in rows),gray=sum(v['status']=='FIXED_GRAY'for v in rows),ledger=ledger.summary(),seconds=time.monotonic()-start,new_PHY=0,TX_truth_used=False))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
