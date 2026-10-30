"""Exact replay from saved actual receiver projections; no source image access."""
from opt_common import *
import traceback,tempfile
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(d)for d in r['code']]
 donors=cr(r['donors']);routes=cr(r['model_routes']);static=r['static'];expected_arrays={}
 assert set(donors)=={'whole','partial','full','gray'}
 for typ,j in donors.items():
  if typ=='gray':continue
  with np.load(checked(j['donor']['image']),allow_pickle=False)as z:
   expected_arrays[typ]=(z['image'].copy(),z['received_tokens'].copy())
  assert asha(expected_arrays[typ][0])==j['donor']['pixel_sha256']
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
 os.sched_setaffinity(0,set(r['cpu_affinity']));start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps'])
 save(out/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),registration=pin(request)))
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('P0 elapsed cap or requested stop')
 try:
  sys.path[:0]=[str(R/'src'),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1'),str(R/'experiments/t1c-m10-test-20261016'),str(R/'experiments/t1-entropy-whole-20261009'),str(R/'experiments/entropy-partial-mismatch-20261010'),str(R/'experiments/submission-closure-20261016')]
  from runtime_boundary import firewall
  firewall(out,[d['path']for d in routes['files']+routes['author_files']]+[routes['dist']['path'],static['table']['path'],static['completion']['path']],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  from kn_native import load,receiver_codec
  n,render=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']))
  codec=receiver_codec(n)
  from h800_ep48_link_core_v1 import bind_static
  bind_static(codec,static)
  rows=[]
  with n.torch.no_grad():
   for typ in ['whole','partial','full','gray']:
    boundary();j=donors[typ];p=j['input']['projection']
    if typ=='gray':
     assert p['status']!='SOURCE_INPUT';image=np.full((3,256,256),.5,np.float32);tokens=np.empty(0,np.int64)
     assert asha(image)==j['expected_pixel_sha256']
    else:
     v=ledger.call('source_decode',typ,lambda:codec.decode(p['family'],np.asarray(p['bits'],np.uint8),p['m'],p['K']))
     assert v['canonical']and not v['receiver_truth_used']and not v['transmitter_probability_table_used']
     tokens=np.asarray(v['received_tokens'],np.int64)
     assert np.array_equal(tokens,expected_arrays[typ][1]),('received token mismatch',typ)
     image=ledger.call('render',typ,lambda:render(n,tokens,p['m'],p['K']))
     # Store an explicit numerical diagnostic even if exact acceptance fails.
     diff=np.abs(image-expected_arrays[typ][0]);save(out/(typ+'_comparison.json'),dict(exact=bool(np.array_equal(image,expected_arrays[typ][0])),max_abs_error=float(diff.max()),different_values=int(np.count_nonzero(diff)),new_sha=asha(image),old_sha=j['donor']['pixel_sha256']))
     assert np.array_equal(image,expected_arrays[typ][0]),('exact original replay mismatch',typ)
    rows.append(dict(kind=typ,pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),exact=True))
  save(out/'completion.json',dict(status='COMPLETE',rows=rows,ledger=ledger.summary(),seconds=time.monotonic()-start,new_PHY=0,new_training=0))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),seconds=time.monotonic()-start,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
