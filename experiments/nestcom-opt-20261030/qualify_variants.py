"""Bounded real-weight baseline/zero-C/seed-path identity qualification."""
from opt_common import *
from receiver_variants import completed_latent,decode_image
import tempfile,traceback
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 donors=cr(r['donors']);routes=cr(r['model_routes']);examples={}
 for k in('whole','partial','full'):
  d=donors[k]
  with np.load(checked(d['donor']['image']),allow_pickle=False)as z:examples[k]=(d['input']['projection'],z['received_tokens'].copy(),z['image'].copy())
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp');os.sched_setaffinity(0,set(r['cpu_affinity']))
 os.environ.update(TMPDIR=str(out/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
 ledger=Ledger(out/'calls.sqlite',dict(model_load=1,prior_scale=50,decoder_forward=3,qualification_completion=5));start=time.monotonic()
 def boundary():assert time.monotonic()-start<r['seconds']and not(L/'STOP').exists()
 try:
  sys.path[:0]=[str(R/'src'),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
  from runtime_boundary import firewall
  firewall(out,[v['path']for v in routes['files']+routes['author_files']]+[routes['dist']['path']],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  from kn_native import load
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));torch=n.torch
  from c_model import Corrector
  torch.manual_seed(2026103001);c=Corrector().to('cuda').eval().requires_grad_(False);rows=[]
  with torch.no_grad():
   for k,(p,tokens,old)in examples.items():
    lat,meta=ledger.call('qualification_completion',k,lambda:completed_latent(n,tokens,p['m'],p['K'],corrector=c))
    image=decode_image(n,lat);assert np.array_equal(image,old),(k,float(np.abs(image-old).max()))
    rows.append(dict(kind=k,zero_corrector=True,exact_original=True,pixel_sha256=asha(image),**meta))
   p,tokens,_=examples['partial'];assert p['m']==7 and p['K']>0
   first,_=ledger.call('qualification_completion','B',lambda:completed_latent(n,tokens,p['m'],p['K'],.3,2026103001))
   second,meta=ledger.call('qualification_completion','B_zeroC',lambda:completed_latent(n,tokens,p['m'],p['K'],.3,2026103001,corrector=c))
   assert torch.equal(first,second)
   rows.append(dict(kind='B_and_zeroC_B',exact_latent=True,known_tokens_preserved=True,seed=2026103001,temperature=.3,**meta))
  save(out/'completion.json',dict(status='COMPLETE',rows=rows,ledger=ledger.summary(),seconds=time.monotonic()-start,image_quality_evaluation=False))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
