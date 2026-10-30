"""Frozen C checkpoint comparisons using only accepted paid received tokens."""
from opt_common import *
from receiver_variants import completed_latent,decode_image
import traceback,tempfile

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 jobs=cr(r['jobs']);reuse=cr(r['reuse']);routes=cr(r['model_routes']);static=r['static'];checkpoints=r['checkpoints']
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'boundary_diagnostics').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'));os.sched_setaffinity(0,set(r['cpu_affinity']))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('C receiver cumulative route remainder')
 try:
  sys.path[:0]=[str(R/'src'),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
  from runtime_boundary import firewall
  allowed=[v['path']for v in routes['files']+routes['author_files']]+[routes['dist']['path'],static['table']['path'],static['completion']['path']]+[x['image']['path']for x in reuse.values()if x['image']is not None]+[v['checkpoint']['path']for v in checkpoints]
  firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  from kn_native import load
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));torch=n.torch
  # The frozen VAR also calls PyTorch SDPA: do NOT change its global backend.
  # Restrict math attention only inside the new corrector's forward call.
  torch.backends.mha.set_fastpath_enabled(False)
  from c_model import Corrector
  class MathCorrector(torch.nn.Module):
   def __init__(self,inner):super().__init__();self.inner=inner;self.last=None
   def forward(self,*args,**kwargs):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):result=self.inner(*args,**kwargs)
    self.last=dict(base=args[1][0].detach().cpu().numpy().copy(),corrected=result[0].detach().cpu().numpy().copy())
    return result
  models={}
  for v in checkpoints:
   def load_corrector():
    cp=torch.load(checked(v['checkpoint']),weights_only=True,map_location='cpu')
    assert cp['arm']==v['arm']and cp['step']==v['step']and cp['seed']==v['seed']and cp['parameter_count']==2899456
    assert cp['numerics']==dict(dtype='FP32',TF32=False,attention='math',torch='2.11.0+cu128')
    model=Corrector().float().to(n.loaded['device']);model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False);return MathCorrector(model).eval()
   models[v['variant']]=ledger.call('corrector_load',v['variant'],load_corrector)
  rows=[]
  with torch.no_grad():
   for j in jobs:
    boundary();key=j['key'];p=j['projection'];base=reuse[key]
    if base['image']is not None:
     with np.load(checked(base['image']),allow_pickle=False)as z:image=z['image'].copy();tokens=z['received_tokens'].copy()
     assert asha(image)==base['pixel_sha256']and asha(tokens)==base['received_tokens_sha256']
    else:tokens=np.empty(0,np.int64)
    rows.append(dict(base,variant='original',origin='EXACT_CACHED_RECEIVER'))
    eligible=p['status']=='SOURCE_INPUT'and base['status']=='ACCEPTED_SOURCE'and p['m']in(7,8)and 0<p['K']<SIZES[p['m']]**2
    for v in checkpoints:
     name=v['variant']
     if not eligible:rows.append(dict(base,variant=name,origin='EXACT_BYPASS',corrector_forwards=0));continue
     lat,meta=ledger.call('completion_forward',key+':'+name,lambda:completed_latent(n,tokens,p['m'],p['K'],corrector=models[name],no_content=v['arm']=='no_content'))
     assert meta['corrector_forwards']==1 and meta['received_tokens_unchanged']
     diagnostic=models[name].last;assert diagnostic is not None;models[name].last=None
     assert np.array_equal(diagnostic['base'][:p['K']],diagnostic['corrected'][:p['K']])
     basepath=out/'boundary_diagnostics'/(key+'_base.npy')
     if not basepath.exists():
      with basepath.open('xb')as f:np.save(f,diagnostic['base'],allow_pickle=False)
     else:
      with basepath.open('rb')as f:assert np.array_equal(np.load(f,allow_pickle=False),diagnostic['base'])
     newpath=out/'boundary_diagnostics'/(key+'_'+name+'.npy')
     with newpath.open('xb')as f:np.save(f,diagnostic['corrected'],allow_pickle=False)
     image=ledger.call('render',key+':'+name,lambda:decode_image(n,lat));path=out/'images'/(key+'_'+name+'.npz')
     with path.open('xb')as f:np.savez(f,image=image,received_tokens=tokens)
     rows.append(dict(key=key,variant=name,image=pin(path),pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),status='ACCEPTED_SOURCE',origin='NEW_C_VARIANT',checkpoint=v['checkpoint'],boundary_logits=dict(base=pin(basepath),corrected=pin(newpath),m=p['m'],K=p['K'],target_supplied_to_RX=False),**meta))
    if len(rows)%100==0:print('C_RX',len(rows),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE',unique_inputs=len(jobs),variants=[v['variant']for v in checkpoints],ledger=ledger.summary(),seconds=time.monotonic()-start,new_PHY=0,new_training=0,source_truth_used=False))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
