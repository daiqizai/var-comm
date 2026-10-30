"""Original ordered A0/B0 intervention on frozen D received tokens only."""
from followup_common import *
from opt_common import definitions,SIZES,OFFSETS
import tempfile,traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 jobs=cr(r['jobs']);routes=cr(r['model_routes']);donor=cr(r['donor_identity'])
 for j in jobs:
  a=np.asarray(j['received_tokens'],np.int64);assert asha(a)==j['received_tokens_sha256']and len(a)==OFFSETS[j['m']]+j['K']
  with np.load(checked(j['donor_image']),allow_pickle=False)as z:assert asha(z['image'])==j['donor_pixel_sha256']and np.array_equal(a,z['received_tokens'])
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp');os.sched_setaffinity(0,set(r['cpu_affinity']))
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps'])
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('Intervention bounded stage')
 try:
  sys.path[:0]=[str(R/'src'),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
  from runtime_boundary import firewall
  firewall(out,[v['path']for v in routes['files']+routes['author_files']]+[routes['dist']['path']],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  from kn_native import load
  from receiver_variants import completed_latent,decode_image
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']))
  identity=read(out/'actual_native_identity.json');assert all(identity[k]==donor[k]for k in ['models','runtime','numeric_flags','source_mathematics','sampling','GPU_name'])
  ns=dict(np=np,SIZES=SIZES,OFFSETS=OFFSETS);definitions(R/'experiments/adatoken-response-20261026/b_ablation.py',{'merge_boundary','ablate'},ns)
  output=[];proofs=[]
  with n.torch.no_grad():
   for j in jobs:
    boundary();a=np.asarray(j['received_tokens'],np.int64);key=j['key']
    if key in r['qualification_keys']:
     def qualify():
      lat,_=completed_latent(n,a,j['m'],j['K']);im=decode_image(n,lat);assert asha(im)==j['donor_pixel_sha256'],'Same-route frozen D replay mismatch'
      return dict(key=key,exact_C0_pixel_sha256=asha(im),new_C0_for_qualification_only=True)
     proofs.append(ledger.call('qualification_C0',key,qualify))
    def receive():
     images,tr=ns['ablate'](n,a,j['m'],j['K']);proof=[]
     for v in tr:
      if v['scale']!=j['m']:assert np.array_equal(v['context_tokens'],v['latent_tokens'])
      else:assert np.array_equal(v['latent_tokens'][:j['K']],a[OFFSETS[j['m']]:])
      proof.append(dict(scale=v['scale'],context_sha256=asha(v['context_tokens']),latent_tokens_sha256=asha(v['latent_tokens'])))
     rows=[]
     for arm,image in images.items():
      assert image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()and 0<=image.min()<=image.max()<=1
      p=out/'images'/(key+'_'+arm+'.npz')
      with p.open('xb')as f:np.savez(f,image=image,received_tokens=a);f.flush();os.fsync(f.fileno())
      row=dict(key=key,arm=arm,image=pin(p),pixel_sha256=asha(image),received_tokens_sha256=asha(a),m=j['m'],K=j['K'],origin='NEW_ORDERED_INTERVENTION',new_PHY=0,TX_truth_used=False)
      save(p.with_suffix('.json'),row);rows.append(row)
     save(out/'images'/(key+'_trajectory.json'),proof);return rows
    output.extend(ledger.call('A0_B0_pair',key,receive))
    if len(output)%40==0:print('INTERVENTION',len(output),2*len(jobs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',output);save(out/'qualification.json',proofs);save(out/'completion.json',dict(status='COMPLETE',unique_received_inputs=len(jobs),new_A0_B0_images=len(output),qualification_C0_renders=len(proofs),new_PHY=0,source_truth_used=False,seconds=time.monotonic()-start,ledger=ledger.summary()))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
