"""Only actual received prefix projection and shared frozen models enter RX."""
from common import *
import argparse,subprocess,traceback
def main(role,path,expected):
 assert sha(path)==expected;r=read(path);assert role==r['role']
 for d in r['code']:checked(d)
 if role=='b_receiver':
  q=read(r['conformance_review_required']);assert q['status']=='PASS'and q['registration_sha256']==sha(O/'b_conformance_registration.json')
  w=read(L/'b_conformance_control/actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process'])and gone(w['parent_process'])
 q=read(L/'b_contract_qualification/actual_wait.json');assert type(q['returncode'])is int and q['returncode']==0
 out=L/role;out.mkdir(parents=True,exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,{0,1});beg=time.monotonic();last=[0.]
 save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 ledger=Ledger(out/'calls.sqlite',r['caps'])
 try:
  inputs=cr(r['inputs']);routes=cr(r['model_routes'])
  sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'))
  from kn_native import load
  from b_ablation import ablate
  from runtime_boundary import firewall
  allowed=[v['path']for v in routes['files']+routes['author_files']]+[routes['dist']['path']]
  # Exact code reads are allowed; all scientific-data roots stay blocked.
  firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  def boundary():
   assert time.monotonic()-beg<r['max_seconds']and not(L/'STOP').exists()
   if time.monotonic()-last[0]>10:
    assert int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>=2048;last[0]=time.monotonic()
  n,render=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,'00000000:A3:00.0'))
  import numpy as np
  (out/'images').mkdir();outputs=[]
  with n.torch.no_grad():
   for item in inputs:
    boundary();a=np.asarray(item['received_tokens'],dtype=np.int64)
    if role=='b_conformance':
     images={'C0':ledger.call('var_render',item['key'],lambda:render(n,a,item['m'],item['K']))};trajectory=[]
    else:
     images,trajectory=ledger.call('var_render',item['key'],lambda:ablate(n,a,item['m'],item['K']))
    for arm,image in images.items():
     assert image.shape==(3,256,256)and image.dtype==np.float32 and np.isfinite(image).all()and image.min()>=0 and image.max()<=1
     p=out/'images'/(item['key']+'_'+arm+'.npz')
     with p.open('xb')as f:np.savez(f,image=image);f.flush();os.fsync(f.fileno())
     output=dict(key=item['key'],arm=arm,m=item['m'],K=item['K'],received_tokens_sha256=asha(a),image=pin(p),image_sha256=asha(image))
     save(p.with_suffix('.json'),output);outputs.append(output)
    if trajectory:
     tp=out/'images'/(item['key']+'_trajectory.npz')
     with tp.open('xb')as f:np.savez(f,**{f'{s}_{k}':v[k]for s,v in enumerate(trajectory)for k in('context_tokens','latent_tokens')});f.flush();os.fsync(f.fileno())
    print('RECEIVED_INPUT_COMPLETE',len(outputs),len(inputs),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'outputs.json',outputs);save(out/'completion.json',dict(status='COMPLETE',role=role,registration_sha256=expected,outputs=pin(out/'outputs.json'),ledger=ledger.summary(),seconds=time.monotonic()-beg,source_reads=0,channel_calls=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
