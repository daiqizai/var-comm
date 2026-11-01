"""Explicitly authorized two-hour extension; unrelated methods do not gate success."""
from followup_common import *
from launch import launch
import subprocess,traceback

def main(request,expected):
 assert sha(request)==expected
 r=read(request);[checked(v) for v in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']))
 began=time.monotonic();launched=[];finished=[];failures=[]
 def remaining():return min(r['seconds']-(time.monotonic()-began),r['deadline_unix']-time.time())
 def check():
  if remaining()<=100 or (L/'STOP').exists():raise BudgetStop('Authorized two-hour extension boundary')
 def wait(role,owned):
  c=L/(role+'_control')
  while not (c/'actual_wait.json').exists():check();time.sleep(3)
  w=read(c/'actual_wait.json');assert ident(w['child']['pid']) is None
  owner=read(c/'owner.json')['owner']
  while True:
   check();live=ident(owner['pid'])
   if live is not None:assert live['start_ticks']==owner['start_ticks']
   if not owned:
    if live is None:break
   else:
    try:got,status=os.waitpid(owner['pid'],os.WNOHANG)
    except ChildProcessError:
     assert live is None;break
    if got:
     save(out/('owner_wait_'+role+'.json'),dict(actual_wait=True,returncode=os.waitstatus_to_exitcode(status)))
     assert os.waitstatus_to_exitcode(status)==w['returncode'];break
   time.sleep(.2)
  assert w['returncode']==0,('Stage failed; no retry',role)
  return closed(role)
 try:
  scope=cr(r['scope']);assert scope['group_indices']==[12,9] and scope['seconds_cap']==7200
  for item in scope['preserved_failures'].values():
   checked(item['failure']);w=cr(item['wait']);assert w['returncode']==1 and ident(w['child']['pid']) is None
   assert ident(cr(item['owner'])['owner']['pid']) is None
  for role,item in scope['reuse_completed'].items():
   done=closed(role);assert done['rows']==160 and done['ledger']['unresolved']==0
   assert pin(L/role/'rows.json')==item['rows']
  plan=Path(r['plan']);cfg=read(plan/'config.json');groups=read(plan/'groups.json')
  for gi in [12,9]:
   try:
    check();is_arpc=gi==9;base=cr(r['ARPC_base'] if is_arpc else r['BPG_base'])
    if is_arpc:
     qualification=wait('E5_ARPC_stdlib_preflight_v1',False)
     assert qualification['model_calls']==qualification['PHY_calls']==qualification['timing_calls']==0
     stat=qualification['statistics_stdlib'];checked(stat)
    else:
     qualification=closed('E5_BPG_two_core_preflight_v3')
     assert pin(L/'E5_BPG_two_core_preflight_v3/completion.json')==base['qualification']
     assert qualification['engineering_encode_calls']==qualification['engineering_decode_calls']==2
    for direction in ['tx','rx']:
     check();prefix='E5_arpc3_g09_' if is_arpc else 'E5_bpg2_g12_';role=prefix+direction
     assert not (D/(role+'_registration.json')).exists() and not (L/role).exists() and not (L/(role+'_control')).exists()
     while True:
      check()
      gs=[x.split(', ') for x in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
      g=next(x for x in gs if x[0]==cfg['GPU_UUID'])
      apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
      if int(g[1])>=60000 and int(g[2])<=50 and cfg['GPU_UUID'] not in apps:break
      time.sleep(5)
     code=r['ARPC_worker_code'] if is_arpc else base['worker_code']
     req=dict(code=code,config=pin(plan/'config.json'),group=groups[gi],direction=direction,
      sources=pin(plan/'sources.json'),cpu_affinity=cfg['cpu_affinity'],sionna_package=base['sionna_package'],
      device_mapping=base['device_mapping'],extension_scope=r['scope'],training=0,scientific_quality_results=0,automatic_retry=False)
     if is_arpc:req.update(qualification=base['qualification'],ARPC_scope=base['scope'],statistics_stdlib=stat,
      stdlib_qualification=pin(L/'E5_ARPC_stdlib_preflight_v1/completion.json'))
     else:req.update(BPG_two_core_qualification=base['qualification'],preserved_failed_role=base['preserved_failed_role'])
     if direction=='tx':req['expected_TX']=pin(plan/'expected_TX.json')
     else:req['received_symbols']=pin(L/(prefix+'tx')/'received_symbols.json')
     worker=E/('timing_arpc_stdlib_v3.py' if is_arpc else 'timing_bpg_two_core_v1.py')
     python=Path('/usr/bin/python3') if is_arpc else RT/'envs/frozen_ldpc_py311/bin/python'
     launch(role,req,worker,python,remaining()-90,gpu=True,required_uuid=cfg['GPU_UUID'],timing=True)
     launched.append(role);save(out/('launch_'+role+'.json'),dict(role=role,time=time.time()))
     done=wait(role,True);assert done['rows']==160 and done['ledger']['unresolved']==0
     if is_arpc:
      service=read(L/role/'phy_service_actual_wait.json');assert service['returncode']==0 and service['final_identity'] is None
      assert service['report']['result']['TX' if direction=='tx' else 'RX']==160
    finished.append(gi)
   except BaseException:
    failure=dict(group_index=gi,traceback=traceback.format_exc(),launched=list(launched),automatic_retry=False)
    failures.append(failure);save(out/f'group_{gi:02d}_failure.json',failure)
    # Continue only after all owned GPU stages really exited. Never hide active errors.
    for role in launched:
     c=L/(role+'_control');assert (c/'actual_wait.json').exists()
     assert ident(read(c/'actual_wait.json')['child']['pid']) is None
     assert ident(read(c/'owner.json')['owner']['pid']) is None
  if failures:
   save(out/'failure.json',dict(groups_finished=finished,failures=failures,launched=launched,automatic_retry=False))
   raise RuntimeError('Some registered groups remain incomplete; see preserved group failures')
  save(out/'completion.json',dict(status='COMPLETE',groups_finished=finished,launched=launched,rows=640,automatic_retry=False))
 except BaseException:
  if not (out/'failure.json').exists():save(out/'failure.json',dict(traceback=traceback.format_exc(),groups_finished=finished,launched=launched,automatic_retry=False))
  raise

if __name__=='__main__':main(*sys.argv[1:])
