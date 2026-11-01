"""Qualified BPG two-core first completion; failed six-core guard retained."""
from followup_common import *
from launch import launch
import traceback,subprocess

def main(request,expected):
 assert sha(request)==expected
 r=read(request);[checked(x) for x in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']))
 beg=time.monotonic();launched=[]
 def rem():return min(r['seconds']-(time.monotonic()-beg),r['deadline_unix']-time.time())
 def check():
  if rem()<=90 or (L/'STOP').exists():raise BudgetStop('Original E5 deadline unchanged')
 def wait(role,owned):
  c=L/(role+'_control')
  while not (c/'actual_wait.json').exists():check();time.sleep(5)
  w=read(c/'actual_wait.json');assert ident(w['child']['pid']) is None
  owner=read(c/'owner.json')['owner']
  while True:
   check();v=ident(owner['pid'])
   if v is not None:assert v['start_ticks']==owner['start_ticks']
   if not owned:
    if v is None:break
   else:
    try:got,status=os.waitpid(owner['pid'],os.WNOHANG)
    except ChildProcessError:
     assert v is None;break
    if got:
     save(out/('owner_wait_'+role+'.json'),dict(actual_wait=True,returncode=os.waitstatus_to_exitcode(status)))
     assert os.waitstatus_to_exitcode(status)==w['returncode'];break
   time.sleep(.2)
  assert w['returncode']==0
  return closed(role)
 try:
  proof=wait('E5_BPG_two_core_preflight_v2',False)
  assert pin(L/'E5_BPG_two_core_preflight_v2/completion.json')==r['qualification']
  assert proof['engineering_encode_calls']==proof['engineering_decode_calls']==2 and proof['real_source_calls']==0
  prior=wait('E5_unlaunched_v7',False);assert prior['groups']==[13,14,15,9]
  plan=Path(r['plan']);cfg=read(plan/'config.json');g=read(plan/'groups.json')[12]
  assert g['method']=='BPG_LDPC'
  for direction in ['tx','rx']:
   check();role='E5_bpg2_g12_'+direction
   assert not (D/(role+'_registration.json')).exists() and not (L/role).exists()
   while True:
    check()
    gs=[x.split(', ') for x in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
    gpu=next(x for x in gs if x[0]==cfg['GPU_UUID'])
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if int(gpu[1])>=60000 and int(gpu[2])<=50 and cfg['GPU_UUID'] not in apps:break
    time.sleep(5)
   req=dict(code=r['worker_code'],config=pin(plan/'config.json'),group=g,direction=direction,
    sources=pin(plan/'sources.json'),cpu_affinity=cfg['cpu_affinity'],sionna_package=r['sionna_package'],
    device_mapping=r['device_mapping'],BPG_two_core_qualification=r['qualification'],
    preserved_failed_role=r['preserved_failed_role'],training=0,scientific_quality_results=0,automatic_retry=False)
   if direction=='tx':req['expected_TX']=pin(plan/'expected_TX.json')
   else:req['received_symbols']=pin(L/'E5_bpg2_g12_tx/received_symbols.json')
   launch(role,req,E/'timing_bpg_two_core_v1.py',RT/'envs/frozen_ldpc_py311/bin/python',rem()-60,gpu=True,required_uuid=cfg['GPU_UUID'],timing=True)
   launched.append(role);save(out/('launch_'+role+'.json'),dict(role=role,time=time.time()))
   done=wait(role,True);assert done['rows']==160 and done['ledger']['unresolved']==0
  save(out/'completion.json',dict(status='COMPLETE',launched=launched,rows=320,old_failure_preserved=True,automatic_retry=False))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),launched=launched,automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
