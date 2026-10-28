"""Finite authorized stage progression; no retries, no publication, no claims."""
from n512_common import *
from pipeline_prepare import closed
from rx_lineage import registrations
import subprocess,traceback

def wait_roles(roles):
 start=time.monotonic()
 while True:
  assert time.monotonic()-start<43380 and not(L/'STOP').exists()
  pending=[]
  for role in roles:
   p=L/'controls'/role/'actual_wait.json'
   if not p.exists():pending.append(role);continue
   w=read(p);assert w['actual_wait']and w['returncode']==0 and not w['timed_out'],(role,w)
   if not gone(w['owner'])or not gone(w['process']):pending.append(role);continue
   closed(role)
  if not pending:return
  time.sleep(15)

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(d)for d in r['code']];assert sha(O/'protocol.json')==r['protocol']['sha256']
 root=L/'science_chain_v2';root.mkdir(exist_ok=False);save(root/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),request=pin(request)));beg=time.monotonic();steps=[]
 def execute(label,script,args=(),timeout=1800):
  assert not(L/'STOP').exists();[checked(d)for d in r['code']]
  c=root/label;c.mkdir(exist_ok=False);start=time.monotonic();argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/script),*args]
  with(c/'stdout.log').open('xb')as so,(c/'stderr.log').open('xb')as se:
   p=subprocess.Popen(argv,cwd=R,stdout=so,stderr=se);proc=ident(p.pid);save(c/'launch.json',dict(process=proc,argv=argv));rc=p.wait(timeout=timeout)
  save(c/'actual_wait.json',dict(actual_wait=True,returncode=rc,seconds=time.monotonic()-start,process=proc));steps.append(pin(c/'actual_wait.json'));assert rc==0,(label,(c/'stderr.log').read_text()[-4000:])
  print('STEP_COMPLETE',label,flush=True)
 def rx_roles(stage):return[Path(cr(p)['output']).name for p in registrations(stage)]
 try:
  # The eight existing pilot RX roles are prerequisites, never dispatched here.
  wait_roles(rx_roles('pilot'))
  execute('pilot_audit','audit_stage.py',['pilot'])
  execute('pilot_dino_dispatch','stage_operations_v5.py',['start_dino','pilot']);wait_roles(['pilot_dino'])
  execute('pilot_selection','select_calibration.py',['pilot'])
  for stage in ('full','formal'):
   execute(stage+'_plan','plan_stage.py',[stage,r['protocol']['sha256']])
   execute(stage+'_phy_dispatch','stage_operations_v5.py',['start_phy',stage]);wait_roles([f'{stage}_phy_{i:02d}'for i in range(8)])
   execute(stage+'_rx_dispatch','stage_operations_v5.py',['start_rx',stage]);wait_roles(rx_roles(stage))
   execute(stage+'_audit','audit_stage.py',[stage])
   if stage=='full':
    execute('full_dino_dispatch','stage_operations_v5.py',['start_dino','full']);wait_roles(['full_dino'])
    execute('final_policy_freeze','select_calibration.py',['full'])
  execute('formal_metric_dispatch','prepare_formal_metrics.py');wait_roles(['formal_three_metrics','formal_clip_metrics'])
  execute('statistics_dispatch','start_statistics.py');wait_roles(['formal_statistics'])
  assert read(O/'delivery/completion.json')['status']=='COMPLETE_N512_13500_LOGICAL_FRAMES'
  save(root/'completion.json',dict(status='SCIENCE_COMPLETE_REVIEW_AND_MANUSCRIPT_PENDING',steps=steps,seconds=time.monotonic()-beg,delivery=pin(O/'delivery/completion.json')))
 except BaseException:
  save(root/'failure.json',dict(traceback=traceback.format_exc(),steps=steps,automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
