"""First-only matched timing after every currently owned GPU stage has closed."""
from followup_common import *
from launch import launch
import traceback,subprocess

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']];out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));start=time.monotonic();launched=[]
 def check():
  if time.monotonic()-start>=r['seconds']-90 or(L/'STOP').exists():raise BudgetStop('Registered finite timing continuation window')
 def wait(role):
  control=L/(role+'_control');w=control/'actual_wait.json'
  while not w.exists():
   check();p=control/'child.json'
   if p.exists():
    c=read(p);live=ident(c['child']['pid'])
    if live is None:
     # Parent may be writing its actual wait between this read and exit.
     time.sleep(1)
     if not w.exists():raise RuntimeError('Child vanished without actual wait: '+role)
    else:assert live['start_ticks']==c['child']['start_ticks']
   time.sleep(20)
  closed(role)
 try:
  wait('completion_followup_queue')
  # No concurrent quality work during any timing boundary.
  for p in L.glob('*_control/owner.json'):
   reg=cr(read(p)['registration'])
   if reg['gpu']:
    w=read(p.parent/'actual_wait.json');assert ident(w['child']['pid'])is None
  plan=Path(r['plan']);cfg=read(plan/'config.json');groups=read(plan/'groups.json');assert len(groups)==16
  for gi,g in enumerate(groups):
   for direction in ['tx','rx']:
    check();role=f'E5_g{gi:02d}_{direction}'
    req=dict(code=r['worker_code'],config=pin(plan/'config.json'),group=g,direction=direction,sources=pin(plan/'sources.json'),cpu_affinity=cfg['cpu_affinity'],training=0,scientific_quality_results=0)
    if direction=='tx':req['expected_TX']=pin(plan/'expected_TX.json')
    else:req['received_symbols']=pin(L/f'E5_g{gi:02d}_tx/received_symbols.json')
    while True:
     check()
     gs=[v.split(', ')for v in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
     q=next(v for v in gs if v[0]==cfg['GPU_UUID'])
     apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
     if int(q[1])>=60000 and int(q[2])<=50 and cfg['GPU_UUID']not in apps:break
     time.sleep(20)
    python=RT/'envs/frozen_ldpc_py311/bin/python'if g['method']=='BPG_LDPC'else Path('/usr/bin/python3')if g['method']=='ARPC'else RT/'envs/frozen_visual_py310/bin/python'
    remaining=r['seconds']-(time.monotonic()-start)-60
    launch(role,req,E/'timing_end_to_end.py',python,min(43200,remaining),gpu=True,required_uuid=cfg['GPU_UUID'],timing=True);launched.append(role);wait(role)
  from summarize_e5 import main as summarize
  summarize(plan)
  save(out/'completion.json',dict(status='COMPLETE',launched=launched,groups=16,all_actual_wait0=True,seconds=time.monotonic()-start,new_training=0));print('E5_ALL_ACTUAL_CLOSED')
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),launched=launched,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
