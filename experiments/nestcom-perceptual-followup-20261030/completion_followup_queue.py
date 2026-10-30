"""Bounded first-only E2/E6 metric/statistical continuation after existing receivers."""
from followup_common import *
from launch import launch as gpu_launch
from launch_cpu_admission_v2 import launch as cpu_launch
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];os.sched_setaffinity(0,set(r['cpu_affinity']));out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic();launched=[]
 def check():
  if time.monotonic()-beg>=r['seconds']-60 or(L/'STOP').exists():raise BudgetStop('Bounded E2/E6 continuation window')
 def wait(role):
  path=L/(role+'_control/actual_wait.json')
  while not path.exists():
   check();cp=L/(role+'_control/child.json')
   if cp.exists():
    c=read(cp);live=ident(c['child']['pid']);assert live and live['start_ticks']==c['child']['start_ticks'],('Unexpected vanished child before actual wait',role)
   time.sleep(20)
  closed(role)
 def stats(exp):
  from_file=L/(exp+'_metric_plan/logical.json')
  modes=['three','clip','dists']if exp=='E2'else['dreamsim','dists']
  req=dict(code=r['code'],logical=pin(from_file),source_sets=pin(D/'source_sets.json'),cpu_affinity=[26,27],**{k:pin(L/(exp+'_'+k)/'scores.json')for k in modes})
  role=exp+'_statistics';cpu_launch(role,req,E/('summarize_'+exp.lower()+'.py'),RT/'envs/frozen_ldpc_py311/bin/python',1800);launched.append(role);wait(role)
 try:
  wait('E2_ada_unlaunched_queue_v2')
  for i in range(4):closed(f'E2_ada_{i:02d}')
  from prepare_e2_metrics import main as prepare_e2
  from launch_followup_metrics import main as metric
  prepare_e2();check()
  for mode in ['three','clip','dists']:
   metric('E2',mode);launched.append('E2_'+mode);wait('E2_'+mode)
  stats('E2')
  closed('E6_phy');p=L/'E6_metric_plan';assert read(p/'completion.json')['status']=='PREPARED'
  old=read(OLD/'final_new_three_registration.json');n=len(read(p/'pairs.json'))
  caches=[pin(OL/(x+'_three')/'completion.json')for x in ['fit_A','confirm_D','final_new']]+[pin(L/(x+'_three')/'completion.json')for x in ['E4','E3','E2']]
  req=dict(code=r['code']+old['code'],plan=pin(p/'plan.json'),metric_registration=old['metric_registration'],cpu_affinity=[22,23],caps=dict(load_dreamsim=1,embedding=n+100,pair=n),caches=caches)
  gpu_launch('E6_dreamsim',req,E/'dreamsim_only.py',RT/'envs/frozen_visual_py310/bin/python',3600,gpu=True,required_uuid='GPU-f5047136-08e5-9416-3c86-87786d0b355e');launched.append('E6_dreamsim');wait('E6_dreamsim')
  metric('E6','dists');launched.append('E6_dists');wait('E6_dists');stats('E6')
  save(out/'completion.json',dict(status='COMPLETE',first_launches=launched,all_actual_wait0=True,new_training=0,new_PHY=0,seconds=time.monotonic()-beg));print('E2_E6_STATISTICAL_CLOSURE')
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),launched=launched,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
