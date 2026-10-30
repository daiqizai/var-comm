"""Only shards 02/03, after exact successful predecessors; no polling admission race."""
from followup_common import *
from launch import launch
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];os.sched_setaffinity(0,set(r['cpu_affinity']))
 out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic();dispatched=[]
 try:
  fail=cr(r['predecessor_failure']);wait=cr(r['predecessor_wait'])
  assert fail['dispatched']==['E2_ada_01']and 'assert len(active)<2' in fail['traceback']and wait['returncode']==1 and ident(wait['child']['pid'])is None
  old=cr(r['old_registration']);assert[s['role']for s in r['shards']]==['E2_ada_02','E2_ada_03']
  for s in r['shards']:assert not(D/(s['role']+'_registration.json')).exists()and not(L/s['role']).exists()
  for s in r['shards']:
   pred=s['wait_for'];p=L/(pred+'_control/actual_wait.json')
   while not p.exists():
    if time.time()>=r['absolute_queue_deadline']or(L/'STOP').exists():raise BudgetStop('Original remaining queue window')
    cp=read(L/(pred+'_control/child.json'));now=ident(cp['child']['pid']);assert now and now['start_ticks']==cp['child']['start_ticks'];time.sleep(20)
   closed(pred)
   jobs=cr(s['jobs']);reuse=cr(s['reuse']);n=len(jobs)-len(reuse)
   req=dict(code=r['child_code'],jobs=s['jobs'],reuse=s['reuse'],weights=old['weights'],upstream_code=old['upstream_code'],cpu_affinity=s['cpu_affinity'],caps=dict(model_load=1,receive_render=n),seed_registration=r['seed_registration'],source_truth_allowed=False,training=False)
   launch(s['role'],req,E/'ada_information_receiver.py',RT/'envs/frozen_visual_py310/bin/python',min(43200,r['absolute_queue_deadline']-time.time()-30),gpu=True)
   dispatched.append(s['role']);save(out/('dispatched_'+s['role']+'.json'),dict(role=s['role'],time=time.time(),registration=pin(D/(s['role']+'_registration.json'))))
  while True:
   if time.time()>=r['absolute_queue_deadline']or(L/'STOP').exists():raise BudgetStop('Original remaining queue window')
   done=0
   for role in ['E2_ada_00','E2_ada_01','E2_ada_02','E2_ada_03']:
    if(L/(role+'_control/actual_wait.json')).exists():closed(role);done+=1
   if done==4:break
   time.sleep(20)
  save(out/'completion.json',dict(status='COMPLETE',first_launches=dispatched,already_launched_reused=['E2_ada_00','E2_ada_01'],all_actual_wait0=True,seconds=time.monotonic()-beg,new_scientific_calls_in_controller=0));print('ADA_ALL_SHARDS_CLOSED')
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),dispatched=dispatched,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
