"""Finite first launches of three disjoint preregistered Ada shards; never retry."""
from followup_common import *
from launch import launch
import subprocess,traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];os.sched_setaffinity(0,set(r['cpu_affinity']))
 out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic();dispatched=[];old=cr(r['old_registration'])
 try:
  roles=['E2_ada_00']+[v['role']for v in r['shards']]
  assert roles==['E2_ada_00','E2_ada_01','E2_ada_02','E2_ada_03'];assert(D/'E2_ada_00_registration.json').exists()
  for spec in r['shards']:assert not(D/(spec['role']+'_registration.json')).exists()
  while True:
   if time.monotonic()-beg>=r['seconds']or(L/'STOP').exists():raise BudgetStop('Finite Ada successor queue boundary')
   active=[];complete=[]
   for p in L.glob('*_control/owner.json'):
    reg=cr(read(p)['registration'])
    if not reg['gpu']:continue
    role=p.parent.name.removesuffix('_control');wp=p.parent/'actual_wait.json'
    if wp.exists():
     w=read(wp)
     if role in roles:
      assert w['returncode']==0,(role,'unknown failure; no retry');closed(role);complete.append(role)
     continue
    cp=p.parent/'child.json'
    if not cp.exists():active.append(role);continue
    c=read(cp);now=ident(c['child']['pid']);assert now and now['start_ticks']==c['child']['start_ticks'];active.append(role)
   pending=[s for s in r['shards']if s['role']not in dispatched]
   if not pending and len(complete)==4:break
   if pending and len(active)<2:
    spec=pending[0];jobs=cr(spec['jobs']);reuse=cr(spec['reuse']);n=len(jobs)-len(reuse)
    gs=[v.split(', ')for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
    if not any(int(g[2])>=60000 and int(g[3])<=50 for g in gs):time.sleep(20);continue
    req=dict(code=r['child_code'],jobs=spec['jobs'],reuse=spec['reuse'],weights=old['weights'],upstream_code=old['upstream_code'],cpu_affinity=spec['cpu_affinity'],caps=dict(model_load=1,receive_render=n),seed_registration=r['seed_registration'],source_truth_allowed=False,training=False)
    launch(spec['role'],req,E/'ada_information_receiver.py',RT/'envs/frozen_visual_py310/bin/python',43200,gpu=True)
    dispatched.append(spec['role']);save(out/('dispatched_'+spec['role']+'.json'),dict(role=spec['role'],time=time.time(),registration=pin(D/(spec['role']+'_registration.json'))))
   else:time.sleep(20)
  save(out/'completion.json',dict(status='COMPLETE',dispatched=dispatched,all_shards_actual_wait0=True,seconds=time.monotonic()-beg,new_model_calls_in_controller=0));print('ADA_SHARDS_ALL_CLOSED')
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),dispatched=dispatched,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
