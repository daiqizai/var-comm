import sys,subprocess,math
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
old='final_successors';w=read(L/(old+'_control/actual_wait.json'));o=read(L/(old+'_control/owner.json'));f=read(L/old/'failure.json')
assert w['returncode']==1 and ident(w['child']['pid'])is None and ident(o['owner']['pid'])is None
assert f['completed']==['final_rx'] and f['traceback'].endswith('AssertionError: No eligible GPU at current resource admission; zero registration/launch\n')
for role in ['final_new_three','final_new_clip']:
 assert not(D/(role+'_registration.json')).exists() and not(D/(role+'_control_registration.json')).exists() and not(L/role).exists() and not(L/(role+'_dispatch.json')).exists()
pw=read(L/'final_new_metric_plan/completion.json');assert pw['status']=='PREPARED'and pw['logical_rows']==12000
ow=read(L/'final_rx_control/actual_wait.json');assert ow['returncode']==0 and ident(ow['child']['pid'])is None
deadline=w['started_unix']+14400;remaining=math.floor(deadline-time.time());assert remaining>1200
proof=D/'final_unlaunched_metric_admission.json'
save(proof,dict(old_actual_wait=pin(L/(old+'_control/actual_wait.json')),old_failure=pin(L/old/'failure.json'),old_completed=f['completed'],zero_metric_Popen=True,metric_registrations_absent=True,plan_reused=pin(L/'final_new_metric_plan/completion.json'),receiver_reused=pin(L/'final_rx/completion.json'),original_absolute_deadline=deadline,new_scope='First executions of previously unlaunched three/clip, audit, statistics only; no scientific retry; original absolute deadline retained',basis='User authorized all A-E through final delivery; known pre-registration admission gap, no altered scientific identity'))
role='final_unlaunched_metrics_v2';code=[pin(E/x)for x in ['opt_common.py','final_unlaunched_metrics_v2.py','launch_registered.py','launch_metric_stage_v2.py','metrics.py','audit_receiver_inputs_v2.py','final_statistics.py','route_statistics_v2.py','frame_plan.py','control_once.py']];rp=D/(role+'_registration.json')
save(rp,dict(code=code,output=str(L/role),seconds=remaining,original_absolute_deadline=deadline,cpu_affinity=[24,25],sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),freeze=pin(D/'frozen_final.json'),continuation_proof=pin(proof)))
cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'final_unlaunched_metrics_v2.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=remaining,gpu=False))
with(L/(role+'_owner.log')).open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid),'remaining_original_seconds',remaining)
