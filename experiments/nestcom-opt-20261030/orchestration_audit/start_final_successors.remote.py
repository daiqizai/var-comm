import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
assert not(D/'final_new_three_registration.json').exists()and not(D/'final_new_metric_plan_registration.json').exists()
role='final_successors';code=[pin(E/x)for x in ['opt_common.py','final_successors.py','launch_registered.py','launch_metric_stage_v2.py','evaluation_metric_plan.py','metrics.py','audit_receiver_inputs_v2.py','final_statistics.py','route_statistics_v2.py','frame_plan.py','control_once.py']];rp=D/(role+'_registration.json')
save(rp,dict(code=code,output=str(L/role),seconds=14400,cpu_affinity=[24,25],sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),freeze=pin(D/'frozen_final.json'),responsibility='Only final_new three/clip, posthoc audit, final_statistics; do not launch manually while active'))
cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'final_successors.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=14430,gpu=False))
with(L/(role+'_owner.log')).open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
