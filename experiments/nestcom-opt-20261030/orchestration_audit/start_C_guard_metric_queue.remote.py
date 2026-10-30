import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
role='C_guard_metric_queue';stages=[dict(role='fit_C_guard',mode='three',route='C',cache_roles=['fit_A','fit_C'],dependencies=['fit_C_guard','fit_metric_queue']),dict(role='fit_C_guard',mode='clip',route='C',cache_roles=['fit_A','fit_C'],dependencies=['fit_C_guard_three'])]
assert all(not(D/(s['role']+'_'+s['mode']+'_registration.json')).exists()for s in stages)
code=[pin(E/x)for x in ['opt_common.py','metric_queue_once.py','launch_registered.py','launch_metric_stage.py','metrics.py','control_once.py']]
rp=D/(role+'_registration.json');save(rp,dict(code=code,output=str(L/role),stages=stages,metric_gpu_uuid='GPU-f5047136-08e5-9416-3c86-87786d0b355e',seconds=7200,finite_queue=True,no_new_scientific_scope=True))
cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'metric_queue_once.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=7230,gpu=False))
with(L/(role+'_owner.log')).open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
