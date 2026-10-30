import sys,subprocess,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
assert not(L/'p0_dispatch.json').exists()and not(L/'p0_control').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
rows=[x.strip().split(', ')for x in gpu.splitlines()];g=rows[3]
assert g[1]=='GPU-ed8012eb-9835-8c43-e5cf-43765b67f343'and int(g[3])>=60000 and int(g[4])<=10,g
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
assert g[1]not in procs,('GPU occupied',procs)
save(D/'p0_resource_admission.json',dict(gpu=gpu,processes=procs,time=time.time(),no_other_user_processes_on_selected_gpu=True))
reg=D/'p0_control_registration.json';assert sha(reg)=='1403604732ba34b17f1929e071a85ff08f6a0a2164c612c62623ea780db71206'
with(L/'p0_owner.log').open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(reg),sha(reg)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/'p0_dispatch.json',dict(owner=ident(p.pid),registration=pin(reg),unique_first_launch=True));print(json.dumps(read(L/'p0_dispatch.json')))
