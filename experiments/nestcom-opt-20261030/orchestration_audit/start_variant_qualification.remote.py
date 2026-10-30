import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
role='variant_qualification';assert not(D/(role+'_registration.json')).exists()
gs=[x.split(', ')for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()];procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
g=next(g for g in gs if int(g[3])>=60000 and int(g[4])<=10 and g[1]not in procs)
p0=read(D/'p0_registration.json');code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/p)for p in ['opt_common.py','receiver_variants.py','c_model.py','qualify_variants.py','control_once.py']]
r=dict(code=code,donors=p0['donors'],model_routes=p0['model_routes'],output=str(L/role),cpu_affinity=[22,23],pci=g[2],seconds=600,route='P0',previous_device_seconds=read(L/'p0_control/actual_wait.json')['device_seconds'])
rp=D/(role+'_registration.json');save(rp,r);cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'qualify_variants.py'),str(rp),sha(rp)],environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=g[0],PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),hard_seconds=650,gpu=True));save(D/(role+'_admission.json'),dict(gpu=g,processes=procs,time=time.time()))
with(L/(role+'_owner.log')).open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,g,ident(p.pid))
