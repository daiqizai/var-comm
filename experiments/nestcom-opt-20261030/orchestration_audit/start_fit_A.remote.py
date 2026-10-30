import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
role='fit_A';assert not(D/(role+'_registration.json')).exists()
w=read(L/'variant_qualification_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
jobs=read(L/'fit_rx_plan/inputs.json');reuse=read(L/'fit_rx_plan/reuse.json');N=len(jobs)
gs=[x.split(', ')for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()];procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
g=next(g for g in gs if int(g[3])>=60000 and int(g[4])<=10 and g[1]not in procs)
p0=read(D/'p0_registration.json');a=read(D/'protocol_resolved_p0.json');code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/p)for p in ['opt_common.py','receiver_variants.py','receiver_worker.py','control_once.py']]+[pin(R/'experiments/progressive-rd-scaleup-20261022/decoder_model.py')]
r=dict(code=code,jobs=pin(L/'fit_rx_plan/inputs.json'),reuse=pin(L/'fit_rx_plan/reuse.json'),model_routes=p0['model_routes'],static=p0['static'],anchors=a['anchors'],output=str(L/role),cpu_affinity=[22,23],pci=g[2],route='A',seconds=6600,caps=dict(model_load=1,prior_scale=N*20,decoder_forward=N,source_decode=N-len(reuse),completion_forward=N,render=N,anchor_load=2,anchor_render=N*2),budget_GPU_seconds=7200,remaining_metrics_reserve_seconds=600)
rp=D/(role+'_registration.json');save(rp,r);cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'receiver_worker.py'),str(rp),sha(rp)],environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=g[0],PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),hard_seconds=6650,gpu=True));save(D/(role+'_admission.json'),dict(gpu=g,processes=procs,time=time.time()))
with(L/(role+'_owner.log')).open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,g,ident(p.pid),sha(rp))
