import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
assert not(D/'c_cache_registration.json').exists()
w=read(L/'p0_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
assert read(L/'p0_replay/completion.json')['ledger']['unresolved']==0
gs=[x.split(', ')for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
eligible=[g for g in gs if int(g[3])>=60000 and int(g[4])<=10 and g[1]not in procs];assert eligible,'No empty PRO6000 currently admitted'
g=eligible[0]
code=[pin(E/p)for p in ['opt_common.py','c_cache.py','control_once.py']]+[v for v in read(D/'p0_registration.json')['code']if not v['path'].startswith(str(E))]
r=dict(code=code,sources=pin(L/'inputs/train10000.json'),model_routes=read(D/'p0_registration.json')['model_routes'],output=str(L/'c_cache'),cpu_affinity=[20,21],pci=g[2],seconds=86300,route='C',route_total_GPU_seconds_cap=86400,cumulative_previous_C_device_seconds=0,cache_limit_bytes=80*(1<<30),reuse_audit='Existing train shards supply exact frozen token labels. Existing eight-endpoint latent caches contain latents/wire/received_tokens, not eligible native FP32 pre-boundary base logits; zero such logits claimed reused.',caps=dict(model_load=1,prior_scale=90000,decoder_forward=0,source_teacher_cache=10000))
save(D/'c_cache_registration.json',r);reg=D/'c_cache_control_registration.json'
c=dict(code=code,control=str(L/'c_cache_control'),command=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'c_cache.py'),str(D/'c_cache_registration.json'),sha(D/'c_cache_registration.json')],environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=g[0],PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),hard_seconds=86400,gpu=True)
save(reg,c);save(D/'c_cache_resource_admission.json',dict(gpu=g,processes=procs,time=time.time()))
with(L/'c_cache_owner.log').open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(reg),sha(reg)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/'c_cache_dispatch.json',dict(owner=ident(p.pid),registration=pin(reg),unique_first_launch=True));print(json.dumps(read(L/'c_cache_dispatch.json')))
