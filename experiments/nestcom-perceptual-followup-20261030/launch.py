"""Finite per-stage first launches; max two owned GPU workers, no retry."""
from followup_common import *
import subprocess
def launch(role,request,worker,python,seconds,gpu=False,required_uuid=None,timing=False):
 assert not(D/(role+'_registration.json')).exists()and not(L/role).exists()and not(L/(role+'_control')).exists()
 protocol=read(D/'protocol.json');active=[];used=0.
 for p in L.glob('*_control/owner.json'):
  reg=cr(read(p)['registration']);w=p.parent/'actual_wait.json'
  if w.exists():
   v=read(w);used+=v['device_seconds'] if gpu else v['CPU_process_wall_seconds'];continue
  if not(p.parent/'child.json').exists():raise RuntimeError('Unresolved owner initialization: '+str(p))
  c=read(p.parent/'child.json');live=ident(c['child']['pid']);assert live and live['start_ticks']==c['child']['start_ticks'],('Unresolved child',p)
  if reg['gpu']:active.append(c)
  if bool(reg['gpu'])==gpu:used+=time.time()-c['started_unix']
 assert len(active)<2 and(not timing or not active)
 limit=protocol['execution_limits']['GPU_device_seconds'if gpu else'CPU_process_wall_seconds'];remaining=limit-used;assert remaining>60
 seconds=min(seconds,43200,remaining-30);env=dict(PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 admission={};r=dict(request,output=str(L/role),seconds=seconds,prior_device_or_CPU_process_seconds=used,protocol=pin(D/'protocol.json'))
 if gpu:
  gs=[v.split(', ')for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
  processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  own={str(c['child']['pid'])for c in active};occupied={v.split(', ')[0]for v in processes.splitlines()if v.split(', ')[1]in own}
  eligible=[g for g in gs if int(g[3])>=60000 and int(g[4])<=50 and g[1]not in occupied and(required_uuid is None or g[1]==required_uuid)]
  assert eligible,'No eligible GPU; zero registration or scientific launch'
  eligible.sort(key=lambda g:(g[1]in processes,int(g[4]),-int(g[3])));g=eligible[0]
  r.update(pci=g[2],gpu_uuid=g[1]);env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=g[0]);admission=dict(gpu=g,processes=processes,timing=timing)
 rp=D/(role+'_registration.json');save(rp,r);cp=D/(role+'_control_registration.json')
 save(cp,dict(code=r['code'],control=str(L/(role+'_control')),command=[str(python),'-B',str(worker),str(rp),sha(rp)],environment=env,hard_seconds=seconds+20,gpu=gpu))
 save(D/(role+'_admission.json'),dict(time=time.time(),resource=admission,already_owned_GPU_workers=len(active)))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True))
 print(json.dumps(dict(role=role,registration_sha256=sha(rp),owner=ident(p.pid),admission=admission,seconds=seconds)))
