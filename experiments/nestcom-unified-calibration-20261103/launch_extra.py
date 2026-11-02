"""One first-only extra worker; same aggregate study budget and admission."""
from study_common import *

def launch_extra(role,r,script,python,seconds=3600,gpu=None,extra_environment=None):
 """Reserve entire stage wall cap before one and only one Popen."""
 protocol=read(D/'protocol.json');assert not(D/(role+'_registration.json')).exists()
 assert not(L/role).exists() and not(L/(role+'_control')).exists()
 spent=0.;reserved=0.
 for p in L.glob('*_control/owner.json'):
  prior=cr(read(p)['registration']);w=p.parent/'actual_wait.json'
  if bool(prior['gpu'])!=(gpu is not None):continue
  if w.exists():spent+=read(w)['elapsed_seconds']
  else:reserved+=prior['hard_seconds']
 limit=protocol['budget']['GPU_device_seconds' if gpu is not None else 'CPU_process_seconds']
 available=min(limit-spent-reserved,protocol['deadline_unix']-time.time())
 assert available>seconds+30,('budget',available,seconds)
 env=dict(PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 env.update(extra_environment or {})
 admission={}
 if gpu is not None:
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
  g=next(v.split(', ') for v in lines if int(v.split(', ')[0])==gpu)
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  assert int(g[3])>=60000 and int(g[4])<=50 and g[1]not in apps,('GPU not free',g,apps)
  env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID');r=dict(r,pci=g[2],gpu_uuid=g[1]);admission=dict(gpu=g,apps=apps)
 own=[pin(E/n)for n in ['study_common.py','control.py','launch_extra.py',script]]
 r=dict(r,code=r['code']+own,output=str(L/role),seconds=seconds,max_seconds=seconds,protocol=pin(D/'protocol.json'),deadline_unix=protocol['deadline_unix'])
 rp=D/(role+'_registration.json');save(rp,r);cp=D/(role+'_control_registration.json')
 save(cp,dict(code=r['code'],control=str(L/(role+'_control')),command=[str(python),'-B',str(E/script),str(rp),sha(rp)],environment=env,hard_seconds=seconds+20,gpu=gpu is not None,admission=admission))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(CPY),'-B',str(E/'control.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),time=time.time(),unique_first_launch=True))
 print(json.dumps(dict(role=role,owner=ident(p.pid),seconds=seconds,gpu=gpu,registration_sha256=sha(rp))),flush=True)
