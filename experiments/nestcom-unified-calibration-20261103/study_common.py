"""Bounded new study; earlier scientific outputs remain immutable inputs."""
from pathlib import Path
import sys,os,time,json,hashlib,subprocess
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,sha,pin,checked,cr,asha,digest,ident,Ledger,BudgetStop,OFFSETS,SIZES
import numpy as np
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/nestcom-unified-calibration-20261103'
D=R/'results/nestcom_unified_calibration_20261103_v1'
L=B/'nestcom_unified_calibration_20261103_v1'
OLD=R/'results/nestcom_opt_20261030_v1';OL=B/'nestcom_opt_20261030_v1'
F=R/'results/nestcom_perceptual_followup_20261030_v2';FL=B/'nestcom_perceptual_followup_20261030_v2'
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine','dists')
CPY=RT/'envs/frozen_ldpc_py311/bin/python';VPY=RT/'envs/frozen_visual_py310/bin/python';APY=Path('/usr/bin/python')

def closed(role):
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and w['final_identity'] is None,role
 for k in ['child','owner']:
  v=w[k];n=ident(v['pid']);assert n is None or n['start_ticks']!=v['start_ticks'] or n['state']=='Z',(role,k,n)
 c=read(L/role/'completion.json');assert c['status'].startswith('COMPLETE'),(role,c)
 if 'ledger' in c:assert c['ledger']['unresolved']==0
 assert not(L/role/'failure.json').exists(),role
 return c

def source_only(s):return {k:s[k] for k in ['source_id','source_index','archive','preprocessing_id','reference_sha256']}

def launch(role,r,kind,python,seconds=3600,gpu=None):
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
 admission={}
 if gpu is not None:
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
  g=next(v.split(', ') for v in lines if int(v.split(', ')[0])==gpu)
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  assert int(g[3])>=60000 and int(g[4])<=50 and g[1]not in apps,('GPU not free',g,apps)
  env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID');r=dict(r,pci=g[2],gpu_uuid=g[1]);admission=dict(gpu=g,apps=apps)
 own=[pin(E/n)for n in ['study_common.py','control.py','worker.py']]
 r=dict(r,code=r['code']+own,output=str(L/role),seconds=seconds,max_seconds=seconds,protocol=pin(D/'protocol.json'),deadline_unix=protocol['deadline_unix'])
 rp=D/(role+'_registration.json');save(rp,r);cp=D/(role+'_control_registration.json')
 save(cp,dict(code=r['code'],control=str(L/(role+'_control')),command=[str(python),'-B',str(E/'worker.py'),kind,str(rp),sha(rp)],environment=env,hard_seconds=seconds+20,gpu=gpu is not None,admission=admission))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(CPY),'-B',str(E/'control.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),time=time.time(),unique_first_launch=True))
 print(json.dumps(dict(role=role,owner=ident(p.pid),seconds=seconds,gpu=gpu,registration_sha256=sha(rp))),flush=True)

def csvout(p,rows):
 import csv
 with Path(p).open('x',newline='',encoding='utf8')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
