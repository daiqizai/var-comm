"""Exclusive new-task launches, live process identity, cumulative route budget."""
from opt_common import *
import subprocess
CAPS=dict(P0=3600,A=7200,B=21600,C=86400,D=14400,E=10800,combinations_and_confirm=28800,final=86400)
def spent(route):
 total=0.;receipts=[]
 for rp in D.glob('*_registration.json'):
  v=read(rp)
  if v.get('route')!=route or 'output'not in v:continue
  control=L/(Path(v['output']).name+'_control');p=control/'actual_wait.json'
  if p.exists():
   w=read(p);total+=w['device_seconds'];receipts.append(pin(p))
  elif(control/'child.json').exists():
   c=read(control/'child.json');live=ident(c['child']['pid']);assert live is not None and live['start_ticks']==c['child']['start_ticks'],('Unresolved vanished child',control)
   total+=time.time()-c['started_unix']
 return total,receipts

def launch(role,request,worker,python,seconds,route,required_gpu_uuid=None):
 assert route in CAPS and Path(worker).parent==E
 assert not(D/(role+'_registration.json')).exists()and not(L/role).exists()
 active=[]
 for p in L.glob('*_control/child.json'):
  if(p.parent/'actual_wait.json').exists():continue
  c=read(p);v=ident(c['child']['pid'])
  assert v is not None and v['start_ticks']==c['child']['start_ticks'],('Unresolved control',p)
  cp=read(p.parent/'owner.json')['registration'];reg=cr(cp)
  if reg['gpu']:active.append(c)
 assert len(active)<2,('Two owned GPU workers already active',active)
 used,receipts=spent(route);remaining=CAPS[route]-used
 assert remaining>60,('BUDGET_STOP',route,used)
 seconds=min(seconds,remaining-30)
 query=['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits']
 gs=[x.split(', ')for x in subprocess.check_output(query,text=True).splitlines()]
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 owned_pids={str(c['child']['pid'])for c in active}
 occupied_by_own={line.split(', ')[0]for line in processes.splitlines()if line.split(', ')[1]in owned_pids}
 eligible=[g for g in gs if int(g[3])>=60000 and int(g[4])<=50 and g[1]not in occupied_by_own and(required_gpu_uuid is None or g[1]==required_gpu_uuid)]
 assert eligible,'No eligible GPU at current resource admission; zero registration/launch'
 eligible.sort(key=lambda g:(g[1]in processes,int(g[4]),-int(g[3]),int(g[0])));g=eligible[0]
 r=dict(request,route=route,output=str(L/role),pci=g[2],seconds=seconds,budget_total_seconds=CAPS[route],prior_route_device_seconds=used,prior_route_wait_receipts=receipts)
 rp=D/(role+'_registration.json');save(rp,r);cp=D/(role+'_control_registration.json')
 save(cp,dict(code=r['code'],control=str(L/(role+'_control')),command=[str(python),'-B',str(worker),str(rp),sha(rp)],environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=g[0],PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),hard_seconds=min(seconds+20,remaining),gpu=True))
 save(D/(role+'_admission.json'),dict(gpu=g,processes=processes,time=time.time(),existing_owned_workers=active,co_resident_workloads_allowed_for_quality=True,benchmark=False))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  dispatch=dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True);save(L/(role+'_dispatch.json'),dispatch)
 print(json.dumps(dict(role=role,gpu=g,registration=pin(rp),dispatch=dispatch,route_seconds_used=used,stage_seconds_cap=seconds)))
