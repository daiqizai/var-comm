"""Explicit stage launches after actual prior closure; never retry existing roles."""
from n512_common import *
from dispatch import launch
from pipeline_lineage_v3 import closed,phy_reuse,prepare_rx,pair_plan
import subprocess

def budget(gpu,workers):
 spent=0
 for p in (L/'control_requests').glob('*.json'):
  r=read(p);is_gpu=bool(r.get('environment',{}).get('CUDA_VISIBLE_DEVICES',''))
  if is_gpu!=gpu:continue
  c=Path(r['control']);wp=c/'actual_wait.json'
  if wp.exists():spent+=read(wp)['seconds']
  elif (c/'launch.json').exists():raise RuntimeError('Another active stage owns resource budget: '+str(c))
 limit=96*3600-spent;assert limit>60*workers
 return min(43200,int(limit/workers)-30),spent

def start_phy(stage):
 plan=read(L/(stage+'_plan')/'completion.json');assert plan['status']=='PLANNED';rbase=read(O/'protocol.json');reuse=phy_reuse(stage)
 reusepath=L/(stage+'_plan')/'reuse.json';save(reusepath,reuse);seconds,spent=budget(False,8)
 deps=read(O/'phy_qualification_registration.json')['code']+[pin(E/p)for p in ['link_worker.py','dispatch.py','stage_operations_v4.py','pipeline_lineage_v3.py','rx_lineage.py']]
 regs=[]
 for i,desc in enumerate(plan['shards']):
  jobs=cr(desc);role=f'{stage}_phy_{i:02d}';r=dict(stage=stage,protocol=pin(O/'protocol.json'),jobs=desc,choices=plan['choices'],reuse=pin(reusepath),output=str(L/role),cpu_affinity=[2*i,2*i+1],code=deps,max_seconds=seconds,CPU_process_seconds_already_spent=spent)
  rp=O/(role+'_registration.json');save(rp,r);regs.append(pin(rp));print(role,len(jobs),launch(role,rp,E/'link_worker.py'),flush=True)
 save(L/(stage+'_phy_dispatch.json'),dict(registrations=regs,logical=plan['logical_count'],physical=plan['physical_count'],prior_physical=len(reuse)))

def start_rx(stage):
 plan=prepare_rx(stage);seconds,spent=budget(True,8);regs=[];bus=['61','62','63','64','a1','a2','a3','a4']
 for codec,offset,template in [('NEST',0,'route_first25_rx'),('ADA',4,'qualification_rx_ada')]:
  old=read(O/(template+'_registration.json'))
  deps=[d for d in old['code']if Path(d['path']).name!='rx_worker.py']
  for p in [E/'rx_worker_v2.py',E/'dispatch.py',E/'stage_operations_v4.py',E/'pipeline_lineage_v3.py','rx_lineage.py']:
   if str(p)not in {x['path']for x in deps}:deps.append(pin(p))
  for i,desc in enumerate(plan['shards'][codec]):
   n=len(cr(desc))
   if not n:continue
   gpu=i+offset;role=f'{stage}_rx_{codec.lower()}_{i:02d}';r=dict(old);r.update(inputs=desc,output=str(L/role),code=deps,cpu_affinity=[2*gpu,2*gpu+1]if codec=='NEST'else list(range(16+6*i,22+6*i)),pci=f'0000:{bus[gpu]}:00.0',max_seconds=seconds,GPU_device_seconds_already_spent=spent,caps=dict(model_load=1,receive_render=n))
   if codec=='NEST':r['caps'].update(prior_scale=n*20,decoder_forward=n)
   rp=O/(role+'_registration.json');save(rp,r);regs.append(pin(rp));print(role,n,launch(role,rp,E/'rx_worker_v2.py',gpu=gpu),flush=True)
 save(L/(stage+'_rx_plan')/'completion.json',dict(plan,worker_registrations=regs))

def start_dino(stage):
 assert stage in ('pilot','full');assert read(O/(stage+'_post_output_audit.json'))['status']=='PASS';plan=pair_plan(stage);seconds,spent=budget(True,1)
 old=read(R/'results/adatoken_response_20261026_v1/c_predclass_cal1000_dino_pro_v1_registration.json')
 deps=[pin(E/p)for p in ['n512_common.py','dino_worker.py','stage_operations_v4.py','pipeline_lineage_v3.py','rx_lineage.py']]+old['code']
 role=stage+'_dino';r=dict(plan=plan['pairs'],output=str(L/role),metric_config=old['metric_config'],pci='0000:A3:00.0',cpu_affinity=list(range(6)),max_seconds=seconds,code=deps,prior_score_dirs=[]if stage=='pilot'else[str(L/'pilot_dino')],GPU_device_seconds_already_spent=spent)
 rp=O/(role+'_registration.json');save(rp,r);print(launch(role,rp,E/'dino_worker.py',gpu=6),flush=True)

if __name__=='__main__':
 action,stage=sys.argv[1:];assert stage in ('pilot','full','formal')
 {'start_phy':start_phy,'start_rx':start_rx,'start_dino':start_dino}[action](stage)
