"""Explicit first-only transitions for the predeclared calibration stages."""
from study_common import *
from launch_extra import launch_extra
import calibration_plan_v2 as planning

def start_phy(stage):
 closed('directory_qualification');planning.plan(stage);root=L/('cal_'+stage+'_plan');p=read(root/'completion.json');reuse={}
 for prior in {'pilot':[],'full':['pilot'],'formal':['pilot','full']}[stage]:
  dispatch=read(L/('cal_'+prior+'_phy_dispatch.json'))
  for role in dispatch['roles']:
   closed(role)
   for row in read(L/role/'frames.json'):reuse[row['physical_key']]=row
 save(root/'reuse.json',reuse);roles=[];base=read(OLD/'final_phy_1024_registration.json')
 for i,jobs in enumerate(p['shards']):
  if not cr(jobs):continue
  role=f'cal_{stage}_phy_{i:02d}';r=dict(jobs=jobs,choices=p['choices'],directories=pin(D/'public_directories.json'),reuse=pin(root/'reuse.json'),cpu_affinity=[i*2,i*2+1],code=base['code']+[pin(E/'calibration_plan_v2.py'),pin(E/'calibration_stages.py')])
  launch_extra(role,r,'cal_phy_worker_v2.py',CPY,7200);roles.append(role)
 save(L/('cal_'+stage+'_phy_dispatch.json'),dict(roles=roles,logical=p['logical'],physical=p['physical'],sources=p['sources']))
if __name__=='__main__':start_phy(sys.argv[1])
