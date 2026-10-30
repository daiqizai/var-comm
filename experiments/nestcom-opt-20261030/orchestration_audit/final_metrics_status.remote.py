import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for role in ['final_unlaunched_metrics_v2','final_new_three','final_new_clip']:
 p=L/(role+'_control/actual_wait.json');o=L/(role+'_control/owner.json')
 print(role,'actualwait',read(p)if p.exists()else None,'owner',ident(read(o)['owner']['pid'])if o.exists()else None)
 for name in ['worker.log']:
  p=L/(role+'_control')/name
  if p.exists():print(p.name,p.read_text()[-600:])
 p=L/role/'failure.json'
 if p.exists():print('FAILURE',p.read_text())
print('final_stats',(D/'final_common500_delivery/statistics_identity.json').exists())
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
