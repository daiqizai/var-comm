import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from collect_progress import main
main()
for role in ['fit_metric_queue','fit_B_clip','fit_C_three','fit_D_guard_rx']:
 for name in ['completion.json','failure.json']:
  p=L/role/name
  if p.exists():print(role,name,read(p))
 p=L/(role+'_control')/'worker.log'
 if p.exists():print(role,'tail',p.read_text()[-250:])
