import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from collect_progress import main
main()
for role in ['fit_C','fit_B_three']:
 for name in ['completion.json','failure.json']:
  p=L/role/name
  if p.exists():print(role,name,read(p))
 p=L/(role+'_control')/'worker.log';print(role,'tail',p.read_text()[-400:])
