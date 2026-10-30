import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for role in ['fit_phy_512','fit_phy_1024']:
 print('ROLE',role)
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json',L/role/'completion.json']:
  if p.exists():print(p.name,read(p))
 p=L/(role+'_control')/'worker.log'
 if p.exists():print('TAIL',p.read_text()[-1500:])
print('CACHE_TAIL',(L/'c_cache_control/worker.log').read_text()[-450:])
