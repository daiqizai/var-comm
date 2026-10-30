import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for role in ['final_phy_512','final_phy_1024','final_historical_missing_clip']:
 print(role,read(L/role/'completion.json'))
for role in ['benchmark_rx','final_rx']:
 p=L/(role+'_control/worker.log');print(role,p.read_text()[-180:])
 p=L/role/'failure.json'
 if p.exists():print(read(p))
