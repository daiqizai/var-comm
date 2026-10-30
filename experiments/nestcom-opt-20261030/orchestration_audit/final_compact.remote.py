import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from collect_progress import main
r=main()
for role in ['final_phy_512','final_phy_1024','final_historical_missing_clip','benchmark_rx']:
 p=L/(role+'_control')/'worker.log'
 if p.exists():print(role,p.read_text()[-450:])
 p=L/role/'failure.json'
 if p.exists():print('FAILURE',role,read(p))
