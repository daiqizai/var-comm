import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for name in ['verify_repository','run_cpu_checks']:
 p=L/'repository_checks'/(name+'.stderr.txt');print(name,p.read_text()[:2400])
p=L/'benchmark_phy_control/actual_wait.json';print('PHY',read(p)if p.exists()else'RUNNING');p=L/'benchmark_phy/failure.json'
if p.exists():print(p.read_text())