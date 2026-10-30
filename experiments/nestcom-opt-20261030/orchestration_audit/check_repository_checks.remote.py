import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for p in (L/'repository_checks').glob('*'):print(p.name,p.stat().st_size,p.read_text()[-500:])
print([x for x in subprocess.check_output(['ps','-eo','pid,ppid,etimes,args'],text=True).splitlines()if any(q in x for q in ['verify_repository.py','run_cpu_checks.py','test_c_contracts.py'])])
p=L/'benchmark_phy_control/actual_wait.json';print('PHY',read(p)if p.exists()else'RUNNING')
p=L/'benchmark_phy/failure.json'
if p.exists():print(p.read_text())