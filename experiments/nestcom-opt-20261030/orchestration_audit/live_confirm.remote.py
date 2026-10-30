import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from collect_progress import main
main()
for role in ['fit_metric_queue','C_guard_metric_queue','fit_C_guard_three','fit_C_guard_clip','confirm_D_source','confirm_base_phy_512','confirm_base_phy_1024','confirm_D_phy_512','confirm_D_phy_1024']:
 p=L/(role+'_control')/'actual_wait.json'
 print(role,read(p)if p.exists()else'NO_WAIT_YET')
 p=L/(role+'_control')/'worker.log'
 if p.exists():print(p.read_text()[-500:])
 p=L/role/'failure.json'
 if p.exists():print('FAILURE',read(p))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader'],text=True))
