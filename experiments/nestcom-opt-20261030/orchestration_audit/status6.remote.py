import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from collect_progress import main
print((R/'AGENTS.md').read_text())
print('HEAD',subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True))
print('TRACKED',subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True))
r=main()
for role in ['fit_E_three','fit_E_clip','c_train_primary','fit_D_guard_phy_512','fit_D_guard_phy_1024','fit_B']:
 print('ROLE',role)
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json',L/role/'completion.json']:
  if p.exists():print(p.name,read(p))
 p=L/(role+'_control')/'worker.log'
 if p.exists():print('LOG_TAIL',p.read_text()[-1000:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader'],text=True))
