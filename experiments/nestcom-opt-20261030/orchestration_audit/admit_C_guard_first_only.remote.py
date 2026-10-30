import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_metric_stage import main
assert read(D/'C_guard_unlaunched_metric_admission.json')['model_calls_in_failed_queue']==0
role='fit_C_guard_three'
assert not(D/(role+'_registration.json')).exists()and not(L/role).exists()and not(L/(role+'_control')).exists()
gs=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
g=next(x.split(', ')for x in gs.splitlines()if x.startswith('GPU-f5047136'))
if int(g[1])<60000 or int(g[2])>50:print('ADMISSION_NOT_YET',g);raise SystemExit(0)
main('fit_C_guard','three','C',('fit_A','fit_C'))
