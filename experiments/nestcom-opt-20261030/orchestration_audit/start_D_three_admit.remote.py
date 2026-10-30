import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True));print('PROCESSES',subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True))
print('B_CLOSED',read(L/'fit_B/completion.json'));print('B_METRIC_PLAN',read(L/'fit_B_metric_plan/completion.json'));print('D_REG_EXISTS',(D/'fit_D_three_registration.json').exists())
from launch_metric_stage import main
main('fit_D','three','D')
