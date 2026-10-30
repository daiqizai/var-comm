import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from metric_plan_variants import main
os.sched_setaffinity(0,{24,25})
main('fit_C',L/'fit_C_rx_plan/logical.json')
from c_token_diagnostics import main as token
token('fit_C',L/'fit_C_rx_plan/logical.json')
print('C_POSTEXIT_TOKEN_DIAGNOSTICS_COMPLETE')
