import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
p=L/'fit_C_guard_control/actual_wait.json'
if not p.exists():print('C_GUARD_RUNNING');raise SystemExit(0)
w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
os.sched_setaffinity(0,{24,25})
from metric_plan_variants import main
main('fit_C_guard',L/'fit_C_guard_rx_plan/logical.json')
from c_token_diagnostics import main as token
token('fit_C_guard',L/'fit_C_guard_rx_plan/logical.json')
