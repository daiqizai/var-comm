import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
p=L/'fit_D_guard_rx_control/actual_wait.json'
if not p.exists():print('D_GUARD_RX_RUNNING');raise SystemExit(0)
w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
from metric_plan_projected import main
main('fit_D_guard','fit_D_guard_rx_plan','fit_D_guard_rx')
