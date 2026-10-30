import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_metric_stage import main
p=L/'fit_D_three_control/actual_wait.json'
if not p.exists():print('D_THREE_RUNNING');raise SystemExit(0)
w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
print(read(L/'fit_D_three/completion.json'))
main('fit_D','clip','D')
