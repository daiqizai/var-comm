import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_metric_stage import main
w=L/'fit_B_control/actual_wait.json'
if not w.exists():
 print('B_RUNNING', (L/'fit_B_control/worker.log').read_text()[-130:]);raise SystemExit(0)
a=read(w);assert a['returncode']==0 and ident(a['child']['pid'])is None
print('B_CLOSED',read(L/'fit_B/completion.json'))
from metric_plan import main as prepare
prepare('fit_B')
main('fit_D','three','D')
