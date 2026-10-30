import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
p=L/'fit_D_clip_control/actual_wait.json'
if not p.exists():print('D_CLIP_RUNNING');raise SystemExit(0)
w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
from route_statistics_v2 import main
main('fit_D')
from screen_decision import main
main('fit_D','D')
from launch_metric_stage import main
main('fit_E','three','E')
