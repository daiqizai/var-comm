import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from route_statistics_v2 import main as stats
from screen_decision import main as decide
from launch_metric_stage import main as metric
for role in ['fit_E_three','fit_E_clip','fit_B']:
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
stats('fit_E');decide('fit_E','E')
metric('fit_B','three','B')
