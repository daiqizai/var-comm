import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from evaluation_metric_plan import main
from launch_metric_stage_v2 import main as launch
role='final_rx';p=L/(role+'_control')/'actual_wait.json'
if not p.exists():print('DEPENDENCY_RUNNING',role);raise SystemExit(0)
w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
rp=D/'final_new_metric_plan_registration.json';save(rp,dict(role='final_new',sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),segments=[dict(receiver_role='final_rx',logical=pin(L/'final_rx_plan/logical.json'))],frozen=pin(D/'frozen_final.json')));main(rp)
launch('final_new','three','final',('fit_A','confirm_D'))
