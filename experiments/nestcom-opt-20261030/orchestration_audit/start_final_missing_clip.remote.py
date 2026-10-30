import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_metric_stage_v2 import main
assert read(D/'confirm_D_delivery/decision.json')['confirm_pass'];p=L/'historical_regression_inputs';c=read(p/'completion.json');assert c['frames']==30000
out=L/'final_historical_missing_metric_plan';out.mkdir(exist_ok=False);save(out/'plan.json',read(p/'missing_clip_plan.json'));save(out/'completion.json',dict(status='PREPARED',historical_audit=pin(p/'completion.json'),scope='Only missing CLIP on exact old native arrays; all existing metric values reused',pairs=c['missing_clip_pairs'],no_rescoring_existing=True))
main('final_historical_missing','clip','final',('fit_A','confirm_D'))
