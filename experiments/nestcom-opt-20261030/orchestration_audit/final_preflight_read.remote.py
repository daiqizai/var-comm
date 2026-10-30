import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for p in [D/'timing_source_rule.json',R/'results/two_route_confirmation_20261028_v1/registration.json',R/'results/two_route_confirmation_20261028_v1/scope.json']:
 if p.exists():print('SMALL',p.name,str(read(p))[:3200])
for p in [R/'results/two_route_confirmation_20261028_v1',B/'two_route_confirmation_20261028_v1']:
 print(str(p),[x.name for x in p.iterdir()][:35]if p.exists()else'MISSING')
print('CONFIRM_CLIP',read(L/'confirm_D_clip_control/actual_wait.json')if(L/'confirm_D_clip_control/actual_wait.json').exists()else'RUNNING')
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader'],text=True))
