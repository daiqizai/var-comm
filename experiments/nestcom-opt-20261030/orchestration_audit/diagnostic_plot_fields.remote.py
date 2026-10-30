import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for p in [L/'historical_regression_inputs/frames.json',L/'final_D_frame_plan/logical.json',L/'fit_E_score/selection.json']:
 print('PATH',p,'exists',p.exists())
 if p.exists():
  obj=read(p);row=obj[0]if isinstance(obj,list)else obj;print('KEYS',list(row)[:45]);print(str(row)[:1600])
for role in ['fit_E_score','benchmark_tx_identity_v2']:
 p=L/role/'completion.json';print(role,read(p)if p.exists()else'RUNNING')
print('Efiles',[x.name for x in (L/'fit_E_score').iterdir()])