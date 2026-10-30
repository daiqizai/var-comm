import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for dirname in ['fit_A_delivery','fit_B_delivery','fit_C_complete_delivery','fit_D_complete_delivery','fit_E_delivery']:
 print(dirname,[p.name for p in(D/dirname).iterdir()])
 for name in ['frame_metrics.csv','frame_scores.csv','per_source_metrics.csv','summary.csv']:
  p=D/dirname/name
  if p.exists():
   with p.open()as f:
    reader=csv.DictReader(f);v=next(reader);print(name,reader.fieldnames,'sample',str(v)[:1800])
for role in ['fit_A','fit_B','c_train_primary']:
 p=L/role/'completion.json';v=read(p);print(role,'keys',list(v),'ledger',v.get('ledger'))
print('display',read(D/'display_source_rule.json'))
