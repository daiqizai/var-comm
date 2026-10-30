import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
n=B/'nestcom_n512_20261028_v1'
print('RXDIRS',[p.name for p in n.glob('formal_rx_*')if p.is_dir()])
with(R/'results/nestcom_n512_20261028_v1/delivery/frame_scores.csv').open()as f:rows=[v for v in csv.DictReader(f)if v['received_key']=='6c56f830f1ab5a77300e30bf8529ab87fcf6b33ee7fac7383369e3f891476cc0']
print('MISSING_ROW',rows[:1])
for p in [n/'formal_metric_plan/plan.json',n/'formal_metric_plan/three_plan.json',n/'formal_rx_plan/completion.json']:
 if p.exists():
  v=read(p);print(p.name,list(v)if isinstance(v,dict)else len(v));
  if p.name=='completion.json':print(str(v)[:2500])
print('D_WAIT',read(L/'confirm_D_rx_control/actual_wait.json')if(L/'confirm_D_rx_control/actual_wait.json').exists()else'RUNNING')
