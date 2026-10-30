import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for path in [B/'nestcom_7db_common500_20261030_v1/rx_nest/outputs.json',L/'fit_E_tx_plan/choices.json',L/'fit_E_txscore/selection.json',D/'fit_E_tx_score_registration.json',D/'fit_C_registration.json']:
 if path.exists():
  v=read(path);print(path.name, str(v[0] if isinstance(v,list) else v)[:2200])
p=B/'nestcom_7db_common500_20261030_v1/rx_nest';print('rx_files',[x.name for x in p.glob('*.json')][:20]);print('timing16',[(x.name,x.stat().st_size)for x in (L/'inputs').glob('*')])
print('Edirs',[x.name for x in L.glob('fit_E*')]);print('Cregs',[x.name for x in D.glob('*C*registration.json')])
