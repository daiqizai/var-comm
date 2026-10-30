import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from route_statistics import bank
_,_,p=bank('confirm100');print('CONFIRM_BANK',p)
from metric_plan_projected import main
w=read(L/'fit_E_rx_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
main('fit_E','fit_E_rx_plan','fit_E_rx')
print('E_RX_CLOSED',read(L/'fit_E_rx/completion.json'))
for role in ['c_train_primary','fit_B']:
 print(role,(L/(role+'_control')/'worker.log').read_text()[-300:])
 p=L/role/'failure.json'
 if p.exists():print(read(p))
a=read(D/'source_manifest.json');print('SOURCE_MANIFEST_KEYS',list(a));print('FILES',[p.name for p in(L/'inputs').iterdir()])
for path in [R/'results/nestcom_n512_20261028_v1/protocol.json',R/'results/nestcom_n512_20261028_v1/delivery/statistics_identity.json']:
 if path.exists():
  v=read(path);print(str(path), {k:x for k,x in v.items()if 'bank'in k or'bootstrap'in k or'stat'in k})
