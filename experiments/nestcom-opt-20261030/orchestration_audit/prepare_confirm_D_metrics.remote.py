import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from evaluation_metric_plan import main
for role in ['confirm_base_rx','confirm_D_rx']:
 p=L/(role+'_control')/'actual_wait.json'
 if not p.exists():print('DEPENDENCY_RUNNING',role);raise SystemExit(0)
 w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
 print(role,read(L/role/'completion.json'))
rp=D/'confirm_D_metric_plan_registration.json'
save(rp,dict(role='confirm_D',sources=pin(L/'inputs/confirm100.json'),segments=[dict(receiver_role='confirm_base_rx',logical=pin(L/'confirm_base_rx_plan/logical.json')),dict(receiver_role='confirm_D_rx',logical=pin(L/'confirm_D_rx_plan/logical.json'))],frozen=pin(D/'D_confirm_frozen.json')))
main(rp)
