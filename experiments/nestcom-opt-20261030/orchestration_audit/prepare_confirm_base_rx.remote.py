import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from evaluation_receiver_plan import main
for role in ['confirm_base_phy_512','confirm_base_phy_1024']:
 p=L/(role+'_control')/'actual_wait.json'
 if not p.exists():print('DEPENDENCY_RUNNING',role);raise SystemExit(0)
 w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
rp=D/'confirm_base_rx_plan_registration.json';save(rp,dict(output='confirm_base_rx_plan',frame_roles=['confirm_base_phy_512','confirm_base_phy_1024'],logical=[pin(L/'confirm_base_frame_plan/logical.json')],current_original_roles=['fit_A','fit_D_rx','fit_D_guard_rx'],historical_n512=True));main(rp)
