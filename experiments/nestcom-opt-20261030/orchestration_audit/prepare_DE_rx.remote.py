import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from prepare_route_rx import from_frames,tx_simulation
from_frames('fit_D_frame_plan',['fit_D_phy_1024'],'fit_D_rx_plan');tx_simulation()
for role in ['c_cache','fit_B']:
 print(role)
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json',L/role/'completion.json']:
  if p.exists():print(p.name,read(p))
 print((L/(role+'_control')/'worker.log').read_text()[-550:])
