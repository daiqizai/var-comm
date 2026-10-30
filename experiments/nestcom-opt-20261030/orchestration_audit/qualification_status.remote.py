import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
N=B/'nestcom_n512_20261028_v1'
p=N/'controls/full_rx_nest_00/actual_wait.json';print(read(p)if p.exists()else'MISSING')
for role in ['variant_qualification']:
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json',L/role/'completion.json']:
  if p.exists():print(str(p),read(p))
 print((L/(role+'_control')/'worker.log').read_text()[-1000:])
