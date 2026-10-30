import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
N=B/'nestcom_n512_20261028_v1'
for name in ['route_existing25_rx','full_rx_nest_00','formal_rx_nest_00','pilot_rx_nest_00']:
 p=N/name;print(name,[x.name for x in p.glob('*.json')]);q=p/'actual_native_identity.json'
 if q.exists():print(read(q))
print('QUAL',(L/'variant_qualification_control/worker.log').read_text()[-800:])
for x in [L/'variant_qualification/failure.json',L/'variant_qualification/completion.json',L/'variant_qualification_control/actual_wait.json']:
 if x.exists():print(x.name,read(x))
