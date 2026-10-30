import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from development_summary import main
for p in ['fit_A_delivery','fit_B_delivery','fit_C_delivery','fit_D_delivery','fit_E_delivery']:
 print(p,sorted(x.name for x in (D/p).glob('*.csv')))
 assert (D/p/'gate.json').exists()
main()
