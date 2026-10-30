import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from collect_progress import main
main()
for role in ['benchmark_rx','final_rx']:
 print(role,(L/(role+'_control/worker.log')).read_text()[-250:])
for name in ['fit_C_registration.json','p0_registration.json']:
 r=read(D/name)
 if name.startswith('p0'):print('models',cr(r['model_routes'])['files'])
