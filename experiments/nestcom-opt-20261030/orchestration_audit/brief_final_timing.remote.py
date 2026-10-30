import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print('ROUTES',[(v['candidate'],v['core_J'])for v in csv.DictReader((D/'development_summary/route_summary.csv').open())])
print('C_TOKEN_DIRS',[str(p.relative_to(D))for p in D.glob('*token_diagnostics/*')])
for role in ['benchmark_rx','benchmark_tx_identity_v2','final_rx','final_successors']:
 p=L/(role+'_control/actual_wait.json');print(role,read(p)if p.exists()else'RUNNING')
 p=L/(role+'_control/worker.log');print(p.read_text()[-300:]if p.exists()else'NOLOG')
 p=L/role/'actual_native_identity.json'
 if p.exists():print('NATIVE',role,read(p))