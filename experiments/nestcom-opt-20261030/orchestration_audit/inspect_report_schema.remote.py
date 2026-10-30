import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
o=cr(read(D/'protocol_resolved_p0.json')['metrics']);receipt=cr(o['config']['config']['metric_config']['weight_receipt']);print('weight_receipt',receipt)
for role in ['benchmark_tx_identity_v2','final_rx']:
 p=L/(role+'_control/actual_wait.json');print(role,read(p)if p.exists()else'RUNNING')
 p=L/(role+'_control/worker.log');print(p.read_text()[-250:]if p.exists()else'NOLOG')
print('root_result_files',[x.name for x in D.iterdir()if x.is_file()])
print('checkpoint1',read(D/'model_storage_and_recovery.json')['C_checkpoints'][0])