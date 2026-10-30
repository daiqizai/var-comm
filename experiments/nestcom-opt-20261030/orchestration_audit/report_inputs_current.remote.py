import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader'],text=True))
print('correction',pin(D/'benchmark_tx_identity_correction.json'))
for x in ['benchmark_tx_identity_v2_registration.json','benchmark_tx_identity_v2_admission.json']:print(x,(D/x).exists())
print('GUARD',read(D/'fit_C_complete_delivery/gate.json'))
print('confirm',read(D/'confirm_D_delivery/gate.json'))
print('storage_keys',read(D/'model_storage_and_recovery.json').keys())
print('storage_models',read(D/'model_storage_and_recovery.json').get('models'))
for p in ['fit_A_delivery','fit_B_delivery','fit_C_delivery','fit_C_complete_delivery','fit_D_complete_delivery','confirm_D_delivery','fit_E_delivery']:
 print('CSV',p,[(q.name,q.stat().st_size)for q in (D/p).glob('*')])