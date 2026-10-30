import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print([{k:v[k]for k in ['variant','component','mean_seconds','median_seconds','p95_seconds','mean_independent_terminations','extra_serialized_checkpoint_bytes']}for v in csv.DictReader((D/'timing_delivery/timing_and_compute.csv').open())if v['snr_db']=='pooled_core' and v['component']in ['TX_total','RX_total']])
for role in ['final_rx','final_successors']:
 p=L/(role+'_control/actual_wait.json');print(role,read(p)if p.exists()else'RUNNING')
 print((L/(role+'_control/worker.log')).read_text()[-250:])