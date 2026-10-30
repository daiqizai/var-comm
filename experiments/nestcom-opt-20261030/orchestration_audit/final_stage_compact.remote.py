import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print('CURRENT_LARGEST',sorted([(p.stat().st_size,str(p.relative_to(D)))for p in D.rglob('*')if p.is_file()],reverse=True)[:8])
for role in ['final_rx','final_successors','final_new_three','final_new_clip']:
 p=L/(role+'_control/actual_wait.json');print(role,read(p)if p.exists()else'NO_EXIT')
 p=L/(role+'_control/worker.log')
 if p.exists():print(p.read_text()[-200:])