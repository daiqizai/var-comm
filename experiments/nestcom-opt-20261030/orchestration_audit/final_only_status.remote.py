import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from collect_progress import main
main()
for role in ['final_rx','final_successors','final_new_three','final_new_clip']:
 p=L/(role+'_control/worker.log')
 if p.exists():print(role,p.read_text()[-400:])
 p=L/role/'failure.json'
 if p.exists():print(role,'FAILURE',p.read_text()[:1500])