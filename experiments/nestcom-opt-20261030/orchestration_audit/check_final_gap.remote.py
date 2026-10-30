import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print('AGENTS', (R/'AGENTS.md').read_text())
print('HEAD',subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip())
print('STATUS',subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=R,text=True)[-1500:])
for role in ['final_rx','final_successors']:
 w=read(L/(role+'_control/actual_wait.json'));o=read(L/(role+'_control/owner.json'))
 print(role,w,'owner_now',ident(o['owner']['pid']),'child_now',ident(w['child']['pid']))
print('failure',read(L/'final_successors/failure.json'))
for role in ['final_new_three','final_new_clip','final_unlaunched_metrics_v2']:
 print(role,'registration', (D/(role+'_registration.json')).exists(),'output',(L/role).exists())
print('plan',read(L/'final_new_metric_plan/completion.json'))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
