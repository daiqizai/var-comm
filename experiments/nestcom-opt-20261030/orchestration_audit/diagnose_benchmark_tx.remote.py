import sys,subprocess,json
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print((R/'AGENTS.md').read_text())
print(subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=R,text=True))
print('HEAD',subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True))
for role in ['benchmark_tx','final_successors']:
 for name in ['actual_wait.json','owner_identity.json']:
  p=L/(role+'_control')/name
  if p.exists(): print(role,name,p.read_text())
 for name in ['failure.json','completion.json']:
  p=L/role/name
  if p.exists(): print(role,name,p.read_text())
 p=L/(role+'_control/worker.log');print(role,'LOG',p.read_text()[-8500:]if p.exists()else 'NONE')