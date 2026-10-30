import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print((R/'AGENTS.md').read_text())
print('GIT',subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True),subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True))
print('TRACKED',subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True))
for role in ['fit_A_clip','c_cache']:
 print('ROLE',role)
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json',L/role/'completion.json']:
  if p.exists():print(p.name,read(p))
 p=L/(role+'_control')/'child.json'
 if p.exists():
  c=read(p);print('CHILD',c,'LIVE',ident(c['child']['pid']))
 print((L/(role+'_control')/'worker.log').read_text()[-500:])
