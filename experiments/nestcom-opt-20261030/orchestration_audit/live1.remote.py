import sys,subprocess,sqlite3
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for role in ['p0','c_cache']:
 c=L/(role+'_control')
 print(role)
 for n in ['owner','child','actual_wait']:
  p=c/(n+'.json')
  if p.exists():
   v=read(p);print(n,v)
   for f in ['owner','child']:
    if isinstance(v.get(f),dict):print('LIVE',ident(v[f]['pid']))
 p=c/'worker.log'
 if p.exists():print('TAIL',p.read_text()[-1400:])
print('HEAD',subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip())
print('STATUS',subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
for root in [R/'results',RT/'assets',Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-data-index')]:
 print('CANDIDATES',root,[p.name for p in root.iterdir() if any(t in p.name.lower() for t in ['source','imagenet','holdout','population','ep_new'])][:50]if root.exists()else'MISSING')
