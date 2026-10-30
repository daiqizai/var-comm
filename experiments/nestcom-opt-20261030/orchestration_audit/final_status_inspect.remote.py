import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from collect_progress import main
p=main()
for r in p['controls']:
 if r['role']=='final_D_source':print(r)
print(subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip())
print(subprocess.check_output(['git','-C',str(R),'diff','--stat'],text=True))
for path in [R/'AGENTS.md',R/'experiments/AGENTS.md',E/'AGENTS.md']:
 if path.exists():print(str(path),path.read_text())
a=read(B/'nestcom_7db_common500_20261030_v1/sources.json');print('sources',type(a).__name__,len(a));print(str(a[0]if isinstance(a,list)else {k:type(v).__name__ for k,v in a.items()})[:4500])

