import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print('CHECKER', (R/'tools/verify_repository.py').read_text()[-18000:])
for x in ['benchmark_protocol.json','source_review_stage.json','repository_checks.json','CPU_checks_visual_environment.json','repository_verify_staged_source.json']:
 v=read(D/x);print(x,str(v)[:2200])
for p in sorted(D.iterdir()):
 if p.is_dir()and 'delivery'in p.name:print('delivery',p.name)
print('AGENTS nested',subprocess.check_output(['find',str(E.parent),str(D),'-name','AGENTS.md','-print'],text=True))
