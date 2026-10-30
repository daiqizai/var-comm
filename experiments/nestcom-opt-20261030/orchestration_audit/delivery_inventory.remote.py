import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
print('topD',[x.name for x in D.glob('*.json')if any(w in x.name.lower()for w in ['asset','inventory','source','block','protocol','audit','recovery'])])
print('source split',read(D/'protocol_resolved_p0.json')['sources'])
for role in ['benchmark_rx','final_rx']:
 print(role,(L/(role+'_control/worker.log')).read_text()[-200:])
 for name in ['completion.json','failure.json']:
  p=L/role/name
  if p.exists():print(role,name,read(p))
