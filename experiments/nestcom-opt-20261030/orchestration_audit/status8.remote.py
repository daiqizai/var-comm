import sys,sqlite3
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
r=cr(read(D/'p0_registration.json')['model_routes'])
for v in r['author_files']:
 if 'basic_var' in v['path']:
  s=checked(v).read_text();print('\n'.join(x for x in s.splitlines()if any(y in x for y in ['attention','flash','slow_attn','using_sdpa'])))
for role in ['fit_D_rx','fit_B']:
 print(role)
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json',L/role/'completion.json']:
  if p.exists():print(p.name,read(p))
 print((L/(role+'_control')/'worker.log').read_text()[-300:])
