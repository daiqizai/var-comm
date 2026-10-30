import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
r=cr(read(D/'protocol_resolved_p0.json')['metrics']);c=r['config']['config']['metric_config'];print('metrickeys',list(c));print('weights',c.get('weights'));print('assets',c.get('assets'))
for role in ['benchmark_rx','final_rx']:
 p=L/role/'completion.json'
 print(role,'COMPLETE'if p.exists()else'RUNNING',(L/(role+'_control/worker.log')).read_text()[-170:])
 if p.exists():print(read(p))
