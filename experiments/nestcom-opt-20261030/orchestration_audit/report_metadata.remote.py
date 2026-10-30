import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
r=cr(read(D/'protocol_resolved_p0.json')['metrics']);c=r['config']['config']['metric_config'];print(cr(c['weight_receipt']))
print('source manifest keys',list(read(D/'source_manifest.json')))
print('source_status', {k:v for k,v in read(D/'source_manifest.json').items()if k not in ['TRAIN','FIT','CONFIRM','training_source_ids','eligible_sources']and len(str(v))<4000})
