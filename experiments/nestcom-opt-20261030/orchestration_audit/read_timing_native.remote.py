import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for rel in ['experiments/five-method-timing-20261020/native.py','experiments/five-method-timing-20261020/library_recovery_v1/adapter.py','experiments/kodak-current-20261026/nest_v1/kn_native.py']:
 s=(R/rel).read_text();print('FILE',rel)
 if 'kn_native' in rel:print(s[:17000])
 else:
  lines=s.splitlines()
  for i,l in enumerate(lines):
   if any(k in l for k in ('def nest','def tokenize','def transmitter','def receiver','def _nest','img_to_idx','native_tx','encode_endpoints','def _source')):print('\n'.join(lines[max(0,i-2):i+45]))
