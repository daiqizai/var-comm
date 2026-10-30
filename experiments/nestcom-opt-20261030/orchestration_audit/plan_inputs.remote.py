import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for p in [D/'protocol_resolved_p0.json',R/'results/t1b_m10_calibration_20261016_v1/delivery/frozen_policies.json',R/'results/nestcom_7db_20261030_v1/frozen_policies.json',R/'results/nestcom_n512_20261028_v1/frozen_policies.json']:
 v=read(p);print(str(p),json.dumps(v)[:24000])
for p in [B/'nestcom_n512_20261028_v1/full_rx_plan/frames.json',B/'nestcom_n512_20261028_v1/full_rx_plan/reuse.json',B/'baseline_strength_20261011/t1b_m10_calibration_20261016_v1/preparation/old_packet_index.json']:
 v=read(p);x=next(iter(v.items()))if isinstance(v,dict)else v[0];print('SHAPE',p,type(v).__name__,len(v),'EXAMPLE',str(x)[:1800])
print('ASSET_TOP',[(str(p),[c.name for c in p.iterdir()][:30])for p in [RT/'assets/var_native_source_seed_v1',RT/'assets/ep_new100_source_code_seed_v1']])
for p in R.glob('results/*/source*'):
 if p.is_file()and p.suffix=='.json':print('SOURCE_CAND',str(p))
