from pathlib import Path
import json
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
for p in [B/'nestcom_n512_20261028_v1/full_rx_nest_00/outputs.json',B/'nestcom_n512_20261028_v1/full_rx_plan/NEST_00.json',B/'nestcom_n512_20261028_v1/full_rx_plan/reuse.json',B/'nestcom_n512_20261028_v1/formal_plan/logical.json',B/'nestcom_7db_common500_20261030_v1/receiver_plan/NEST.json']:
 if not p.exists():print('MISSING',p);continue
 v=json.loads(p.read_bytes());print('FILE',p,'TYPE',type(v).__name__,'LEN',len(v));print(json.dumps(v[:1] if isinstance(v,list) else {k:(x[:1] if isinstance(x,list) else x)for k,x in list(v.items())[:1]}))
v=json.loads((R/'results/t1b_m10_calibration_20261016_v1/registration.json').read_bytes());print('T1B_KEYS',list(v));print('T1B_CONFIG',json.dumps({k:x for k,x in v.items() if k not in ['code','source_bindings','input_bindings','bindings','sources','frozen_sources']} )[:8000])
v=json.loads((B/'baseline_strength_20261011/manifest.json').read_bytes());print('MANIFESTFIRST',v['files'][:2]);print('INPUTMETA', [str(x)for x in (B/'baseline_strength_20261011').glob('*.json')])
print('RAW_SOURCE_DIRS',[str(p)for p in R.parent.iterdir() if any(x in p.name.lower()for x in ['data','imagenet','runtime'])])
