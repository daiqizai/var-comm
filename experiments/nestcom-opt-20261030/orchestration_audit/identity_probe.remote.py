from pathlib import Path
import json,hashlib
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
def show(p):
 v=json.loads(p.read_bytes());print('FILE',str(p));print(json.dumps(v if p.stat().st_size<25000 else ({'keys':list(v),'samples':{k:(x[:1] if isinstance(x,list) else x) for k,x in v.items() if k in ['sources','source_ids','splits','dataset','source_root','inputs','parents','output']}} if isinstance(v,dict) else v[:1])))
for s in ['results/nextscale_selective_blocks_20261028_v1/source_manifest.json','results/nestcom_evidence_20261029_v1/three_metrics_registration.json','results/nestcom_evidence_20261029_v1/clip_metrics_path_v2_registration.json','results/checkpoint_relocation_20261028_v1/model_routes.json','results/nested_residual_rd_20261028_v1/data_identity.json']:
 show(R/s)
show(B/'baseline_strength_20261011/manifest.json')
for p in (R/'experiments/baseline-strength-20261011').glob('*.py'):print('CODE',p.name)
for s in ['full_rx_nest_00','formal_rx_nest_00','full_rx_plan','formal_rx_plan','route_e2e','formal_plan']:
 p=B/'nestcom_n512_20261028_v1'/s;print('FILES',s,[x.name for x in p.iterdir()][:14])
for p in [B/'nestcom_n512_20261028_v1/formal_rx_nest_00/rows.json',B/'nestcom_n512_20261028_v1/formal_rx_nest_00/results.json']:
 if p.exists():show(p)
for p in (R/'results/checkpoint_relocation_20261028_v1').glob('*.json'):print('RELOCATION',p.name)
