from pathlib import Path
import json,csv
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
def read(p):return json.loads(Path(p).read_bytes())
v=read('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/baseline_strength_20261011/parents/p/registration.json')
print('TRAIN_PARENT_KEYS',list(v));print('TRAIN_PARENT_SMALL',json.dumps({k:x for k,x in v.items()if len(str(x))<4000}))
v=read(R/'results/progressive_rd_scaleup_20261022_v1/registration.json');print('VAL',len(v['validation_sources']),v['validation_sources'][0]);print('PILOTIDS',len(v['prior_calibration100_ids']))
print('SPLIT_HEADER',next(csv.DictReader((R/'results/nested_residual_rd_20261028_v1/split_manifest.csv').open())))
for folder in ['results/t1b_m10_calibration_20261016_v1','results/nestcom_7db_20261030_v1','results/nextscale_selective_blocks_20261028_v1','results/nestcom_evidence_20261029_v1']:
 print('REGISTRY',folder,[x.name for x in (R/folder).glob('*manifest*')])
for sub in ['pilot_rx_plan','full_rx_plan','formal_rx_plan']:
 v=read(B/'nestcom_n512_20261028_v1'/sub/'reuse.json');print(sub,'REUSE_NEST',sum(x['record']['codec']=='NEST'for x in v.values()),'first',next(iter(v.values()))['origin'])
v=read(B/'nestcom_n512_20261028_v1/formal_rx_plan/frames.json');print('FRAMES_TYPE',type(v).__name__,str(v[0]if isinstance(v,list)else next(iter(v.items())))[:2500])
v=read(R/'results/nestcom_n512_20261028_v1/public_catalogue.compact.json');print('CAT',type(v).__name__,str(v)[:2000])
v=read(R/'results/n1024_consistency_20261019_v1/repair_v1/registration.json');print('POLICY',v.get('policies'),v.get('frozen_policy'),[k for k in v if 'policy' in k or 'catalog' in k]);print('REPAIRINPUTS',json.dumps({k:x for k,x in v.items()if k in ['frozen_policies','catalogue','sources','static','bootstrap']}))
