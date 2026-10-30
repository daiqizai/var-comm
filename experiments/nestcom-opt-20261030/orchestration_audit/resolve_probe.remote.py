from pathlib import Path
import json,hashlib
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
def brief(v):
 if isinstance(v,list):return {'n':len(v),'first':v[0] if v and not isinstance(v[0],(list,dict)) else (brief(v[0]) if v else None)}
 if isinstance(v,dict):return {k:brief(x) for k,x in v.items() if not k.endswith('hashes')}
 return v
paths=[B/'progressive_rd_scaleup_20261022_v1/train_index.json',R/'results/progressive_rd_20261022_v1/stage0_registration.json',R/'results/shared_codeword_fit_20261028_v1/source32_numeric_v1_registration.json',R/'results/nestcom_7db_20261030_v1/calibration_sources.json',R/'results/nestcom_7db_20261030_v1/protocol.json',R/'results/nestcom_7db_common500_20261030_v1/protocol.json',R/'results/nestcom_n512_20261028_v1/protocol.json']
for p in paths:
 v=json.loads(p.read_bytes());print(str(p),json.dumps(brief(v)))
for p in [R/'results/nestcom_n512_20261028_v1',R/'results/nestcom_evidence_20261029_v1',R/'results/nested_residual_rd_20261028_v1',B/'baseline_strength_20261011/inputs',B/'nestcom_n512_20261028_v1']:
 print('DIRECTORY',str(p),[x.name for x in p.iterdir()] if p.exists() else 'MISSING')
print('DATASET_CODE', [str(x) for x in (R/'experiments').iterdir() if any(s in x.name for s in ['baseline','source','imagenet'])])
