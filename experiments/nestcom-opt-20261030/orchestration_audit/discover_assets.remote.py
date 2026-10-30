from pathlib import Path
import json,subprocess
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
paths=['results/progressive_rd_scaleup_20261022_v1/registration.json','results/progressive_rd_scaleup_20261022_v1/report_v1/recovery_manifest.json','results/progressive_rd_20261022_v1/registration.json','results/nestcom_7db_20261030_v1/frozen_policies.json','results/nestcom_n512_20261028_v1/frozen_policies.json','results/nestcom_n512_20261028_v1/protocol.json','results/n1024_consistency_20261019_v1/repair_v1/registration.json','results/two_route_confirmation_20261028_v1/registration.json','results/nestcom_evidence_20261029_v1/protocol.json']
for rel in paths:
 p=R/rel
 if not p.exists():print('MISSING',rel);continue
 d=json.loads(p.read_text());print('JSON',rel,'KEYS',list(d))
 if 'frozen_policies'in rel:print(json.dumps(d))
 else:
  print(json.dumps({k:v for k,v in d.items()if any(t in k for t in ['source','split','model','cache','train','valid','calib','asset','weight','policy','runtime','decoder','STOP'])})[:18000])
for root in [R/'results/progressive_rd_20261022_v1',R/'results/progressive_rd_scaleup_20261022_v1',R/'results/shared_codeword_fit_20261028_v1',R/'results/nested_residual_rd_20261028_v1']:
 print('FILES',str(root))
 if root.exists():
  print('\n'.join(str(p) for p in root.rglob('*')if p.is_file() and any(t in p.name for t in ['split','manifest','registration','identity','protocol','STOP']))[:9000])
for rel in ['progressive_rd_20261022_v1','progressive_rd_scaleup_20261022_v1','nestcom_n512_20261028_v1/qualified_sources']:
 p=B/rel;print('PFS_TOP',str(p),[x.name for x in p.iterdir()]if p.exists()else'MISSING')
