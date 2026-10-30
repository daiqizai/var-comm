from pathlib import Path
import json,hashlib
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
def brief(x,depth=0):
 if isinstance(x,list):return dict(count=len(x),first=brief(x[0],depth+1)if x else None)
 if isinstance(x,dict):return {k:brief(v,depth+1)for k,v in x.items()} if depth<4 else dict(keys=list(x))
 return x
rels=['results/progressive_rd_scaleup_20261022_v1/registration.json','results/progressive_rd_scaleup_20261022_v1/report_v1/recovery_manifest.json','results/progressive_rd_20261022_v1/decoder_pilot_registration.json','results/progressive_rd_20261022_v1/stage0_registration.json','results/shared_codeword_fit_20261028_v1/source32_numeric_v1_registration.json','results/nestcom_7db_20261030_v1/calibration_sources.json','results/layered_entropy_feasibility_20261014_v1/assets/model_routes.json','results/nestcom_n512_20261028_v1/source_route_amendment_v2.json','results/nestcom_n512_20261028_v1/formal_metrics_registration.json']
for rel in rels:
 p=R/rel
 if p.exists():print(rel,json.dumps(brief(json.loads(p.read_text())),ensure_ascii=False))
 else:print('MISSING',rel)
for rel in ['progressive_rd_scaleup_20261022_v1/train_index.json','nestcom_n512_20261028_v1/qualified_sources/cal1000.json','progressive_rd_20261022_v1/seed_index_v1.json']:
 p=B/rel;print(rel,json.dumps(brief(json.loads(p.read_text())),ensure_ascii=False))
for seed in [2026102201,2026102202]:
 p=B/f'progressive_rd_scaleup_20261022_v1/training_v1/multiprefix_full_anchor_s{seed}/step_010000.pt';print('ANCHOR_EXISTS',str(p),p.exists(),p.is_symlink(),p.stat().st_size if p.exists()else None)
for parent in [R/'experiments',R/'results',R/'paper']:
 print('NESTED_AGENTS',str(parent),[str(p)for p in parent.glob('AGENTS.md')])
