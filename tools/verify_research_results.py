#!/usr/bin/env python3
"""Asset-free acceptance of published real research evidence, not GPU inference."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/review_20260923_phase2'
def read(p):return list(csv.DictReader(p.open()))
def main():
    from collect_research_comparison import require_grid
    index=json.loads((R/'artifact_lineage.json').read_text())
    for rec in index['artifacts']:
        p=ROOT/rec['published_path']
        assert p.is_relative_to(R) and p.is_file() and not p.is_symlink()
        assert p.stat().st_size==rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256']
    data=read(R/'comparison/per_frame.csv');require_grid(data,7)
    projection=read(R/'comparison/projection_per_frame.csv');require_grid(projection,4)
    require_grid(read(R/'diagnostics/per_frame.csv'),5)
    ids={}
    for r in data:
        assert int(r['N'])==4084 and int(r['E'])==8168
        i=int(r['source_index'])
        assert ids.setdefault(i,r['image_id'])==r['image_id']
    for prefix,records in [('analysis',data),('projection_analysis',projection)]:
        for r in read(R/'comparison'/prefix/'summary.csv'):
            part=[x for x in records if x['method']==r['method']]
            assert len(part)==1500
            for k in ('psnr_db','lpips_alex','dino_cosine'):
                np.testing.assert_allclose(float(r[k]),np.mean([float(x[k]) for x in part]),atol=1e-10,rtol=0)
    done=json.loads((R/'training/completion.json').read_text())
    assert done['state']['updates']=={'pure_continuous':10000,'full_tx_control':5000,'light_tx':5000}
    curves=[json.loads(p.read_text()) for p in (R/'training/calibration').glob('*.json')]
    for name,rec in done['selected'].items():
        candidates=[(c['summary'][name],c['step']) for c in curves if name in c['summary']]
        assert (rec['utility'],rec['step'])==min(candidates)
        assert rec['arm_key']==name and len(rec['checkpoint_sha256'])==64
    numerical=json.loads((R/'digital_strict/registration.json').read_text())
    assert numerical['precision']=={'matmul_tf32':False,'cudnn_tf32':False}
    assert numerical['N']==4084 and numerical['renderer']=='Dc'
    for path in (R/'digital_strict/calibration_by_method').glob('*.csv'):
        rows=read(path)
        assert len(rows)==15000 and len({(r['image_index'],r['snr_db'],r['seed']) for r in rows})==15000
    assert len(read(R/'digital_strict/development.csv'))==9000
    policy=json.loads((R/'system_policy/frozen_policy.json').read_text());scores=read(R/'system_policy/calibration_scores.csv')
    assert policy['precision']==numerical['precision']
    assert policy['decoder_state_sha256']==numerical['decoder_state_sha256']
    for snr,action in policy['actions'].items():
        values=[(float(r['utility']),r['method']) for r in scores if float(r['snr_db'])==float(snr)]
        assert min(values)[1]==action and len(values)==8
    expected_cal={(i,s,n) for i in range(1000) for s in (1.,4.,7.,13.,19.) for n in (4101,4102,4103)}
    for p in (R/'system_policy/calibration_by_method').glob('*.csv'):
        rows=read(p);keys=[(int(r['source_index']),float(r['snr_db']),int(r['seed'])) for r in rows]
        assert len(keys)==len(expected_cal) and set(keys)==expected_cal
    consistency=read(R/'evaluation/quality_timing_consistency.csv')
    assert len(consistency)==300 and all(float(r['max_abs_error'])<=2e-5 for r in consistency)
    identity=json.loads((R/'comparison/source_identity.json').read_text())
    assert len(identity['train'])==20000 and len(identity['calibration'])==1000 and len(identity['development'])==100
    assert all(identity['development'][i]==v for i,v in ids.items())
    assert not (set(identity['train'])&set(identity['calibration']))
    assert not (set(identity['train'])&set(identity['development']))
    assert not (set(identity['calibration'])&set(identity['development']))
    print(json.dumps({'status':'PUBLISHED_RESEARCH_EVIDENCE_PASS','files':len(index['artifacts']),
                      'quality_rows':len(data),'GPU_inference':'NOT_RUN_BY_THIS_CHECK','new_holdout_used':False}))
if __name__=='__main__':main()
