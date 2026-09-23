#!/usr/bin/env python3
"""Copy an explicit allowlist of completed lightweight research evidence."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/followup'
DEST=ROOT/'results/review_20260923_phase2'
RUNS={
 'training':('research_20260923_training_v3',['completion.json','qualification.json','registration.json','selected_*.json','calibration/*.json','calibration/*.csv']),
 'diagnostics':('research_20260923_diagnostics',['completion.json','per_frame.csv']),
 'evaluation':('research_20260923_evaluation',['completion.json','registration.json','per_frame.csv','timing.csv','warmup.csv','quality_timing_consistency.csv']),
 'system_policy':('research_20260923_system_policy_v2',['completion.json','frozen_policy.json','digital_reuse_validation.json','calibration_scores.csv','calibration_by_method/*.csv','per_frame.csv','folded_development.csv','digital_adaptive_policies.json','digital_adaptive_per_frame.csv']),
 'digital_strict':('research_20260923_digital_strict',['completion.json','registration.json','runtime_identity.json','numerical_precision_incident.json','development.csv','calibration_by_method/*.csv']),
 'pca_basis':('research_20260923_pca_basis',['projection_identity.json']),
 'pca_evaluation':('research_20260923_pca_evaluation',['completion.json','registration.json','projection.json','calibration_noise_identity.json','per_frame.csv']),
 'comparison':('research_20260923_comparison',['*.json','*.csv','*.png','*.pdf','analysis/*.json','analysis/*.csv','projection_analysis/*.json','projection_analysis/*.csv']),
}
def main():
    if DEST.exists():raise RuntimeError('refuse to overwrite existing published evidence')
    for name,(run,_) in RUNS.items():
        if name not in ('comparison','pca_basis') and not (BASE/run/'completion.json').exists():
            raise RuntimeError('incomplete run: '+run)
    import csv
    digital=BASE/'research_20260923_digital_strict'
    all_cal=list(csv.DictReader((digital/'calibration.csv').open()))
    partitions=digital/'calibration_by_method';partitions.mkdir(exist_ok=True)
    for family in ('raw','arithmetic'):
        for mode in ('7','8','9'):
            part=[r for r in all_cal if r['family']==family and r['mode']==mode]
            if len(part)!=15000:raise RuntimeError('digital partition coverage')
            with (partitions/f'{family}_m{mode}.csv').open('w',newline='') as h:
                writer=csv.DictWriter(h,fieldnames=list(part[0]));writer.writeheader();writer.writerows(part)
    files=[]
    for name,(run,patterns) in RUNS.items():
        for pattern in patterns:
            matches=sorted((BASE/run).glob(pattern))
            if not matches:raise RuntimeError('missing artifact '+run+'/'+pattern)
            for source in matches:
                if source.is_symlink() or source.stat().st_size>10_000_000 or source.suffix not in {'.json','.csv','.png','.pdf'}:
                    raise RuntimeError('excluded publication artifact: '+str(source))
                target=DEST/name/source.relative_to(BASE/run)
                files.append((source,target,run))
    for source,target,run in files:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    receipts=[{'run_id':run,'source':str(s.relative_to(ROOT)),'published_path':str(t.relative_to(ROOT)),
               'bytes':s.stat().st_size,'sha256':hashlib.sha256(s.read_bytes()).hexdigest()} for s,t,run in files]
    (DEST/'artifact_lineage.json').write_text(json.dumps({'phase1_commit':'3f229fa73dd8eee117b746a3ee064eeecc0661c4',
        'real_image_results':True,'new_holdout_used':False,'excluded':'weights, data, tensor caches, raw images, partial outputs',
        'artifacts':receipts},indent=2)+'\n')
    (DEST/'README.md').write_text('# Bounded research after the local review repair\n\n'
        'All quality metrics are from the existing 100-image development population, five SNRs and three seeds. '
        'No new holdout was opened. Training selection and resource policy were frozen using the original 1,000-image calibration population.\n\n'
        'See ../../reports/review_20260923_phase2.md for interpretation and limitations. '
        'artifact_lineage.json records original run IDs, paths and SHA256. Original A/B weights and runs remain unchanged. '
        'The full resource calibration table is preserved locally; its eight exact per-method CSV partitions are published here. '
        'GPU weights/data are intentionally not distributed.\n')
    print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in receipts),'destination':str(DEST)}))
if __name__=='__main__':main()
