"""Cold publication audit repair; original replay, scoring and R1 files remain frozen.

The legacy Phase2/grid tables have exactly three quality metrics. The original
final audit iterated four tolerances, incorrectly requiring an unavailable
legacy mismatch delta. This version audits the three recorded metrics with
unchanged limits and also checks their full historical/new delta closure.
It never rewrites a checkpoint, image, metric, queue or original source.
"""
from pathlib import Path
import math
from history_common import read, sha, verify, identity, row_ids, checkpoint_valid
from phase2_completed import study_paths, STUDY, OLD, require


LEGACY_METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')


def verify_metric_closure(parity, metrics, limits):
    """Only legacy-recorded quantities; new independent mismatch stays separate."""
    require(tuple(metrics) == LEGACY_METRICS, 'Original three-metric inventory changed')
    wanted = set(metrics)
    mappings = {key: parity.get(key, {}) for key in
        ('measured_native_grid_metrics', 'measured_grid_minus_original_grid',
         'measured_grid_minus_original_phase2')}
    alias = parity.get('historical_alias', {})
    mappings.update({key: alias.get(key, {}) for key in
        ('grid_original_metrics', 'phase2_original_metrics', 'historical_grid_minus_phase2')})
    for key, values in mappings.items():
        require(isinstance(values, dict) and set(values) == wanted,
                'Recorded legacy metric keys differ: ' + key)
        require(all(math.isfinite(float(v)) for v in values.values()),
                'Nonfinite legacy metric evidence: ' + key)
    for metric in metrics:
        measured = float(mappings['measured_native_grid_metrics'][metric])
        grid = float(mappings['grid_original_metrics'][metric])
        phase2 = float(mappings['phase2_original_metrics'][metric])
        residual = float(mappings['measured_grid_minus_original_grid'][metric])
        direct = float(mappings['measured_grid_minus_original_phase2'][metric])
        historical = float(mappings['historical_grid_minus_phase2'][metric])
        require(abs(residual) <= limits[metric],
                'New pure grid metric exceeded its unchanged strict tolerance')
        require(all(abs(value) <= 1e-12 for value in
                    (measured-grid-residual, measured-phase2-direct,
                     grid-phase2-historical, direct-(historical+residual))),
                'Recorded historical/new metric differences do not close')


def verify_complete_phase2(root,queue):
    """Final CPU gate: all original 81 payloads plus 19 new strict-grid payloads."""
    from phase2_inheritance import inherited_checkpoint
    from historical_phase2_alias import create_adapter, LIMITS, METRICS
    from history_float_cache import verify_saved
    root=Path(root).resolve();out,result=study_paths(root,queue,STUDY)
    rp=result/'registration.json';reg=read(rp);donepath=out/'completion.json';done=read(donepath)
    require(done.get('status')=='HISTORICAL_STUDY_METRICS_COMPLETE' and done.get('frames')==4500
        and done.get('sources')==100 and done.get('inherited_sources')==81 and done.get('new_sources')==19
        and done.get('inherited_rows')==3645 and done.get('new_rows')==855
        and done.get('new_metric_offset_applied') is False and done.get('original_r3_files_written') is False,
        'Recovery Phase2 total/new/inherited coverage differs')
    inherited_path=Path(queue['phase2_recovery']['inheritance_path'])
    verify({str(inherited_path):queue['phase2_recovery']['inheritance_sha256']})
    inherited=read(inherited_path);verify(inherited['bindings'])
    aliaspath=result/'historical_numerical_alias.json';adapter=create_adapter(root,STUDY)
    require(read(aliaspath)==adapter.historical_alias_manifest() and sha(aliaspath)==done['historical_numerical_alias_sha256']
        ==reg['historical_numerical_alias_sha256'],'Complete 100-source historical alias disclosure differs')
    fields=('modelmanifest_sha256','evaluator_identity','numerical_runtime','metric_batch_size',
            'batch_qualification_sha256','metric_qualified_batch_sizes')
    evaluation={k:reg[k] for k in fields}
    expected_new={out/'source_checkpoints'/f'{i:04d}.json' for i in range(81,100)}
    require(set((out/'source_checkpoints').glob('*.json'))==expected_new,'Recovery must contain exactly 19 new checkpoints')
    bindings={str(p):sha(p) for p in (rp,donepath,inherited_path,aliaspath)}
    bindings.update(inherited['bindings']);fresh_pure=0
    for i in range(100):
        rows=list(adapter.expected_rows(i))
        if i<81:
            cp=inherited_checkpoint(inherited,i,rows)
        else:
            path=out/'source_checkpoints'/f'{i:04d}.json'
            cp=checkpoint_valid(read(path),identity(reg),row_ids(STUDY,i,rows),expected_rows=rows,
                study=STUDY,source_index=i,evaluation_identity=evaluation,
                qualified_batch_sizes=reg['metric_qualified_batch_sizes'])
            verify_saved(cp['float_reconstructions'],cp['rows'])
            proof=cp['float_reconstructions']
            require(proof['path']==str(out/'reconstructions'/f'{i:04d}.npz'),'New float cache path differs')
            bindings[str(path)]=sha(path);bindings[proof['path']]=proof['sha256']
            for row,parity in zip(cp['rows'],cp['parity']):
                if row['method']!='pure_continuous':continue
                native=parity.get('native_grid_replay_proof',{})
                require(parity.get('new_metric_offset_applied') is False
                    and parity.get('original_phase2_pixel_identity_claimed') is False
                    and parity.get('original_phase2_scalar_parity_claimed') is False
                    and native.get('passed') is True and native.get('replay_parity_passed') is True,
                    'New pure replay lacks explicit strict-grid/historical-alias proof')
                verify_metric_closure(parity, METRICS, LIMITS)
                fresh_pure+=1
    require(fresh_pure==285,'Fresh pure-grid proof coverage differs')
    # Bind the actual old qualification, without claiming it was recomputed.
    original_qualification=root/'outputs'/OLD/STUDY/'first_source_qualification.json'
    bindings[str(original_qualification)]=sha(original_qualification)
    return dict(status='PHASE2_RECOVERY_100_SOURCES_VERIFIED',sources=100,rows=4500,
        inherited_sources=81,inherited_rows=3645,new_sources=19,new_rows=855,
        new_pure_strict_grid_proofs=285,historical_alias_rows_disclosed=1500,
        new_metric_offset_applied=False,original_files_written=False,synthetic=False,
        bindings=bindings)
