"""N2048 confirmation100 source means and predeclared paired comparisons.

Uses the original source-bootstrap functions in a private module instance with
SOURCE_COUNT=100. Function bodies, seed, replicate count and quantiles are intact.
The immutable original500 statistics file and all earlier results stay unchanged.
"""
import argparse, inspect, math
from pathlib import Path
import numpy as np
from t6_score import (COUNT,N,SNRS,SEEDS,METRICS,RAW_WHOLE,RAW_PARTIAL,require,read,write,sha,
    verify,pin,pinned,load_module,csv_read,csv_write,bound_output,frame_grid,metric_value,point,methods,comparisons,digest)

STATISTICS_SHA='e401a77ac216368dc968ce023853ff7cdb2e0a2c94edbd3dd33407c0ba90054a'
REPLICATES=10000
BOOTSTRAP_SEED=2026100701

def original_statistics(path):
    verify(path,STATISTICS_SHA)
    stat=load_module(pin(path),'_t6_private_original_source_statistics_population100')
    require(stat.SOURCE_COUNT==500 and stat.REPLICATES==REPLICATES and stat.BOOTSTRAP_SEED==BOOTSTRAP_SEED,
        'Original bootstrap source configuration differs')
    bodies={name:digest(inspect.getsource(getattr(stat,name))) for name in ('draws','interval')}
    stat.SOURCE_COUNT=COUNT
    require(bodies=={name:digest(inspect.getsource(getattr(stat,name))) for name in bodies},
        'Only source population size may be configured; function bodies are immutable')
    return stat,bodies

def source_means(rows,ids,family):
    keyed=frame_grid(rows,ids,family,scored=True);values={};table=[]
    for method in methods(family):
        for snr in SNRS:
            p=point(method,snr)
            for metric in METRICS:
                a=np.asarray([np.mean([metric_value(keyed[method,snr,i,seed],metric) for seed in SEEDS],dtype=np.float64)
                    for i in range(COUNT)],dtype=np.float64)
                require(a.shape==(COUNT,) and np.isfinite(a).all(),'All100 source means must remain finite')
                values[p,metric]=a
                table.extend(dict(N=N,method_id=method,point_id=p,snr_db=snr,metric=metric,source_index=i,source_id=sid,
                    mean=float(a[i]),source_count=COUNT,noise_count=3,noise_seeds=';'.join(map(str,SEEDS))) for i,sid in enumerate(ids))
    return values,table

def units(metric,paired):
    scale=100. if metric==METRICS[-1] else 1.
    unit={'psnr_db':'dB','lpips_alex':'LPIPS','dinov2_vitl14_cosine':'cosine_similarity',
        'convnext_top1_source_prediction':'absolute_agreement_fraction' if paired else 'agreement_fraction'}[metric]
    shown='percentage_points' if paired else 'percent'
    return dict(unit=unit,display_scale=scale,display_unit=shown if metric==METRICS[-1] else unit,
        improvement_direction='negative' if metric=='lpips_alex' and paired else 'lower' if metric=='lpips_alex' else 'positive' if paired else 'higher')

def decorate(metric,result,paired):
    detail=units(metric,paired);scale=detail['display_scale']
    return dict(result,**detail,display_mean=scale*result['mean'],display_ci_low=scale*result['ci_low'],display_ci_high=scale*result['ci_high'])

def interval_direction(metric,low,high):
    if low==high==0:return 'exact_zero_difference'
    if low<=0<=high:return 'interval_includes_zero'
    improved=high<0 if metric=='lpips_alex' else low>0
    return 'method_better' if improved else 'reference_better'

def summarize(rows,ids,family,stat):
    values,means=source_means(rows,ids,family);samples=stat.draws()
    require(samples.shape==(10000,100),'Registered100-source draw matrix required')
    cache={};calls=0
    def interval(metric,vector):
        nonlocal calls
        key=metric,vector.tobytes()
        if key in cache:return dict(cache[key],interval_provenance='EXACT_SAME_SOURCE_VECTOR_REUSE_WITHIN_T6')
        if np.all(vector==0):
            value=dict(mean=0.,ci_low=0.,ci_high=0.,source_count=COUNT,noise_count=3,frame_count=300,
                bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=REPLICATES,
                bootstrap_unit='source after registered3-noise mean',interval_provenance='EXACT_ZERO_SOURCE_VECTOR')
        else:
            value=stat.interval(vector,samples);calls+=1
            # Original function's fixed500 metadata label is adapted, not its estimates.
            require(value['source_count']==100 and value['noise_count']==3 and value['frame_count']==1500,
                'Unexpected original bootstrap result metadata')
            value.update(frame_count=300,bootstrap_unit='source after registered3-noise mean',
                interval_provenance='ORIGINAL_BOOTSTRAP_FUNCTION_POPULATION100')
        cache[key]=value;return dict(value)
    summary=[];paired=[]
    for method in methods(family):
        for snr in SNRS:
            p=point(method,snr)
            for metric in METRICS:
                summary.append(dict(N=N,method_id=method,point_id=p,snr_db=snr,metric=metric,
                    **decorate(metric,interval(metric,values[p,metric]),False)))
    for a,b in comparisons(family):
        for snr in SNRS:
            pa,pb=point(a,snr),point(b,snr)
            for metric in METRICS:
                result=interval(metric,values[pa,metric]-values[pb,metric])
                paired.append(dict(N=N,method=pa,reference=pb,method_id=a,reference_method_id=b,snr_db=snr,metric=metric,
                    delta_definition='method minus reference',frame_level_noise_pairing_claimed=False,
                    interval_direction=interval_direction(metric,result['ci_low'],result['ci_high']),
                    **decorate(metric,result,True)))
    require(len(summary)==36 and len(paired)==36 and len(means)==3600,'All declared T6 comparisons required')
    return summary,paired,means,calls

def run(args):
    score=read(args.score_completion)
    require(score['status']=='T6_CONFIRMATION100_FOUR_METRICS_COMPLETE' and score['N']==N
        and score['source_count']==COUNT and score['noise_count']==3 and score['frame_count']==2700
        and score['noise_seeds']==SEEDS and score['snrs']==SNRS and score['actual_children_waited'] is True
        and score['worker_exit_codes']==[0] and score['selection_used_confirmation'] is False,
        'Actual complete, waited, frozen confirmation100 metrics required')
    frozen=pinned(score['calibration_freeze']);selected=pinned(score['entropy_family_selection']);source=pinned(score['source_manifest'])
    require(frozen['status']=='T6_N2048_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and frozen['selection_used_confirmation'] is False
        and frozen['holdout_used_for_selection'] is False and selected['holdout_used_for_selection'] is False
        and frozen['entropy_family_selection']==score['entropy_family_selection']
        and selected['family']==score['family'],'No confirmation-driven policy or family selection')
    ids=score['source_ids'];require(ids==source['source_ids'] and len(ids)==len(set(ids))==COUNT,'Exact new100 paired identities')
    request=read(score['request_path']);verify(score['request_path'],score['request_sha256'])
    require(request['comparisons']==score['comparisons']==comparisons(score['family'])
        and request['methods']==score['methods']==methods(score['family']),'Three comparisons fixed before scoring')
    pinned(score['parent_launch']);observed=pinned(score['parent_exit'])
    require(observed['actual_child_waited'] is True and observed['exit_code']==0,'Actual score owner exit proof required')
    rows_path=bound_output(score,'per_frame.csv');rows=csv_read(rows_path)
    require(all(row['metric_evaluator_identity']==score['metric_evaluator_identity'] for row in rows),'No evaluator mixture')
    frame_grid(rows,ids,score['family'],scored=True)
    out=Path(args.out).resolve();require(not out.exists(),'Fresh immutable T6 statistics version required')
    stat,bodies=original_statistics(args.statistics_module)
    registration=dict(schema='T6_N2048_CONFIRMATION100_STATISTICS_REQUEST_V1',score_completion=pin(args.score_completion),
        source_manifest=score['source_manifest'],calibration_freeze=score['calibration_freeze'],entropy_family_selection=score['entropy_family_selection'],
        source_ids=ids,N=N,source_count=100,noise_count=3,noise_seeds=SEEDS,snrs=SNRS,methods=score['methods'],
        comparisons=score['comparisons'],metrics=METRICS,source_bootstrap_replicates=REPLICATES,bootstrap_seed=BOOTSTRAP_SEED,
        original_statistics_module=pin(args.statistics_module),original_function_source_sha256=bodies,
        private_module_population_override={'SOURCE_COUNT':100},metadata_only_override={'frame_count':300,'bootstrap_unit':'source after registered3-noise mean'},
        worker_sha256=sha(__file__),source_bindings={str(Path(__file__).resolve()):sha(__file__),
            str(Path(__file__).with_name('t6_score.py').resolve()):sha(Path(__file__).with_name('t6_score.py'))},
        selection_used_confirmation=False,multiple_comparison_adjustment=False,old_intervals_modified=False)
    out.mkdir(parents=True);write(out/'request.json',registration)
    summary,paired,means,calls=summarize(rows,ids,score['family'],stat)
    failures=[]
    for method in score['methods']:
        for snr in SNRS:
            allrows=[r for r in rows if r['method_id']==method and int(r['snr_db'])==snr]
            for status in sorted({r['status'] for r in allrows}):
                subset=[r for r in allrows if r['status']==status]
                failures.append(dict(N=N,method_id=method,snr_db=snr,status=status,frame_count=len(subset),
                    sources_with_state=len({r['source_id'] for r in subset}),total_source_count=100,noise_count=3,
                    total_frame_count=300,quality_rows_included=len(subset),selection_used_confirmation=False))
    outputs={}
    for name,table in [('summary.csv',summary),('paired.csv',paired),('source_means.csv',means),('failure_breakdown.csv',failures)]:
        p=out/name;csv_write(p,table);outputs[str(p)]=sha(p)
    labels={RAW_WHOLE:'Complete-scale raw transmission with VAR completion',RAW_PARTIAL:'Partial-scale raw transmission with VAR completion (NeST-Com)',
        score['methods'][2]:'VAR-conditional entropy-coded complete-scale transmission with VAR completion' if score['family']=='EC_VAR_WHOLE'
        else 'Static entropy-coded complete-scale transmission with VAR completion'}
    points={point(method,snr):dict(method_id=method,public_name=labels[method],N=N,snr_db=snr,source_count=100,noise_count=3,
        noise_seeds=SEEDS,calibration_freeze=score['calibration_freeze'],metrics=METRICS) for method in score['methods'] for snr in SNRS}
    pairdefs=[dict(method=point(a,s),reference=point(b,s),delta_definition='method minus reference')
        for a,b in score['comparisons'] for s in SNRS]
    for name,value in [('points.json',points),('pairs.json',pairdefs)]:write(out/name,value);outputs[str(out/name)]=sha(out/name)
    text=['# N2048 confirmation100: frozen digital branches','',
        'Three methods, SNR 4/10/19 dB, 100 preregistered sources and seeds 9201/9202/9203: all 2,700 method-frames are retained.',
        'Every source is averaged over its three noises before inference across sources. The three paired comparisons were fixed before scoring: partial minus whole, entropy minus whole, and partial minus entropy.',
        '', 'The original SuiteBackend evaluates the actual reconstructed RGB against each new confirmation source. References are read from the new100 authenticated uint8 archives and converted exactly to float32 / 255; no original500 feature or quality cache is consumed.',
        '', 'The unchanged original bootstrap functions are imported privately with SOURCE_COUNT=100. The seed is 2026100701 and there are 10,000 source resamples; 2.5/97.5 percentiles give pointwise 95% intervals without multiplicity adjustment. Only returned population metadata is adapted from 1,500 to 300 frames. The original file and prior intervals are unchanged. Exact zero increments remain zero; identical source vectors reuse their within-run intervals.',
        '', 'LPIPS deltas retain method minus reference; a negative value is better. ConvNeXt measures agreement with the source prediction, not classification accuracy. Canonical means and deltas are fractions; display_mean/display_ci_low/display_ci_high multiply all agreement values by 100, yielding percent for summaries and percentage points for differences. The CSV states both units explicitly.',
        '', 'Pairing is by source, not a claim that different transmitted waveforms share identical received observations. Failed outputs remain in quality means. Zero increments and fallback to complete-scale configurations are legitimate measured outcomes.',
        '', f"The duplicate audit covered {score['available_reference_hash_count']} available prior content hashes; {score['unavailable_reference_hash_count']} prior hashes were unavailable. Its scope is `{score['content_duplicate_audit_scope']}`. This does not establish that these images were unseen in every historical context.",
        '', 'The selected entropy family is '+score['family']+'. Its identity and all N2048 policies were frozen using original calibration, without using confirmation outcomes.',
        '', 'Only this second tested budget is supported. Two budget points cannot identify an equal-quality bandwidth saving or justify interpolation.',
        '', '## DINOv2-L source-paired differences','', '| Method minus reference | SNR (dB) | Mean | 95% CI | Interpretation |','|---|---:|---:|---|---|']
    for row in paired:
        if row['metric']!='dinov2_vitl14_cosine':continue
        text.append(f"| {labels[row['method_id']]} minus {labels[row['reference_method_id']]} | {row['snr_db']} | {row['mean']:.8f} | [{row['ci_low']:.8f}, {row['ci_high']:.8f}] | {row['interval_direction']} |")
    text+=['','All four metrics and their original-precision paired intervals are in the CSV files. An interval containing zero is not evidence of equivalence.','']
    with (out/'REPORT_T6.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text))
    outputs[str(out/'REPORT_T6.md')]=sha(out/'REPORT_T6.md')
    result=dict(status='T6_N2048_CONFIRMATION100_SOURCE_PAIRED_STATISTICS_COMPLETE',request=pin(out/'request.json'),
        source_count=100,noise_count=3,frame_count=2700,N=N,snrs=SNRS,noise_seeds=SEEDS,source_ids=ids,
        family=score['family'],methods=score['methods'],comparisons=score['comparisons'],summary_rows=36,paired_rows=36,source_mean_rows=3600,
        source_mean_first=True,bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=REPLICATES,new_interval_calls=calls,
        original_function_source_sha256=bodies,private_population_size=100,old_statistics_file_unchanged=sha(args.statistics_module)==STATISTICS_SHA,
        multiple_comparison_adjustment=False,selection_used_confirmation=False,holdout_used_for_selection=False,
        source_manifest=score['source_manifest'],calibration_freeze=score['calibration_freeze'],entropy_family_selection=score['entropy_family_selection'],
        metric_evaluator_identity=score['metric_evaluator_identity'],content_duplicate_check_completion=score['content_duplicate_check_completion'],
        new_model_calls=0,new_packet_decodes=0,input_bindings={str(Path(p).resolve()):sha(p) for p in (args.score_completion,rows_path,args.statistics_module)},outputs=outputs)
    write(out/'completion.json',result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('score-completion','statistics-module','out'):parser.add_argument('--'+name,required=True)
    print(run(parser.parse_args())['status'])
if __name__=='__main__':main()
