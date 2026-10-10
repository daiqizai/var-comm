"""New source-paired entropy comparisons using the unchanged common500 bootstrap.

Only new entropy summary intervals and new pre-frozen comparisons are computed.
Original raw source means and summary intervals are read and copied unchanged.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
from t1_entropy_core import read,require,sha,verify,write,csv_write
from t1_holdout_metadata import FAMILIES,SNRS,SEEDS,pinned,RAW_REFS,RAW_DROP,paired_prefixes

METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
STATISTICS_SHA='e401a77ac216368dc968ce023853ff7cdb2e0a2c94edbd3dd33407c0ba90054a'
RAW=RAW_REFS;NEW_METHODS=FAMILIES+RAW_DROP

def metric_value(row,metric):
    value=row[metric]
    if metric=='convnext_top1_source_prediction' and value in ('True','False'):
        return 1. if value=='True' else 0.
    return float(value)

def bound_output(done,filename):
    pairs=[(p,h) for p,h in done['outputs'].items() if Path(p).name==filename]
    require(len(pairs)==1,'Ambiguous completed output '+filename);p,h=pairs[0];verify(p,h);return Path(p)

def run(args):
    out=Path(args.out);require(not out.exists(),'Fresh new-statistics output required')
    scored=read(args.score_completion)
    require(scored['status']=='T1_POSTHOC_COMMON500_FOUR_METRICS_COMPLETE' and scored['source_count']==500
        and scored['frame_count']==18000 and scored['holdout_used_for_selection'] is False,
        'Complete post-freeze actual four-metric results required')
    policy=pinned(scored['freeze'])
    require(policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and not policy['holdout_used_for_selection'],
        'Calibration-only frozen policies required')
    require(scored['original_statistics_completion']==dict(path=str(Path(args.original_statistics_completion).resolve()),sha256=sha(args.original_statistics_completion)),
        'Original statistics must match the completed score binding')
    original=read(args.original_statistics_completion)
    require(original['scientific_statistics_completed'] and original['source_count']==500
        and original['source_ids']==scored['source_ids'],'Exact original500 ordered source pairing required')
    ids=scored['source_ids'];require(len(ids)==len(set(ids))==500,'Complete unique500 required')
    rows_path=bound_output(scored,'per_frame.csv')
    with rows_path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    expected={(family,snr,i,seed) for family in NEW_METHODS for snr in SNRS for i in range(500) for seed in SEEDS}
    keyed={}
    for row in rows:
        key=(row['family'],int(row['snr_db']),int(row['source_index']),int(row['noise_seed']))
        require(key not in keyed and key in expected and row['source_id']==ids[key[2]],'Duplicate or mismatched actual frame')
        require(row['point_id']==key[0]+'_SNR_'+str(key[1]),'Point identity differs')
        require(row['metric_evaluator_identity']==scored['metric_evaluator_identity'],'Metric evaluator differs')
        require(all(math.isfinite(metric_value(row,m)) for m in METRICS),'Nonfinite metric cannot be silently omitted')
        require(metric_value(row,'convnext_top1_source_prediction') in (0.,1.),'Agreement must be an event, not class accuracy')
        keyed[key]=row
    require(set(keyed)==expected and len(rows)==18000,'All failures and complete500x3x3x4 grid must be retained')
    old_means=bound_output(original,'source_means.json.gz');old_summary=bound_output(original,'summary.json')
    old_points=bound_output(original,'points.json');points=read(old_points);original_summaries=read(old_summary)
    values={};newmeans=[]
    for family in NEW_METHODS:
        for snr in SNRS:
            point=family+'_SNR_'+str(snr)
            for metric in METRICS:
                a=np.array([np.mean([metric_value(keyed[family,snr,i,seed],metric) for seed in SEEDS],dtype=np.float64) for i in range(500)])
                values[point,metric]=a
                newmeans.extend(dict(point_id=point,metric=metric,source_index=i,source_id=sid,mean=float(a[i])) for i,sid in enumerate(ids))
    refs={ref+'_SNR_'+str(snr) for ref in RAW for snr in SNRS};old_values={}
    with gzip.open(old_means,'rt',encoding='utf-8') as f:
        for row in json.load(f):
            if row['point_id'] not in refs or row['metric'] not in METRICS:continue
            key=row['point_id'],row['metric'],int(row['source_index'])
            require(key not in old_values and row['source_id']==ids[key[2]],'Original source mean alignment differs')
            old_values[key]=float(row['mean'])
    require(len(old_values)==12000,'Complete two-raw-reference source means required')
    for point in refs:
        for metric in METRICS:
            require(points[point]['metric_identity'][metric]==scored['metric_identity'][metric], 'Metric definitions differ')
            values[point,metric]=np.asarray([old_values[point,metric,i] for i in range(500)],dtype=np.float64)
    verify(args.statistics_module,STATISTICS_SHA)
    spec=importlib.util.spec_from_file_location('_wcl_unchanged_common500_statistics',args.statistics_module)
    stat=importlib.util.module_from_spec(spec);sys.modules[spec.name]=stat;spec.loader.exec_module(stat)
    require(stat.SOURCE_COUNT==500 and stat.REPLICATES==10000 and stat.BOOTSTRAP_SEED==2026100701,'Original source-bootstrap contract changed')
    samples=stat.draws();summaries=[];paired=[];interval_cache={};new_interval_calls=0
    interval_keys=('mean','ci_low','ci_high','source_count','noise_count','frame_count','bootstrap_seed','bootstrap_replicates','bootstrap_unit')
    for oldrow in original_summaries:
        if oldrow['point_id'] not in refs or oldrow['metric'] not in METRICS:continue
        key=(oldrow['metric'],values[oldrow['point_id'],oldrow['metric']].tobytes())
        item={k:oldrow[k] for k in interval_keys}
        if key in interval_cache:require(interval_cache[key][0]==item,'Identical old source values have different frozen intervals')
        interval_cache[key]=(item,'EXACT_ORIGINAL_SOURCE_VECTOR_INTERVAL_REUSE')
    def interval(metric,vector):
        nonlocal new_interval_calls
        key=(metric,vector.tobytes())
        if key in interval_cache:
            item,origin=interval_cache[key];return dict(item,interval_provenance=origin)
        if not np.any(vector):
            item=dict(mean=0.,ci_low=0.,ci_high=0.,source_count=500,noise_count=3,frame_count=1500,
                bootstrap_seed=2026100701,bootstrap_replicates=10000,bootstrap_unit='source after original3-noise mean')
            origin='EXACT_ZERO_SOURCE_VECTOR_NO_RESAMPLING'
        else:
            item=stat.interval(vector,samples);new_interval_calls+=1;origin='NEW_COMPARISON_REGISTERED_SOURCE_BOOTSTRAP'
        interval_cache[key]=(item,'EXACT_SAME_NEW_SOURCE_VECTOR_INTERVAL_REUSE')
        return dict(item,interval_provenance=origin)
    for family in NEW_METHODS:
        for snr in SNRS:
            point=family+'_SNR_'+str(snr)
            for metric in METRICS:
                summaries.append(dict(point_id=point,snr_db=snr,metric=metric,metric_identity=scored['metric_identity'][metric],
                    **interval(metric,values[point,metric])))
    expected_pairs=paired_prefixes()
    require(policy['new_paired_prefixes']==expected_pairs,'New comparisons must be frozen before holdout')
    for a,b in expected_pairs:
        for snr in SNRS:
            method=a+'_SNR_'+str(snr);reference=b+'_SNR_'+str(snr)
            for metric in METRICS:
                paired.append(dict(method=method,reference=reference,snr_db=snr,metric=metric,
                    delta_definition='method minus reference',frame_level_noise_pairing_claimed=False,
                    metric_identity=scored['metric_identity'][metric],**interval(metric,values[method,metric]-values[reference,metric])))
    copied=[r for r in original_summaries if r['point_id'] in refs and r['metric'] in METRICS]
    require(len(copied)==24,'Exactly24 old summary rows required')
    out.mkdir(parents=True)
    outputs={}
    for name,table in [('summary.csv',summaries),('source_means.csv',newmeans),('paired.csv',paired)]:
        path=out/name;csv_write(path,table);outputs[str(path)]=sha(path)
    write(out/'raw_reference_summary_unchanged.json',copied);outputs[str(out/'raw_reference_summary_unchanged.json')]=sha(out/'raw_reference_summary_unchanged.json')
    result=dict(status='T1_POSTHOC_COMMON500_NEW_PAIRED_STATISTICS_COMPLETE',source_count=500,noise_count=3,
        source_ids=ids,summary_rows=48,paired_rows=132,old_summary_rows_copied=24,old_intervals_recomputed=False,
        source_mean_first=True,frame_level_noise_pairing_claimed=False,policy_selection=False,
        new_interval_calls=new_interval_calls,identical_source_vectors_reuse_existing_intervals=True,
        post_hoc_supplement=True,multiple_comparison_adjustment=False,bootstrap_seed=2026100701,bootstrap_replicates=10000,
        metric_identity=scored['metric_identity'],metric_evaluator_identity=scored['metric_evaluator_identity'],freeze=scored['freeze'],
        new_model_calls=0,new_packet_decodes=0,input_bindings={str(Path(p).resolve()):sha(p) for p in (
            args.score_completion,args.original_statistics_completion,args.statistics_module,rows_path,old_means,old_summary,old_points)},outputs=outputs)
    write(out/'completion.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('score-completion','original-statistics-completion','statistics-module','out'):p.add_argument('--'+name,required=True)
    print(run(p.parse_args())['status'])

if __name__=='__main__':main()
