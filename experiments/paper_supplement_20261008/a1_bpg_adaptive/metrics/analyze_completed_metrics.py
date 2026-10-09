"""Read-only checks and manuscript tables from actually completed CSV outputs.

No bootstrap draws, model imports, codec, or PHY execution. Missing local source
feature files are reported as not downloaded, never treated as locally verified.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SNRS=[1,4,7,10,13,19]
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
ADAPT='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'
REFS=['RAW64_PARTIAL_VAR_COMPLETION','P1024','SWIN80K_N1024','BPG_LDPC_N1024']
LABELS=dict(zip(REFS,['Proposed','Continuous latent JSCC','SwinJSCC-80k (adapted)','BPG + LDPC (native256)']))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csv_read(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def write(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve()
    base=w/'results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1'
    out=w/'results/paper_supplement_20261008/a1_bpg_adaptive/metrics_analysis_v1';out.mkdir(parents=True,exist_ok=True)
    done=read(base/'completion.json');wait=read(base/'owner_wait.json')
    require(done['status']=='ADAPTIVE_BPG_FOUR_METRICS_NEW_PAIRED_STATISTICS_COMPLETE_V1'
        and wait['actual_child_waited'] and wait['exit_code']==0,'Actual metric owner wait-zero required')
    prefix='/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1/'
    local_verified=[];not_local=[]
    for remote,expected in done['outputs'].items():
        require(remote.startswith(prefix),'Unexpected completion output location')
        relative=remote[len(prefix):];path=base/relative
        if path.exists():
            require(sha(path)==expected,'Downloaded scientific output changed: '+relative);local_verified.append(relative)
        else:
            require(relative.startswith('source_feature_cache/'),'Required downloaded output absent: '+relative);not_local.append(relative)
    require(len(local_verified)==507 and len(not_local)==1000,'Unexpected local/remote output inventory; examine before reporting')
    summary=csv_read(base/'summary.csv');paired=csv_read(base/'paired.csv');means=csv_read(base/'source_means.csv')
    comparison=csv_read(base/'comparison_summary.csv');rows=read(base/'rows.json');failure=csv_read(base/'failure_breakdown.csv')
    ids=done['source_ids'];require(len(ids)==len(set(ids))==500,'Wrong frozen population')
    expected_summary={(ADAPT+'_SNR_'+str(s),m) for s in SNRS for m in METRICS}
    require(len(summary)==24 and {(r['point_id'],r['metric'])for r in summary}==expected_summary,'Incomplete24 summary grid')
    expected_pairs={(ADAPT+'_SNR_'+str(s),ref+'_SNR_'+str(s),m)for s in SNRS for ref in REFS for m in METRICS}
    require(len(paired)==96 and {(r['method'],r['reference'],r['metric'])for r in paired}==expected_pairs,'Incomplete96 paired grid')
    for row in summary+paired:
        require(int(row['source_count'])==500 and int(row['noise_count'])==3 and int(row['frame_count'])==1500
            and int(row['bootstrap_seed'])==2026100701 and int(row['bootstrap_replicates'])==10000,'Statistics population/draw metadata mismatch')
        lo,mean,hi=map(float,[row['ci_low'],row['mean'],row['ci_high']])
        require(all(math.isfinite(x)for x in [lo,mean,hi]) and lo<=mean<=hi,'Invalid existing interval')
    require(all(r['delta_definition']=='method minus reference' and r['frame_level_noise_pairing_claimed']=='False' for r in paired),'Changed difference direction or pairing claim')
    rowkeys={(r['source_index'],r['snr_db'],r['noise_seed'])for r in rows}
    require(len(rows)==len(rowkeys)==9000 and rowkeys=={(i,s,n)for i in range(500)for s in SNRS for n in [2001,2002,2003]},'Actual500x6x3 row grid incomplete')
    noise_values=defaultdict(list)
    for row in rows:
        require(row['source_id']==ids[row['source_index']] and row['point_id']==ADAPT+'_SNR_'+str(row['snr_db'])
            and row['metric_evaluator_identity']==done['metric_evaluator_identity'],'Frame identity differs')
        for metric in METRICS:noise_values[row['point_id'],metric,row['source_index']].append(float(row[metric]))
    require(len(means)==len(noise_values)==12000,'Incomplete12000 source means')
    values={}
    for row in means:
        key=row['point_id'],row['metric'],int(row['source_index'])
        require(key not in values and row['source_id']==ids[key[2]] and len(noise_values[key])==3,'Duplicate/unpaired source mean')
        value=float(row['mean']);require(value==float(np.mean(noise_values[key],dtype=np.float64)),'Stored source mean differs from its exact three recorded noises')
        values[key]=value
    for row in summary:
        mean=float(np.asarray([values[row['point_id'],row['metric'],i]for i in range(500)],dtype=np.float64).mean())
        require(mean==float(row['mean']),'Stored summary differs from its500 source means')
    oldbase=w/'.research/main_raw64_20261007/take_over_v1/current/unified500_actual_source_statistics_r6'
    nativebase=w/'.research/main_raw64_20261008_paper_supplement/bpg_delivery_actual/actual_metrics'
    olddone=read(oldbase/'completion.json');nativedone=read(nativebase/'completion.json')
    require(olddone['source_ids']==nativedone['source_ids']==ids and nativedone['metric_evaluator_identity']==done['metric_evaluator_identity'],'Original reference identity differs')
    def check_old(control,path):
        matched=[h for remote,h in control['outputs'].items()if remote.endswith('/'+path.name)]
        require(matched==[sha(path)],'Original reference file not bound: '+path.name)
    for control,root,names in [(olddone,oldbase,['source_means.json.gz','summary.json']),(nativedone,nativebase,['source_means.csv','summary.csv'])]:
        for name in names:check_old(control,root/name)
    with gzip.open(oldbase/'source_means.json.gz','rt',encoding='utf-8')as f:previous=json.load(f)
    previous+=csv_read(nativebase/'source_means.csv')
    oldvalues={}
    reference_points={ref+'_SNR_'+str(s)for ref in REFS for s in SNRS}
    for row in previous:
        if row['point_id']not in reference_points or row['metric']not in METRICS:continue
        key=row['point_id'],row['metric'],int(row['source_index'])
        require(key not in oldvalues and row['source_id']==ids[key[2]],'Original source mean duplicated/misaligned');oldvalues[key]=float(row['mean'])
    require(len(oldvalues)==48000,'Missing four-reference source means')
    for row in paired:
        diff=np.asarray([values[row['method'],row['metric'],i]-oldvalues[row['reference'],row['metric'],i]for i in range(500)],dtype=np.float64)
        require(float(diff.mean())==float(row['mean']),'Stored paired direction/mean differs from exact source-paired means')
    oldsummary=read(oldbase/'summary.json')+csv_read(nativebase/'summary.csv')
    oldlookup={(r['point_id'],r['metric']):r for r in oldsummary if r['point_id']in reference_points and r['metric']in METRICS}
    copied=[r for r in comparison if r['provenance'].endswith('_UNCHANGED')]
    require(len(comparison)==120 and len(copied)==96 and len(oldlookup)==96,'Unexpected comparison summary population')
    for row in copied:
        original=oldlookup[row['point_id'],row['metric']]
        require(all(float(row[k])==float(original[k])for k in ['mean','ci_low','ci_high','source_count','noise_count']),'An old summary/interval changed')
    published=w/'results/main_raw64_20261007/final_common500_r6/summary.csv';published_checked=0
    if published.exists():
        published_lookup={(r['point_id'],r['metric']):r for r in csv_read(published)}
        for row in copied:
            if row['point_id'].startswith('BPG_LDPC_'):continue
            original=published_lookup[row['point_id'],row['metric']]
            require(all(float(row[k])==float(original[k])for k in ['mean','ci_low','ci_high','source_count','noise_count']),'Published original figure number changed');published_checked+=1
    categories=['source_unfit','header_reject','body_CRC_reject','body_parser_reject','BPG_decode_reject','decoded']
    require(len(failure)==6 and {int(r['snr_db'])for r in failure}==set(SNRS),'Incomplete failures')
    for row in failure:
        require(sum(int(row[k])for k in categories)==1500 and int(row['source_unfit'])==0,'Failures omitted or unexpected source-unfit count')
    interpretations=[];counts=defaultdict(Counter)
    for row in paired:
        lo,mean,hi=map(Decimal,[row['ci_low'],row['mean'],row['ci_high']]);lower=row['metric']=='lpips_alex'
        favored='adaptive_supported' if (hi<0 if lower else lo>0) else 'reference_supported' if (lo>0 if lower else hi<0) else 'interval_includes_zero'
        ref=row['reference'].rsplit('_SNR_',1)[0];counts[ref][favored]+=1
        scale=Decimal(100)if row['metric']=='convnext_top1_source_prediction'else Decimal(1)
        interpretations.append(dict(row,reference_label=LABELS[ref],desirable_delta='negative'if lower else 'positive',
            interval_interpretation=favored,display_unit='percentage points'if scale==100 else 'dB'if row['metric']=='psnr_db'else 'unitless',
            display_mean=str(mean*scale),display_ci_low=str(lo*scale),display_ci_high=str(hi*scale)))
    csv_write(out/'paired_interpretation.csv',interpretations)
    csv_write(out/'adaptive_summary_original_precision.csv',summary)
    resolutions=[]
    for snr in SNRS:
        selected=[r for r in rows if r['snr_db']==snr and r['noise_seed']==2001]
        for row in selected:
            same=[x for x in rows if x['source_index']==row['source_index'] and x['snr_db']==snr]
            require(len({(x['selected_resolution'],x['selected_qp'],x['profile_id'])for x in same})==1,'Source setting depends on receive noise')
        for side,count in sorted(Counter(r['selected_resolution']for r in selected).items()):
            resolutions.append(dict(snr_db=snr,resolution=side,source_count=count,denominator=500))
    csv_write(out/'selected_resolution_counts.csv',resolutions)
    table=['# Existing source-paired results (adaptive BPG minus reference)','','Every entry below is mean [existing95% CI]. ConvNeXt differences are percentage points. LPIPS retains its original sign: negative is better. No interval was recomputed.','']
    for ref in REFS:
        table += ['## '+LABELS[ref],'','| SNR (dB) | ΔPSNR (dB) | ΔLPIPS | ΔDINOv2-L | ΔConvNeXt agreement (pp) |','|---:|---:|---:|---:|---:|']
        for snr in SNRS:
            cells=[]
            for metric in METRICS:
                row=next(x for x in interpretations if x['reference']==ref+'_SNR_'+str(snr)and x['metric']==metric)
                digits=2 if metric in ['psnr_db','convnext_top1_source_prediction'] else 4
                cells.append(f'{float(row["display_mean"]):+.{digits}f} [{float(row["display_ci_low"]):+.{digits}f}, {float(row["display_ci_high"]):+.{digits}f}]')
            table.append('| '+str(snr)+' | '+' | '.join(cells)+' |')
        table.append('')
    table += ['SwinJSCC-80k is the adapted checkpoint;19 dB is outside its training and calibration range. These are pointwise intervals, not multiplicity-adjusted global rankings. The full-precision CSV accompanies this rounded reading table.']
    (out/'paired_results.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    checks=dict(status='LOCAL_AVAILABLE_OUTPUT_AUDIT_PASS',summary_rows=24,paired_rows=96,source_mean_cells=12000,
        actual_rows=9000,source_count=500,noise_count_per_source_SNR=3,mean_and_pair_direction_exactly_checked=True,
        CI_order_checked=True,old_summary_CI_cells_unchanged=96,published_common500_cells_checked=published_checked,
        local_completion_outputs_verified=len(local_verified),not_downloaded_source_feature_outputs=len(not_local),
        not_downloaded_scope='Source feature cache JSON/NPZ: not locally opened or hash-verified; no claim that all1507 outputs are present locally.',
        outputs_declared_by_completed_worker=len(done['outputs']),remote_full_hash_check_repeated_here=False,
        owner_wait=wait,new_bootstrap_calls=0,new_metric_calls=0,new_PHY_calls=0,scientific_inputs_modified=False,
        interval_interpretation_counts={k:dict(v)for k,v in counts.items()},new_quality_calls_recorded=done['new_unique_quality_calls_total'],
        inherited_native_same_image_reuses_recorded=done['inherited_native256_same_image_reuses'],reference_preparations_recorded=done['new_reference_preparations_this_run'],
        input_control_hashes={str(base/'completion.json'):sha(base/'completion.json'),str(base/'owner_wait.json'):sha(base/'owner_wait.json')},
        scientific_CSV_hashes={str(base/name):sha(base/name)for name in ['summary.csv','paired.csv','source_means.csv','comparison_summary.csv','failure_breakdown.csv']})
    write(out/'validation.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
