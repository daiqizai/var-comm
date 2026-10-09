"""New native-BPG versus published-reference source-paired intervals only.

Reads frozen per-source three-noise means. Imports the hash-bound pure NumPy
draws/interval functions and never invokes their original summarize pipeline.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import numpy as np

SNRS=(1,4,7,10,13,19)
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
REFERENCES=('RAW64_PARTIAL_VAR_COMPLETION','P1024','SWIN80K_N1024')
NATIVE='BPG_LDPC_N1024'
STAT_SHA='e401a77ac216368dc968ce023853ff7cdb2e0a2c94edbd3dd33407c0ba90054a'
COMMIT='252176e041758ecb2d3e81b6fde5b587e7e17bb7'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(ok,why):
    if not ok:raise ValueError(why)
def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def bound(receipt,path):
    matches=[(p,h)for p,h in receipt['outputs'].items()if p.endswith('/'+path.name)]
    require(len(matches)==1 and matches[0][1]==sha(path),'Completion-bound file differs: '+str(path))
    return dict(original_remote_path=matches[0][0],sha256=matches[0][1])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out',type=Path)
    a=parser.parse_args();root=a.root.resolve()
    out=(a.out or root/'results/paper_supplement_20261008/native_bpg_paired_v1').resolve()
    require(not out.exists(),'Use a fresh output directory; existing delivery will not be overwritten')
    started=time.perf_counter()
    native_dir=root/'.research/main_raw64_20261008_paper_supplement/bpg_delivery_actual/actual_metrics'
    published_dir=root/'.research/main_raw64_20261007/take_over_v1/current/unified500_actual_source_statistics_r6'
    native_request=root/'.research/main_raw64_20261008_paper_supplement/bpg_metrics_v1/materials_r2/request.json'
    module=root/'.research/main_raw64_20261007/take_over_v1/unified500_statistics_v1/raw64_common500_source_statistics.py'
    native_completion=native_dir/'completion.json';published_completion=published_dir/'completion.json'
    native=read(native_completion);published=read(published_completion);request=read(native_request)
    require(native['status']=='BPG_CACHED_FOUR_METRICS_AND_NEW_SOURCE_STATISTICS_COMPLETE_V1'
        and published['scientific_statistics_completed']is True,'Actual completed source statistics required')
    require(sha(native_request)==native['request_sha256']and sha(module)==STAT_SHA
        and native['original_statistics_source']==request['statistics_module']
        and request['statistics_module']['sha256']==STAT_SHA,'Original executed request/statistics module bindings')
    require(native['metrics']==list(METRICS)and native['source_count']==published['source_count']==500
        and native['frame_count']==9000 and native['bootstrap_seed']==published['bootstrap_seed']==2026100701
        and native['bootstrap_replicates']==published['bootstrap_replicates']==10000
        and published['source_mean_first']is True and not published['source_intersection_used']
        and not native['old_summary_recomputed'],'Original500-source/3-noise scope')
    ids=native['source_ids'];require(ids==published['source_ids']and len(ids)==len(set(ids))==500,'Exact ordered same500 source IDs')
    paths={'native_completion':native_completion,'published_completion':published_completion,
        'native_request':native_request,'original_statistics_module':module,'new_script':Path(__file__).resolve(),
        'native_source_means':native_dir/'source_means.csv','native_summary_unchanged':native_dir/'summary.csv',
        'published_source_means':published_dir/'source_means.json.gz','published_summary_unchanged':published_dir/'summary.json'}
    pins={key:dict(path=path.relative_to(root).as_posix(),sha256=sha(path))for key,path in paths.items()}
    for prefix,receipt in [('native',native),('published',published)]:
        for suffix in ['source_means','summary_unchanged']:
            pins[prefix+'_'+suffix].update(bound(receipt,paths[prefix+'_'+suffix]))
    with gzip.open(paths['published_source_means'],'rt',encoding='utf8')as f:published_means=json.load(f)
    native_means=csv_rows(paths['native_source_means'])
    selected_points={p+'_SNR_'+str(s)for p in (NATIVE,)+REFERENCES for s in SNRS}
    indexed={}
    for rows,allowed in [(native_means,{NATIVE}),(published_means,set(REFERENCES))]:
        for row in rows:
            point,metric=row['point_id'],row['metric']
            if point not in selected_points or metric not in METRICS:continue
            require(any(point==prefix+'_SNR_'+str(s)for prefix in allowed for s in SNRS),'Method in wrong input population')
            i=int(row['source_index']);require(0<=i<500 and row['source_id']==ids[i],'Source ID/index mismatch')
            group=indexed.setdefault((point,metric),{})
            require(i not in group,'Duplicate source mean')
            value=float(row['mean']);require(np.isfinite(value),'Nonfinite source mean')
            if metric=='convnext_top1_source_prediction':require(0<=value<=1,'Agreement is a source-mean fraction')
            group[i]=value
    require(len(indexed)==96 and all(set(g)==set(range(500))for g in indexed.values()),'Exactly96 complete500-source input groups required')
    arrays={key:np.asarray([g[i]for i in range(500)],dtype=np.float64)for key,g in indexed.items()}
    summary_index={}
    for rows in [csv_rows(paths['native_summary_unchanged']),read(paths['published_summary_unchanged'])]:
        for row in rows:
            if row['point_id']not in selected_points or row['metric']not in METRICS:continue
            key=row['point_id'],row['metric'];require(key not in summary_index,'Duplicate existing summary')
            require(int(row['source_count'])==500 and int(row['noise_count'])==3,'Existing summaries must retain500x3 denominator')
            summary_index[key]=row
    require(set(summary_index)==set(arrays),'Existing summary/source table scopes differ')
    spec=importlib.util.spec_from_file_location('_frozen_source_statistics',module)
    stat=importlib.util.module_from_spec(spec);spec.loader.exec_module(stat)
    require(stat.SOURCE_COUNT==500 and stat.REPLICATES==10000 and stat.BOOTSTRAP_SEED==2026100701,'Frozen NumPy constants')
    samples=stat.draws()  # Exactly one10000x500 index array, shared by all72 NEW contrasts.
    results=[]
    for reference in REFERENCES:
        for snr in SNRS:
            method=NATIVE+'_SNR_'+str(snr);ref=reference+'_SNR_'+str(snr)
            for metric in METRICS:
                delta=arrays[method,metric]-arrays[ref,metric]
                interval=stat.interval(delta,samples)
                require(interval['ci_low']<=interval['mean']<=interval['ci_high'],'New paired interval order')
                results.append(dict(method=method,reference=ref,snr_db=snr,metric=metric,
                    delta_definition='method minus reference',method_noise_seeds='2001|2002|2003',
                    reference_noise_seeds='6201|6202|6203'if reference.startswith('RAW64')else'2001|2002|2003',
                    frame_level_noise_pairing_claimed=False,comparison_scope='new post-hoc native BPG versus frozen published reference',
                    **interval))
    require(len(results)==72,'Exactly72 new comparisons')
    require(all(sha(path)==pins[key]['sha256']for key,path in paths.items()),'Read-only inputs changed during computation')
    out.mkdir(parents=True)
    with (out/'paired.csv').open('w',encoding='utf8',newline='')as f:
        writer=csv.DictWriter(f,list(results[0]));writer.writeheader();writer.writerows(results)
    pin_document=dict(published_commit=COMMIT,inputs=pins,source_ids=ids,source_id_order_identical=True,
        existing_three_noise_means_reused=True,selected_input_groups=96,selected_input_values=48000,
        selected_input_values_native=12000,selected_input_values_published=36000)
    write_json(out/'pins.json',pin_document)
    readme='''# Native BPG source-paired comparisons

This supplementary table adds72 new comparisons: native-resolution BPG+LDPC minus each of the published proposed partial-scale+VAR, latent continuous JSCC and adapted SwinJSCC-80k methods, at N1024 and SNR1,4,7,10,13,19, for PSNR, LPIPS-Alex, DINOv2-ViT-L/14 cosine and ConvNeXt source-prediction agreement.

The500 source IDs and their exact order are identical in the two actual completion receipts. Input source means already average each arm's three frozen noise realizations. No missing source intersection, imputation, new metric evaluation or failure filtering is used. All96 selected source-mean groups/48000 values are present. Input tables are verified against their original completion hashes, and all read-only input hashes remain unchanged after computation.

For each new contrast, subtract the two existing500-element source-mean vectors, then use the unchanged pure-NumPy statistics module's interval function. Its draws function is called exactly once to generate one10000x500 source-resampling index array with seed2026100701, shared by all72 new comparisons. These are pointwise95% percentile intervals, with no simultaneous/multiplicity claim. No original mean/CI or old bootstrap result is recomputed or overwritten; single-method CIs are never subtracted.

Direction is always native BPG minus reference. Positive favors BPG for PSNR,DINOv2-L and prediction agreement; negative favors BPG for LPIPS. Agreement values and intervals remain fractions in the CSV; multiply by100 for percentage-point differences, never relative improvement percentages. Agreement is not classification accuracy.

This is post-hoc supplementary analysis, with no policy selection on holdout. Pairing is by source, not a claim of identical transmitted waveforms or shared noisy observations. Native BPG uses2001/2002/2003; the proposed digital system uses6201/6202/6203; latent continuous and Swin use2001/2002/2003 in their respective method-specific channels. Swin19dB remains outside its training and calibration range. No training, model inference, channel simulation, PHY decoding, old-result mutation or new weight download occurs.

Published original result commit:252176e041758ecb2d3e81b6fde5b587e7e17bb7. See pins.json for exact input/code hashes and ordered source IDs; completion.json binds the new outputs.

Reproduce into a fresh directory from the repository root:

```text
python experiments/paper_supplement_20261008/native_bpg_paired.py --root . --out results/paper_supplement_20261008/native_bpg_paired_reproduction
```
'''
    (out/'README.md').write_text(readme,encoding='utf8')
    completion=dict(status='NATIVE_BPG_NEW_SOURCE_PAIRED_STATISTICS_COMPLETE_V1',finished_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,source_count=500,noise_count=3,new_paired_rows=72,
        new_contrast_families=3,source_ids=ids,metrics=list(METRICS),snr_db=list(SNRS),N=1024,
        bootstrap_seed=2026100701,bootstrap_replicates=10000,draws_calls=1,draws_shape=list(samples.shape),
        shared_draws_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),source_mean_first=True,
        frame_level_noise_pairing_claimed=False,old_summary_recomputed=False,old_bootstrap_recomputed=False,
        inputs_unchanged=True,new_metric_calls=0,new_packet_decodes=0,new_channel_simulations=0,training_updates=0,
        policy_selection=False,new_paired_bootstrap_intervals=72,numpy_version=np.__version__,python_version=platform.python_version(),
        outputs={p.relative_to(root).as_posix():sha(p)for p in [out/'paired.csv',out/'pins.json',out/'README.md']})
    write_json(out/'completion.json',completion)
    print(json.dumps({k:completion[k]for k in ['status','elapsed_seconds','new_paired_rows','draws_calls','draws_shape','inputs_unchanged']}))

if __name__=='__main__':main()
