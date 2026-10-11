"""Once-only source-paired supplement using the published bootstrap functions.

Preparation does not draw samples. Run requires the normally closed scoring chain.
It does not import a model, recompute an old interval, or select a checkpoint.
"""
import argparse
import collections
import csv
import fcntl
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT/'results/baseline_strength_20261011'
CHAIN = RESULT/'automatic_followthrough_v1'
TMP = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
OLD = TMP/'postprocess_original500_v1'
OUT = RESULT/'paired_supplement_v1'
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine', 'convnext_top1_source_prediction')
PINS = {'source_means.json.gz':'414a507bf7d58dd852084321ab963b7c9b06a856e671f8dca43714c0244b17ea',
        'source_ids.json':'3fa217d5e85cc8967f0274d2ea7593c948b140190e21ff2c7c729405e56cf84c',
        'summary.json':'8881d842d0d6a2431266cf2f5bc2a2acffad35bccc3995688dffb71caa9bf24d',
        'points.json':'971e26293956fe79ddcffd6917ac528ad45127ff299361bda4e5296da44fb860',
        'raw64_common500_source_statistics.py':'e401a77ac216368dc968ce023853ff7cdb2e0a2c94edbd3dd33407c0ba90054a'}
OLD_METHODS = {'P40000':'P1024_SNR_{}', 'Swin80000':'SWIN80K_N1024_SNR_{}',
               'RAW_PARTIAL':'RAW64_PARTIAL_VAR_COMPLETION_SNR_{}'}
PAIRS = {'common500':[('P100000','P40000'), ('Swin200000','Swin80000'),
                       ('RAW_PARTIAL','P100000'), ('RAW_PARTIAL','Swin200000')],
         'new100':[('EC_VAR_PARTIAL','P100000'), ('EC_VAR_PARTIAL','Swin200000')]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    assert rows
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)


def original_statistics(count):
    p=OLD/'raw64_common500_source_statistics.py'
    assert sha(p)==PINS[p.name]
    spec=importlib.util.spec_from_file_location('_private_original_statistics_'+str(count),p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert (module.SOURCE_COUNT,module.REPLICATES,module.BOOTSTRAP_SEED)==(500,10000,2026100701)
    bodies={name:hashlib.sha256(json.dumps(inspect.getsource(getattr(module,name)),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest() for name in ('draws','interval')}
    assert bodies=={'draws':'08458f6546ee82a6d3550bc2906dda676b847872f20c779f49fc36de19378f2b',
                    'interval':'e3a17cf502a97b93788a7815dfc9bd1f7a568c4f3f87be2cc8d7aea3d138ce5a'}
    module.SOURCE_COUNT=count  # Same population-only adaptation as published T6/new100.
    return module,bodies


def old_vectors(population, ids, snrs):
    count=len(ids);vectors={};summaries=[]
    if population=='common500':
        assert read(OLD/'source_ids.json')==ids
        with gzip.open(OLD/'source_means.json.gz','rt') as f: rows=json.load(f)
        points=read(OLD/'points.json');summary=read(OLD/'summary.json')
        index={(r['point_id'],r['metric']):r for r in summary}
        mapping={pattern.format(s):(m,s) for m,pattern in OLD_METHODS.items() for s in snrs}
        selected=[r for r in rows if r['point_id'] in mapping and r['metric'] in METRICS]
        for method,pattern in OLD_METHODS.items():
            for snr in snrs:
                p=pattern.format(snr);pt=points[p]
                assert pt['noise_seeds']==([6201,6202,6203] if method=='RAW_PARTIAL' else [2001,2002,2003])
                for metric in METRICS:
                    row=dict(index[p,metric]);assert row['source_count']==count and row['noise_count']==3
                    assert row['metric_identity']==pt['metric_identity'][metric]
                    summaries.append(dict(method=method,**row))
        normalized=[dict(method=mapping[r['point_id']][0],snr_db=mapping[r['point_id']][1],**r) for r in selected]
    else:
        base=ROOT/'results/entropy_partial_new100_20261011'
        normalized=csv_rows(base/'source_means.csv')
        summaries=csv_rows(base/'summary.csv')
        assert len(normalized)==4*3*4*100
    for r in normalized:
        i=int(r['source_index']);assert 0<=i<count and r['source_id']==ids[i]
        key=(r['method'],int(r['snr_db']),r['metric'])
        assert key[1] in snrs and key[2] in METRICS
        if key not in vectors:vectors[key]=np.full(count,np.nan)
        assert np.isnan(vectors[key][i]),'Duplicate old source mean'
        vectors[key][i]=float(r['mean'])
    assert all(np.isfinite(a).all() for a in vectors.values())
    for r in summaries:
        key=(r['method'],int(r['snr_db']),r['metric'])
        assert abs(float(r['mean'])-float(vectors[key].mean()))<1e-12
        assert int(r['source_count'])==count and int(r['noise_count'])==3
        assert float(r['ci_low'])<=float(r['mean'])<=float(r['ci_high'])
    return vectors,summaries


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'registration.json').exists(),'Already registered'
    chain=read(CHAIN/'registration.json')
    bindings={str(OLD/n):h for n,h in PINS.items()}
    for p,h in bindings.items():assert sha(p)==h,p
    for n in ['source_means.csv','summary.csv','points.json','fairness_notes.json']:
        p=ROOT/'results/entropy_partial_new100_20261011'/n;bindings[str(p)]=sha(p)
    p=CHAIN/'registration.json';bindings[str(p)]=sha(p)
    for population in ('common500','new100'):
        ids=[r['source_id'] for r in chain['populations'][population]['records']]
        assert len(ids)==len(set(ids))==({'common500':500,'new100':100}[population])
        old_vectors(population,ids,[1,4,7,10,13,19] if population=='common500' else [4,10,19])
        original_statistics(len(ids))
    write(OUT/'registration.json',dict(schema='BASELINE_SOURCE_PAIRED_SUPPLEMENT_V1', created_unix=time.time(),
        inputs=bindings, code_sha256=sha(__file__), pairs=PAIRS, source_counts={'common500':500,'new100':100},
        new_single_method_intervals={'common500':48,'new100':24}, new_paired_intervals={'common500':96,'new100':24},
        old_intervals_recomputed=0,bootstrap_replicates=10000,bootstrap_seed=2026100701,
        extra_training=0,extra_inference=0,automatic_retries=False,
        cohort_note='Subsequent same-source baseline supplement; no newly blind test claim.',
        numeric_note='Frozen metric definitions/weights are retained. New scores use the documented H800 environment; cross-host bitwise equivalence is not claimed.'))
    print('PREPARED_NO_BOOTSTRAP',flush=True)


def metric_gate(stage):
    metadata=read(CHAIN/stage/'metric_identity.json')['metadata']
    assert metadata['metrics']==list(METRICS)
    expected={'alexnet':'7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
              'lpips':'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0',
              'convnext':'983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d',
              'dinov2_vitl14':'d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'}
    assert {k:v['sha256'] for k,v in metadata['assets']['weights'].items()}==expected
    assert metadata['inference_precision']=='float32' and not metadata['AMP']
    assert metadata['adapter_source']['sha256']=='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
    assert not metadata['convnext_used_for_selection'] and not metadata['received_truth_supplied_to_reconstruction']
    assert set(metadata['original_math_proof'])=={'LPIPS','ConvNeXt','PSNR'}
    assert not metadata['numeric_flags']['matmul_tf32'] and not metadata['numeric_flags']['cudnn_tf32']
    return dict(identity_sha256=sha(CHAIN/stage/'metric_identity.json'),weights=expected,
                original_math_proof=metadata['original_math_proof'],numeric_flags=metadata['numeric_flags'])


def validate_grid(rows, method, ids, snrs, seeds, reference_hashes):
    expected={(i,s,n) for i in range(len(ids)) for s in snrs for n in seeds};keyed={}
    for r in rows:
        i,s,n=(int(r[k]) for k in ('source_index','snr_db','noise_seed'));key=(i,s,n)
        assert key in expected and key not in keyed,'Invalid or duplicate frame'
        assert r['source_id']==ids[i] and r['method']==method and int(r['N'])==1024
        assert int(r['checkpoint_step'])==({'P100000':100000,'Swin200000':200000}[method])
        assert r['reference_sha256']==reference_hashes[i],'Source pixels differ'
        assert all(np.isfinite(float(r[m])) for m in METRICS)
        assert float(r[METRICS[-1]]) in (0,1)
        assert r['header_accepted'] in ('True','False')
        assert r['reused']=='False','New model evaluation must be fresh'
        keyed[key]=r
    assert set(keyed)==expected,'Incomplete frozen frame grid'
    return keyed


def run():
    lock=(OUT/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(),'Once-only call; recovery needs explicit audit'
    reg=read(OUT/'registration.json');assert sha(__file__)==reg['code_sha256']
    for p,h in reg['inputs'].items():assert sha(p)==h,p
    assert not list(CHAIN.glob('*failure.json')),'Scoring failure exists'
    done=read(CHAIN/'completion.json');assert done['status']=='ALL_DIAGNOSTIC_AND_FROZEN_FRAME_SCORING_COMPLETE'
    for wait in done['actual_waits']:
        assert wait['exit_code']==0 and wait['actual_child_waited'] and wait['stop_reason'] is None
    chain=read(CHAIN/'registration.json');admitted={}
    for p,h in {**chain['code_bindings'],**chain['training_bindings']}.items():assert sha(p)==h,p
    for chosen in chain['selections'].values():
        assert sha(chosen['checkpoint']['path'])==chosen['checkpoint']['sha256']
    for stage in ('common500','new100'):
        folder=CHAIN/stage;completion=read(folder/'completion.json')
        assert completion['frames']==chain['stage_frames'][stage] and completion['status']=='COMPLETE_FROZEN_FRAME_SCORING'
        for p,h in completion['outputs'].items():assert sha(p)==h,p
        snrs=[1,4,7,10,13,19] if stage=='common500' else [4,10,19]
        seeds=chain[stage+'_noise_seeds'];records=chain['populations'][stage]['records'];ids=[r['source_id'] for r in records]
        reference_hashes=[]
        for record in records:
            p=Path(record['pixels']['path']);assert sha(p)==record['pixels']['sha256']
            with np.load(p,allow_pickle=False) as z:pixels=z['pixels']
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==record['preprocessing_id']
            a=np.ascontiguousarray(pixels.astype(np.float32)/np.float32(255))
            reference_hashes.append(hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest())
        vectors,old_summary=old_vectors(stage,ids,snrs);sources=[];failures=[]
        for method in ('P100000','Swin200000'):
            rows=csv_rows(folder/(method+'.csv'));keyed=validate_grid(rows,method,ids,snrs,seeds,reference_hashes)
            for s in snrs:
                group=[r for r in rows if int(r['snr_db'])==s]
                failures.append(dict(method=method,snr_db=s,frame_count=len(group),header_rejected=sum(r['header_accepted']=='False' for r in group),
                                     fallback_count=sum(r['fallback']!='none' for r in group),duplicate_count=0,missing_count=0))
                for metric in METRICS:
                    a=np.array([np.mean([float(keyed[i,s,n][metric]) for n in seeds],dtype=np.float64) for i in range(len(ids))])
                    vectors[method,s,metric]=a
                    sources.extend(dict(method=method,snr_db=s,metric=metric,source_index=i,source_id=sid,mean=float(a[i]),source_count=len(ids),noise_count=3) for i,sid in enumerate(ids))
        admitted[stage]=(ids,snrs,vectors,old_summary,sources,failures,metric_gate(stage))
    # All identities, full grids, failures, sources and old anchors pass before drawing.
    write(OUT/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),registration_sha256=sha(OUT/'registration.json')))
    for stage,(ids,snrs,vectors,old_summary,sources,failures,identity) in admitted.items():
        dest=OUT/stage;dest.mkdir(exist_ok=False)
        stat,bodies=original_statistics(len(ids));draws=stat.draws();assert draws.shape==(10000,len(ids))
        intervals=0
        def estimate(a):
            nonlocal intervals
            if np.count_nonzero(a)==0:
                value=dict(mean=0.,ci_low=0.,ci_high=0.,source_count=len(ids),noise_count=3,
                           bootstrap_replicates=10000,bootstrap_seed=2026100701,bootstrap_unit='source after original3-noise mean')
            else:value=stat.interval(a,draws);intervals+=1
            value['frame_count']=len(ids)*3
            return value
        summaries=[];pairs=[];differences=[]
        for method in ('P100000','Swin200000'):
            for s in snrs:
                for metric in METRICS:
                    summaries.append(dict(method=method,N=1024,snr_db=s,metric=metric,**estimate(vectors[method,s,metric])))
        for method,reference in PAIRS[stage]:
            for s in snrs:
                for metric in METRICS:
                    delta=vectors[method,s,metric]-vectors[reference,s,metric]
                    pairs.append(dict(method=method,reference=reference,N=1024,snr_db=s,metric=metric,delta_definition='method - reference',**estimate(delta)))
                    differences.extend(dict(method=method,reference=reference,snr_db=s,metric=metric,source_index=i,source_id=sid,delta=float(delta[i])) for i,sid in enumerate(ids))
        for row in summaries+pairs:
            row['display_scale']=100 if row['metric']==METRICS[-1] else 1
            row['display_unit']=('percentage_points' if 'reference' in row else 'percent') if row['metric']==METRICS[-1] else ('dB' if row['metric']=='psnr_db' else 'raw_metric_units')
            row['improvement_direction']='negative' if row['metric']=='lpips_alex' else 'positive'
            assert row['ci_low']<=row['mean']<=row['ci_high']
        assert len(summaries)==reg['new_single_method_intervals'][stage] and len(pairs)==reg['new_paired_intervals'][stage]
        for name,rows in [('summary',summaries),('paired',pairs),('source_means',sources),('source_paired_differences',differences),('failures',failures)]:csv_write(dest/(name+'.csv'),rows)
        write(dest/'original_summary_reused.json',old_summary)
        write(dest/'admission.json',dict(source_count=len(ids),noise_count=3,noise_seeds=chain[stage+'_noise_seeds'],metric_identity=identity,
                                       old_intervals_recomputed=0,original_function_sha256=bodies,draw_matrix_calls=1,new_interval_calls=intervals,
                                       source_pairing_only=True,all_failure_rows_retained=True,selection_feedback=False))
        write(dest/'completion.json',dict(status='COMPLETE_SOURCE_PAIRED_SUPPLEMENT',new_summary_rows=len(summaries),new_pair_rows=len(pairs),
                                         outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    write(OUT/'completion.json',dict(status='COMPLETE_BOTH_SOURCE_PAIRED_SUPPLEMENTS',source_counts=reg['source_counts'],old_intervals_recomputed=0,
                                    inference_calls=0,bootstrap_replicates=10000,bootstrap_seed=2026100701))
    print('COMPLETE_BOTH_SOURCE_PAIRED_SUPPLEMENTS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);args=parser.parse_args()
    try:prepare() if args.command=='prepare' else run()
    except BaseException:
        if args.command=='run' and (OUT/'launch.json').exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
        raise
