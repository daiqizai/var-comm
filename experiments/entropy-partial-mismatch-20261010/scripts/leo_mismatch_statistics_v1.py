"""One registered source-paired bootstrap after actual closed leo raw metrics.

Preparation authenticates metadata only. Execution draws once using the published
bootstrap and evaluates exactly48 nonmatched contrasts;24 matched rows are zero.
No model, image decoding, PHY, policy selection or historical CI is called/changed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np

SCHEMA='LEO_RAW100_MISMATCH_SOURCE_PAIRED_STATISTICS_V1'
PASS='PASS_LEO_RAW100_MISMATCH_SOURCE_PAIRED_STATISTICS'
METRIC_PASS='PASS_LEO_RAW100_MISMATCH_FOUR_METRICS_GPU5'
METRIC_SCHEMA='LEO_RAW100_ONE_BIN_MISMATCH_METRIC_OWNER_V3_GPU5'
METRIC_OWNER_SHA='db6e009ada8266424f9afa81609f8f65f1dc7503cf5affa7f12aaecfc0cfb521'
METRIC_ADAPTER_SHA='38bfd1130a2ecc881b945712c3a0629d1fd9abd943b9b4a31d28310565896396'
T6_STATISTICS_SHA='31becdf802973ff6e58d794c62e99be1b0da47bb9c133dcd1a1945b85fa7777c'
T6_SCORE_SHA='5b24c1fcab6242521c015296112f283c96d381f8e9f2a67c040aa034094db7ab'
STATISTICS_SHA='e401a77ac216368dc968ce023853ff7cdb2e0a2c94edbd3dd33407c0ba90054a'
MANIFEST_SHA='b27128fb8eedee7f2cc74bed25c63e39e8add8c938c76d449f85c4b0d5a51e09'
POLICY_SHA='7871ac7f9c619bd21c63ba31157738cd56c0dda9344c993e99b935f60ce70c0c'
PROTOCOL_SHA='c458844e95b0e448352067356edb8bd2328d4f881ef1bf0bce87ef7dd4fa62be'
PLAN_SHA='c15e7975334bd40eb46fe35cdaa65a7e4532b63514540eb98cb05bf36a9ba0bf'
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
SNRS=(4,7,10);LOOKUPS={4:(1,4,7),7:(4,7,10),10:(7,10,13)}
FAMILIES=('WHOLE','PARTIAL');SEEDS=(6201,6202,6203)
REPLICATES=10000;SEED=2026100701
CAPS=dict(draw_matrix=1,nonmatched_intervals=48,matched_exact_zero_rows=24)
RT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
METRIC_ROOT=RT/'qualification/leo_mismatch_metrics_v3_attempt1'
OUT=RT/'qualification/leo_mismatch_statistics_v1_attempt1'

def require(ok,message):
    if not ok:raise RuntimeError(message)

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def readpin(p):
    path=Path(p['path']);require(path.is_file() and not path.is_symlink(),'Regular metadata required')
    data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==p['sha256'],'Bound metadata changed: '+str(path))
    return json.loads(data)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:f.write(canonical(x)+'\n');f.flush();os.fsync(f.fileno())
def csv_write(p,rows):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def definitions():
    return [dict(actual_snr_db=s,config_snr_db=k,reference_config_snr_db=s,family=f,metric=m,
        contrast_kind='matched_zero' if k==s else 'underestimate' if k<s else 'overestimate',
        delta_definition='mismatch_minus_same_true_SNR_matched')
        for s in SNRS for k in LOOKUPS[s] for f in FAMILIES for m in METRICS]

def source_bindings():
    e=Path(__file__).resolve().parents[1];t=e.parent/'wcl-evidence-closure-20261009/scripts'
    expected={e/'PROTOCOL.md':PROTOCOL_SHA,e/'scripts/mismatch_plan.py':PLAN_SHA,
        t/'t6_statistics.py':T6_STATISTICS_SHA,t/'t6_score.py':T6_SCORE_SHA}
    for path,h in expected.items():require(sha(path)==h,'Frozen statistical dependency changed: '+str(path))
    return {str(p):h for p,h in expected.items()}|{str(Path(__file__).resolve()):sha(__file__)}

def original_statistics(path):
    """Run only the exact T6 private loader; it never draws or estimates an interval."""
    source_bindings();require(sha(path)==STATISTICS_SHA,'Original common500 bootstrap source changed')
    folder=Path(__file__).resolve().parents[2]/'wcl-evidence-closure-20261009/scripts'
    sys.path.insert(0,str(folder))
    try:
        spec=importlib.util.spec_from_file_location('_raw_mismatch_published_t6',folder/'t6_statistics.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        require(module.COUNT==100,'Published T6 population changed')
        return module.original_statistics(path)
    finally:sys.path.remove(str(folder))

def validate_grid(rows,ids):
    require(len(ids)==len(set(ids))==100,'Exactly100 original source IDs required')
    keys={}
    for r in rows:
        key=tuple(r[x] for x in ('source_index','actual_snr_db','config_snr_db','family','noise_seed'))
        require(key not in keys,'Duplicate logical metric row')
        i,s,k,f,n=key
        require(type(i) is int and 0<=i<100 and r['source_id']==ids[i] and
            type(s) is int and s in SNRS and type(k) is int and k in LOOKUPS[s] and f in FAMILIES and
            type(n) is int and n in SEEDS,'Metric row outside frozen diagnostic grid')
        require(all(type(r[m]) in (int,float) and math.isfinite(r[m]) for m in METRICS) and
            r[METRICS[-1]] in (0,1),'Finite four metrics and binary source agreement required')
        require(all(type(r[x]) is bool for x in ('actual_header_ok','actual_crc_accepted','body_attempted','gray')),
            'Actual logical failure booleans required')
        require(r['gray']==(not r['actual_header_ok']) and (not r['body_attempted'] or r['actual_header_ok']),
            'Only actual header rejection is fixed gray; preserve body CRC KEEP')
        keys[key]=r
    expected={(i,s,k,f,n) for i in range(100) for s in SNRS for k in LOOKUPS[s] for f in FAMILIES for n in SEEDS}
    require(len(rows)==5400 and set(keys)==expected,'Complete5400 unique logical metric rows required')
    return keys

def metric_closure(pins):
    require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(METRIC_ROOT/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(METRIC_ROOT/'run_owner_actual_wait.json'),'Exact leo metric attempt required')
    done=readpin(pins['completion']);wait=readpin(pins['owner_actual_wait'])
    require(done['status']==METRIC_PASS and done['schema']==METRIC_SCHEMA and done['actual_wait']['success'] is True and
        done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and done['bootstrap_calls']==0 and
        done['policy_selection'] is False and wait['actual_wait'] is True and wait['returncode']==0 and wait['timeout'] is False,
        'Actually waited complete leo metrics required before statistics')
    root=METRIC_ROOT/'registered';request_pin=dict(path=str(root/'request.json'),sha256=done['request_sha256']);r=readpin(request_pin)
    worker=readpin(done['worker_completion']);result=worker['results'];counts=result['counts'];c=counts['completed']
    require(done['worker_completion']['path']==str(root/'run/worker_completion.json') and worker['status']==METRIC_PASS and
        worker['request_sha256']==done['request_sha256'] and r['schema']==METRIC_SCHEMA and r['target_index']==5 and
        r['policy_selection'] is False and r['bootstrap_calls']==0 and result['bootstrap_calls']==0 and result['policy_selection'] is False and
        result['logical_count']==5400 and counts['unresolved']==0 and c==counts['reserved'] and c['model_constructions']==3 and
        c['reference_preparations']==100 and c['image_scores']==result['actual_unique_pairs']<=4500 and
        result['actual_unique_pairs']+result['reused_pairs']==5400 and all(c[k]<=v for k,v in counts['caps'].items()),
        'Actual complete finite diagnostic metric ledger required')
    require(r['tool_bindings']['leo_mismatch_metric_owner_v3.py']==METRIC_OWNER_SHA and
        r['tool_bindings']['h800_mismatch_metrics_v1.py']==METRIC_ADAPTER_SHA,'Frozen metric implementation identity changed')
    policy=r['metric_config']['raw_frozen_policy'];require(policy['sha256']==POLICY_SHA,'Wrong raw policy');readpin(policy)
    fixture=Path(__file__).resolve().parents[1]/'tests/fixtures/common500_v1/manifest.json'
    manifest=readpin(dict(path=str(fixture),sha256=MANIFEST_SHA));ids=manifest['source_ids'][:100]
    require(len(r['records'])==100 and [x['source_index'] for x in r['records']]==list(range(100)) and
        [x['source_id'] for x in r['records']]==ids,'Original common500 first100 order required')
    identity=readpin(worker['metric_identity']);require(identity['identity']==digest(identity['metadata']) and
        identity['metadata']['policy_selection'] is False and identity['metadata']['selection_objective'] is None,
        'Actual post-hoc four-metric identity required')
    rows=readpin(result['metric_rows']);validate_grid(rows,ids);seen={}
    for row in rows:
        pp=row['metric_pair_result'];key=canonical(pp)
        if key not in seen:seen[key]=readpin(pp)
        p=seen[key]
        require(p['metric_identity']==identity['identity'] and p['metric_pair_key']==row['metric_pair_key'] and
            p['reference_archive']==r['records'][row['source_index']]['actual_archive'] and
            all(p['metrics'][m]==row[m] for m in METRICS),'Rows must match actual scored pair bytes and identity')
    require(len(seen)==result['actual_unique_pairs'],'Metric pair count changed')
    child=readpin(pin(root/'run/actual_child_wait.json'));require(child==done['actual_wait'],'Actual child wait changed')
    prior_arithmetic=readpin(result['paired_outputs'])
    require(prior_arithmetic['bootstrap_calls']==0 and prior_arithmetic['policy_selection'] is False and
        len(prior_arithmetic['source_paired_differences'])==7200 and len(prior_arithmetic['mean_differences'])==72,
        'Preserved unbootstrapped diagnostic arithmetic required')
    return rows,ids,dict(metric_request=request_pin,metric_worker=done['worker_completion'],metric_rows=result['metric_rows'],
        metric_identity=worker['metric_identity'],existing_no_bootstrap_arithmetic=result['paired_outputs'],
        source_manifest=pin(fixture),raw_policy=policy,metric_identity_sha256=identity['identity'])

def source_vectors(rows,ids):
    keys=validate_grid(rows,ids);values={};means=[];failures=[]
    for d in definitions():
        s,k,f,m=(d[x] for x in ('actual_snr_db','config_snr_db','family','metric'))
        a=np.asarray([np.mean([keys[i,s,k,f,n][m] for n in SEEDS],dtype=np.float64) for i in range(100)],dtype=np.float64)
        values[s,k,f,m]=a
        means.extend(dict(source_index=i,source_id=ids[i],actual_snr_db=s,config_snr_db=k,family=f,metric=m,
            mean=float(v),source_count=100,noise_count=3) for i,v in enumerate(a))
    for s in SNRS:
        for k in LOOKUPS[s]:
            for f in FAMILIES:
                cell=[keys[i,s,k,f,n] for i in range(100) for n in SEEDS]
                failures.append(dict(actual_snr_db=s,config_snr_db=k,family=f,logical_frames=300,
                    header_failures=sum(not r['actual_header_ok'] for r in cell),body_attempted=sum(r['body_attempted'] for r in cell),
                    body_CRC_rejections=sum(r['body_attempted'] and not r['actual_crc_accepted'] for r in cell),
                    fixed_gray=sum(r['gray'] for r in cell),body_CRC_denominator='body_attempted only'))
    for row in failures:
        matched=next(x for x in failures if x['actual_snr_db']==row['actual_snr_db']==x['config_snr_db'] and x['family']==row['family'])
        for name in ('header_failures','body_CRC_rejections','fixed_gray'):
            row[name+'_minus_matched']=row[name]-matched[name]
    return values,means,failures

def summarize(rows,ids,stat,event=lambda *x:None):
    values,means,failures=source_vectors(rows,ids);event('reserved','draw_matrix',0,{})
    samples=stat.draws();require(samples.shape==(10000,100),'Exact registered draw matrix required')
    event('completed','draw_matrix',0,dict(shape=list(samples.shape),sha256=hashlib.sha256(samples.tobytes()).hexdigest()))
    paired=[];deltas=[];calls=0
    for index,d in enumerate(definitions()):
        s,k,f,m=(d[x] for x in ('actual_snr_db','config_snr_db','family','metric'))
        a=values[s,k,f,m]-values[s,s,f,m]
        require(a.shape==(100,) and np.isfinite(a).all(),'Finite source-paired differences required')
        vector_sha=hashlib.sha256(a.tobytes()).hexdigest()
        if k==s:
            require(np.all(a==0),'Matched source difference must be exactly zero')
            result=dict(mean=0.,ci_low=0.,ci_high=0.,source_count=100,noise_count=3,frame_count=300,
                bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,bootstrap_unit='source after original3-noise mean')
            provenance='EXACT_MATCHED_ZERO_NO_INTERVAL_CALL'
        else:
            event('reserved','nonmatched_intervals',index,dict(comparison=d,source_delta_sha256=vector_sha))
            result=stat.interval(a,samples);calls+=1
            require(result['source_count']==100 and result['noise_count']==3 and result['frame_count']==1500 and
                result['bootstrap_seed']==SEED and result['bootstrap_replicates']==REPLICATES,'Original bootstrap result metadata changed')
            result=dict(result,frame_count=300);provenance='PUBLISHED_BOOTSTRAP_BODY_POPULATION100'
            event('completed','nonmatched_intervals',index,dict(result=result,source_delta_sha256=vector_sha))
        scale=100. if m==METRICS[-1] else 1.
        unit='absolute_agreement_fraction' if scale==100 else 'dB' if m==METRICS[0] else 'LPIPS' if m==METRICS[1] else 'cosine_similarity'
        paired.append(dict(d,**result,source_delta_sha256=vector_sha,interval_provenance=provenance,unit=unit,
            display_scale=scale,display_unit='percentage_points' if scale==100 else unit,
            display_mean=scale*result['mean'],display_ci_low=scale*result['ci_low'],display_ci_high=scale*result['ci_high'],
            improvement_direction='negative' if m=='lpips_alex' else 'positive',LPIPS_sign_not_flipped=True))
        deltas.extend(dict(d,source_index=i,source_id=ids[i],mean=float(v),display_mean=scale*float(v),unit=unit,
            display_unit='percentage_points' if scale==100 else unit,noise_count=3) for i,v in enumerate(a))
    require(calls==48 and len(paired)==72 and len(deltas)==len(means)==7200,'All declared contrasts must be retained')
    return paired,means,deltas,failures,calls

def prepare(a):
    require(Path(a.out)==OUT and not OUT.exists(),'Fresh exact statistics registration required')
    closed=dict(completion=dict(path=str(METRIC_ROOT/'registered/run/completion.json'),sha256=a.metric_completion_sha256),
        owner_actual_wait=dict(path=str(METRIC_ROOT/'run_owner_actual_wait.json'),sha256=a.metric_owner_wait_sha256))
    _,ids,bound=metric_closure(closed);_,bodies=original_statistics(a.statistics_module)
    request=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',metric_closed=closed,inputs=bound,source_ids=ids,
        source_count=100,noise_count=3,noise_seeds=list(SEEDS),comparisons=definitions(),caps=CAPS,
        source_bootstrap_replicates=REPLICATES,bootstrap_seed=SEED,original_statistics=pin(a.statistics_module),
        original_function_source_sha256=bodies,source_bindings=source_bindings(),source_mean_first=True,
        private_population_override={'SOURCE_COUNT':100},returned_metadata_override={'frame_count':300},
        multiple_comparison_adjustment=False,policy_selection=False,new_model_calls=0,new_PHY_calls=0,
        automatic_retry=False,automatic_successor=False,max_seconds=900)
    OUT.mkdir(parents=True);write(OUT/'request.json',request);return pin(OUT/'request.json')

def run(a):
    require(Path(a.request)==OUT/'request.json','Exact statistics request required')
    rp=dict(path=a.request,sha256=a.request_sha256);r=readpin(rp)
    require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS and r['comparisons']==definitions() and
        r['source_bindings']==source_bindings() and r['bootstrap_seed']==SEED and r['source_bootstrap_replicates']==REPLICATES and
        r['source_mean_first'] is True and r['automatic_retry'] is False and r['max_seconds']==900,'Frozen finite statistics request changed')
    rows,ids,bound=metric_closure(r['metric_closed']);require(ids==r['source_ids'] and bound==r['inputs'],'Closed metric inputs changed')
    stat,bodies=original_statistics(r['original_statistics']['path'])
    require(pin(r['original_statistics']['path'])==r['original_statistics'] and bodies==r['original_function_source_sha256'],
        'Exact original bootstrap functions required')
    out=OUT/'run';out.mkdir();started=time.monotonic();write(out/'claim.json',dict(request=rp,pid=os.getpid(),started_unix=time.time()))
    def event(phase,kind,index,detail):
        require(time.monotonic()-started<900 and not (OUT/'STOP').exists(),'Statistics finite deadline or STOP')
        write(out/'calls'/f'{kind}-{index:03d}-{phase}.json',dict(phase=phase,kind=kind,index=index,detail=detail))
    try:
        paired,means,deltas,failures,calls=summarize(rows,ids,stat,event)
        outputs={}
        for name,table in [('paired.csv',paired),('source_means.csv',means),('source_paired_differences.csv',deltas),('failure_counts.csv',failures)]:
            path=out/name;csv_write(path,table);outputs[name]=pin(path)
        write(out/'pairs.json',definitions());outputs['pairs.json']=pin(out/'pairs.json')
        text=['# One-bin transmitter configuration mismatch: original holdout first100','',
            'True SNR is 4, 7 or 10 dB. The transmitter uses the adjacent lower, matched or adjacent higher frozen lookup configuration. The receiver demodulates using the true noise variance. These are discrete one-bin diagnostics, not a claim about typical deployment estimation error.',
            '', 'The same100 original common500 sources and three original noises (6201/6202/6203) give5400 logical method-frames. This is a post-hoc subset of an already used holdout, not a new unseen confirmation population. Frozen policies are unchanged; no diagnostic outcome selects a policy or protection threshold.',
            '', 'For every method and condition, first average the three noises separately within each source. Subtract the matched condition at the same true SNR and for the same method. The published source-bootstrap functions are unchanged, with a private source-count override from500 to100; seed2026100701,10000 draws and pointwise2.5/97.5 percentiles. One draw matrix serves48 nonmatched contrasts. The24 matched controls have exact zero differences and no interval calls. There is no multiplicity adjustment; intervals containing zero do not demonstrate equivalence.',
            '', 'All quality rows, failed outputs, negative results and zero increments remain. LPIPS uses mismatch minus matched without sign reversal: negative is better. ConvNeXt is source-prediction agreement, not accuracy. Its canonical delta and CI are absolute fractions; display_mean/display_ci_low/display_ci_high multiply all three by100 to give percentage points.',
            '', 'Failure counts use each logical frame\'s actual header/body evidence, not the first event that shared a reconstructed image. Body CRC rejection is distinct from fixed gray: the frozen raw receiver keeps actual hard tokens after body CRC rejection; only header rejection gives the fixed gray output.',
            '', 'The earlier no-bootstrap metric arithmetic averages per-noise differences with math.fsum. This statistical table follows the published source-mean-first float64 reduction. Any last-bit reduction-order difference is documented; the measured frame values are unchanged. No marginal confidence intervals are subtracted.',
            '', '| Method | True / lookup SNR (dB) | Metric | Mismatch − matched | Pointwise95% CI | Units |',
            '|---|---:|---|---:|---|---|']
        labels={'WHOLE':'Complete-scale raw transmission + VAR','PARTIAL':'Partial-scale raw transmission + VAR'}
        for d in paired:
            text.append(f"| {labels[d['family']]} | {d['actual_snr_db']} / {d['config_snr_db']} | {d['metric']} | {d['display_mean']:.8g} | [{d['display_ci_low']:.8g}, {d['display_ci_high']:.8g}] | {d['display_unit']} |")
        report=out/'REPORT.md'
        with report.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(text)+'\n')
        outputs['REPORT.md']=pin(report)
        require(source_bindings()==r['source_bindings'] and sha(r['original_statistics']['path'])==STATISTICS_SHA,'Frozen statistical code changed')
        require(pin(r['inputs']['metric_rows']['path'])==r['inputs']['metric_rows'],'Source metric rows changed')
        require(time.monotonic()-started<900 and not (OUT/'STOP').exists(),'Statistics final deadline or STOP')
        done=dict(schema=SCHEMA,status=PASS,request=rp,metric_closed=r['metric_closed'],inputs=bound,outputs=outputs,
            paired_rows=72,nonmatched_rows=48,matched_zero_rows=24,source_mean_rows=7200,source_delta_rows=7200,
            draw_matrix_calls=1,new_interval_calls=calls,unresolved_calls=0,bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,
            source_count=100,noise_count=3,logical_frames=5400,source_mean_first=True,LPIPS_sign_not_flipped=True,
            numpy=np.__version__,python=platform.python_version(),original_function_source_sha256=bodies,
            original_statistics_unchanged=True,old_intervals_modified=False,multiple_comparison_adjustment=False,
            policy_selection=False,new_model_calls=0,new_PHY_calls=0,automatic_retry=False,elapsed_seconds=time.monotonic()-started)
        write(out/'completion.json',done);return pin(out/'completion.json')
    except BaseException as error:
        write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc()));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('metric-completion-sha256','metric-owner-wait-sha256','statistics-module','out'):q.add_argument('--'+name,required=True)
    q=sub.add_parser('run');q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
    a=p.parse_args();print(canonical(prepare(a) if a.command=='prepare' else run(a)))

if __name__=='__main__':main()
