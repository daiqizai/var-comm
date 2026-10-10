"""One finite source-paired bootstrap after actually closed new100 four metrics.

The original published source-mean/draw/interval functions and seed are retained.
No policies, sources, images, models or channels are selected or evaluated here.
Preparation never draws. Unfinished paid calls and existing attempts block replay.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
import numpy as np
import ep_new100_statistics_core_v1 as core
import ep_new100_metric_core_v1 as metric
import ep_new100_content_io_v1 as io
import leo_mismatch_statistics_v2 as published

SCHEMA='H800_EP_NEW100_SOURCE_PAIRED_STATISTICS_V1'
PASS='PASS_H800_EP_NEW100_48SUMMARY_24PAIRED_STATISTICS'
CORE_SHA='e5513d8ef0d885d91ef809d38a6cc111c8a472a09841c517a15824768ce49321'
METRIC_CORE_SHA='47b1dc3f6ab563c16e9779067c95f9c4cd8647ec86a13d1372778d25fb757427'
PUBLISHED_SHA='f4f2dd4d65eef5b06f9c82c25d828d60f2024bfb78522d379572fbfe9a469cd2'
METRIC_OWNER_NAME='h800_ep_new100_metric_owner_v3.py'
METRIC_OWNER_SHA='562f209b8db97268fad8a5fdbfa70659ebcc4d73cfc2c07ddda86ec844f4f672'
METRIC_ROOT=io.RT/'qualification/h800_ep_new100_metrics_v3_attempt1'
OUT=io.RT/'qualification/h800_ep_new100_statistics_v1_attempt1'
CAPS=core.CAPS
require=io.require;read=io.read;pin=io.pin;write=io.save


def unchanged(desc):
    actual=pin(desc['path'])
    return actual['sha256']==desc['sha256'] and ('bytes' not in desc or actual['bytes']==desc['bytes'])


def visual_metadata_closure(r,m):
    # Authenticate closed metadata without re-entering an expired execution window
    # or loading either reference or reconstruction pixel arrays.
    v=m.visual_implementation();root=io.RT/'qualification/h800_ep_new100_visual_v3_attempt1'
    pins=r['visual_closed']
    require(set(pins)=={'completion','owner_actual_wait'} and
        pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact alternate visual closure required')
    done=read(pins['completion']);wait=read(pins['owner_actual_wait'])
    rp=dict(path=str(root/'registered/request.json'),sha256=done['request_sha256']);vr=read(rp)
    worker=read(done['worker_completion']);result=worker['results']
    require(done['schema']==worker['schema']==vr['schema']==v.SCHEMA and done['status']==worker['status']==v.PASS and
        done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and done['actual_wait']['success'] is True and
        wait['actual_wait'] is True and wait['returncode']==0 and not wait.get('timeout',False) and not wait.get('interrupted_or_timeout',False) and
        worker['request_sha256']==done['request_sha256'] and
        done['worker_completion']['path']==str(root/'registered/run/worker_completion.json') and
        read(pin(root/'registered/run/actual_child_wait.json'))==done['actual_wait'],'Actual complete visual child/owner waits required')
    metric.plan.check_ledger(result['counts'],metric.plan.scientific_caps()['visual'])
    counts=result['counts']['completed']
    require(vr['records']==r['records'] and vr['inputs']==r['inputs'] and vr['logical_events']==r['logical_events'] and
        result['logical_frames']==r['reconstruction_results'] and len(result['logical_frames'])==3600 and
        worker['quality_scores']==0 and counts['model_load']==1 and counts['encoder']==counts['source_tx']==0 and
        counts['source_rx']<=600 and counts['var_render']==counts['decoder_forward'] and vr['target_index']==r['target_index'] and
        vr['device_identity_binding']==r['device_identity_binding'],'Every score must use the closed same-device actual reconstruction')
    previous,recovery=v.failed_attempt()
    require(vr['recovery_budget']==recovery and vr['CPU_closed']==previous['CPU_closed'] and
        recovery['previous_model_load_attempts']==1 and recovery['previous_model_load_completed']==0 and
        recovery['previous_unresolved_model_load']==1 and recovery['additional_model_load_attempt_cap']==1 and
        recovery['cumulative_model_load_attempt_cap']==2 and recovery['prior_failed_ledger_preserved'] is True and
        all(value==0 for value in recovery['previous_scientific_calls'].values()) and
        recovery['cumulative_scientific_caps']=={k:value for k,value in metric.plan.scientific_caps()['visual'].items() if k!='model_load'},
        'Preserved failed startup and unchanged cumulative scientific budget required')
    return dict(completion=pins['completion'],owner_actual_wait=pins['owner_actual_wait'],request=rp,worker=done['worker_completion'],
        recovery_budget=recovery,successful_attempt_model_loads=counts['model_load'],cumulative_model_load_attempts=2,
        previous_failed_model_load_reservation_still_unresolved=True)


def metric_implementation():
    path=Path(__file__).with_name(METRIC_OWNER_NAME)
    require(re.fullmatch('[0-9a-f]{64}',METRIC_OWNER_SHA) and io.sha(path)==METRIC_OWNER_SHA,
        'Final admitted alternate-GPU metric owner required; a preparation placeholder cannot execute')
    spec=importlib.util.spec_from_file_location('_new100_closed_metric_for_statistics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_bindings():
    require(io.sha(core.__file__)==CORE_SHA and io.sha(metric.__file__)==METRIC_CORE_SHA and
        io.sha(published.__file__)==PUBLISHED_SHA,'Frozen statistics/metric/core changed')
    m=metric_implementation()
    values=published.source_bindings()
    for module in (core,core.confirmation,metric,io,published,m):values[str(Path(module.__file__).resolve())]=io.sha(module.__file__)
    values[str(Path(__file__).resolve())]=io.sha(__file__);return values


def metric_closure(pins):
    m=metric_implementation();root=METRIC_ROOT/'registered'
    require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'run/completion.json') and
        pins['owner_actual_wait']['path']==str(METRIC_ROOT/'run_owner_actual_wait.json'),'Exact actual alternate-H800 new100 metric attempt required')
    done=read(pins['completion']);outer=read(pins['owner_actual_wait']);rp=pin(root/'request.json')
    require(rp['sha256']==done['request_sha256'],'Actual closed metric request changed')
    r=read(rp);worker=read(done['worker_completion']);result=worker['results']
    require(done['schema']==worker['schema']==r['schema']==m.SCHEMA and done['status']==worker['status']==m.PASS and
        done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and done['actual_wait']['success'] is True and
        outer['actual_wait'] is True and outer['returncode']==0 and not outer.get('timeout',False) and not outer.get('interrupted_or_timeout',False) and
        worker['request_sha256']==done['request_sha256'] and done['worker_completion']['path']==str(root/'run/worker_completion.json'),
        'Actual complete metric owner, worker and outer wait required before statistics')
    require(done['bootstrap_calls']==worker['bootstrap_calls']==result['bootstrap_calls']==r['bootstrap_calls']==0 and
        all(x['policy_selection'] is False for x in (done,worker,result,r)) and r['historical_score_reuse_allowed'] is False and
        result['historical_score_reuse']==0 and worker['population']==result['population']==r['population']==metric.POPULATION,
        'No prior bootstrap, policy selection or historical score reuse is admitted')
    require(r['tool_bindings'][METRIC_OWNER_NAME]==METRIC_OWNER_SHA and r['tool_bindings']['ep_new100_metric_core_v1.py']==METRIC_CORE_SHA,
        'Actual metric owner and unchanged four computations required')
    for name,h in r['tool_bindings'].items():
        require('/' not in name and '\\' not in name and io.sha(Path(__file__).with_name(name))==h,'Closed metric code changed: '+name)
    require(read(pin(root/'run/actual_child_wait.json'))==done['actual_wait'],'Actual metric child wait differs from completion')
    visual=visual_metadata_closure(r,m)
    inputs=r['inputs'];selection=read(inputs['selection']);policy=read(inputs['policy_bundle']);content=read(inputs['content_gate']);manifest=read(inputs['source_manifest'])
    expected=core.confirmation.frame_grid(selection,inputs['selection'],content,policy)
    require(r['logical_events']==expected and r['records']==manifest['records'] and manifest['source_count']==100 and
        manifest['policy_bundle']==inputs['policy_bundle'] and r['metric_config']['confirmation_source_manifest']==inputs['source_manifest'] and
        r['metric_config']['confirmation_policy_bundle']==inputs['policy_bundle'],'Complete same fixed100 and frozen four policies required')
    rows=read(result['metric_rows']);metric.validate_completion(result,rows,r)
    identity=read(worker['metric_identity'])
    require(identity['identity']==core.confirmation.digest(identity['metadata']) and identity['metadata']['population']==metric.POPULATION and
        identity['metadata']['policy_selection'] is False and identity['metadata']['selection_objective'] is None,
        'Actual independent new100 evaluator identity required')
    require(identity['actual_runtime']==m.g.EXPECTED_RUNTIME,'Frozen actual metric runtime changed')
    # Check linked actual metric/reconstruction bytes, never open image arrays.
    pairs={};reconstructions={}
    for index,row in enumerate(rows):
        pp=row['metric_pair_result'];key=core.confirmation.canonical(pp)
        if key not in pairs:pairs[key]=read(pp)
        pair=pairs[key];ap=row['reconstruction_result'];ak=core.confirmation.canonical(ap)
        if ak not in reconstructions:reconstructions[ak]=read(ap)
        actual=reconstructions[ak];recon=actual['reconstruction'];record=r['records'][row['source_index']]
        require(ap==r['reconstruction_results'][index] and actual['frame_index']==index and actual['logical_event']==expected[index] and
            actual['physical_frame']==row['physical_frame'] and all(row[k]==actual[k] for k in metric.FAILURE_FIELDS),
            'Every scored logical event and its own failures must match the actual reconstruction')
        require(pair['metric_identity']==identity['identity'] and pair['metric_pair_key']==row['metric_pair_key'] and
            pair['population']==metric.POPULATION and pair['historical_score_reuse'] is False and
            pair['reference_archive']==record['pixels_archive'] and pair['reconstruction_archive']==recon['image_archive'] and
            pair['reconstruction_pixel_sha256']==recon['image_sha256'] and all(pair['metrics'][k]==row[k] for k in metric.METRICS),
            'Reported four scores must match the exact actual source/reconstruction pair')
        require(row['metric_pair_key']==core.confirmation.digest(dict(reference_sha256=pair['reference_pixel_sha256'],
            reconstruction_sha256=pair['reconstruction_pixel_sha256'],metric_identity=identity['identity'])),
            'Both pixel identities and evaluator identity must bind every metric pair')
    require(len(pairs)==result['actual_unique_pairs'] and len(reconstructions)==3600,'Actual scored-pair and logical-reconstruction counts differ')
    return rows,expected,dict(metric_request=rp,metric_worker=done['worker_completion'],metric_rows=result['metric_rows'],
        metric_identity=worker['metric_identity'],metric_identity_sha256=identity['identity'],source_manifest=inputs['source_manifest'],
        policy_bundle=inputs['policy_bundle'],selection=inputs['selection'],content_gate=inputs['content_gate'],
        metric_runtime_python=r['spec']['python'],metric_runtime_identity=r['metric_config']['runtime_identity'],
        metric_actual_runtime=identity['actual_runtime'],visual_closed=visual)


class CallLedger:
    def __init__(self,out,guard):
        self.out=Path(out);self.guard=guard;self.pending={};self.counts={k:0 for k in CAPS};self.completed={k:0 for k in CAPS}
        self.out.mkdir()
    def event(self,phase,kind,index,detail):
        self.guard();require(kind in CAPS and type(index) is int and index>=0 and phase in ('reserved','completed'),'Finite statistics call identity required')
        key=kind,index;path=self.out/f'{kind}-{index:03d}-{phase}.json'
        if phase=='reserved':
            require(key not in self.pending and self.counts[kind]<CAPS[kind],'Statistics duplicate or call cap exhausted')
            write(path,dict(phase=phase,kind=kind,index=index,detail=detail));self.pending[key]='reserved';self.counts[kind]+=1
        else:
            require(self.pending.get(key)=='reserved','Statistics completion without a unique reservation')
            write(path,dict(phase=phase,kind=kind,index=index,detail=detail));self.pending[key]='completed';self.completed[kind]+=1
    def summary(self):return dict(caps=CAPS,reserved=dict(self.counts),completed=dict(self.completed),unresolved=sum(v=='reserved' for v in self.pending.values()))


def prepare(a):
    require(io.inside(a.out)==OUT and not OUT.exists(),'One fresh exact new100 statistics registration required')
    closed=dict(completion=dict(path=str(METRIC_ROOT/'registered/run/completion.json'),sha256=a.metric_completion_sha256),
        owner_actual_wait=dict(path=str(METRIC_ROOT/'run_owner_actual_wait.json'),sha256=a.metric_owner_wait_sha256))
    _,expected,inputs=metric_closure(closed);_,bodies=published.original_statistics(io.inside(a.statistics_module))
    require(type(a.max_seconds) is int and 0<a.max_seconds<=900 and 0<a.deadline_unix-time.time()<=a.max_seconds and
        len(a.cpu_slots)==len(set(a.cpu_slots))==2 and all(type(v) is int and v>=0 for v in a.cpu_slots),
        'Bounded900-second2CPU statistics window required')
    summaries,pairs=core.definitions();r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',metric_closed=closed,inputs=inputs,
        expected_events=expected,source_count=100,noise_count=3,noise_seeds=list(core.confirmation.SEEDS),summaries=summaries,pairs=pairs,caps=CAPS,
        original_statistics=pin(io.inside(a.statistics_module)),original_function_source_sha256=bodies,source_bindings=source_bindings(),
        source_mean_first=True,source_bootstrap_replicates=core.REPLICATES,bootstrap_seed=core.SEED,
        private_population_override={'SOURCE_COUNT':100},returned_metadata_override={'frame_count':300},
        deadline_unix=a.deadline_unix,max_seconds=a.max_seconds,CPU_slots=a.cpu_slots,CUDA_VISIBLE_DEVICES='',
        automatic_retry=False,automatic_successor=False,multiple_comparison_adjustment=False,old500_mean_subtraction=False,
        new_model_calls=0,new_image_scores=0,new_PHY_calls=0,policy_selection=False)
    OUT.mkdir(parents=True);return write(OUT/'request.json',r)


def registration(path,digest):
    require(io.inside(path)==OUT/'request.json','Exact independent confirmation statistics request required')
    r=read(dict(path=str(path),sha256=digest));summary,pairs=core.definitions()
    require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS and r['summaries']==summary and
        r['pairs']==pairs and r['source_bindings']==source_bindings() and r['source_count']==100 and r['noise_count']==3 and
        r['noise_seeds']==list(core.confirmation.SEEDS) and r['bootstrap_seed']==core.SEED and r['source_bootstrap_replicates']==core.REPLICATES and
        r['source_mean_first'] is True and r['automatic_retry'] is False and r['automatic_successor'] is False and
        r['old500_mean_subtraction'] is False and r['policy_selection'] is False and
        r['new_model_calls']==r['new_PHY_calls']==r['new_image_scores']==0 and type(r['max_seconds']) is int and 0<r['max_seconds']<=900 and
        0<r['deadline_unix']-time.time()<=r['max_seconds'] and r['CUDA_VISIBLE_DEVICES']=='' and
        len(r['CPU_slots'])==len(set(r['CPU_slots']))==2 and all(type(v) is int and v>=0 for v in r['CPU_slots']) and
        r['private_population_override']=={'SOURCE_COUNT':100} and r['returned_metadata_override']=={'frame_count':300} and
        r['multiple_comparison_adjustment'] is False,
        'Frozen finite new100 statistics definitions or implementation changed')
    rows,expected,inputs=metric_closure(r['metric_closed']);require(expected==r['expected_events'] and inputs==r['inputs'],'Closed complete actual metric inputs changed')
    stat,bodies=published.original_statistics(io.inside(r['original_statistics']['path']))
    require(pin(r['original_statistics']['path'])==r['original_statistics'] and bodies==r['original_function_source_sha256'],
        'Published draw and interval source bodies changed')
    return r,rows,expected,stat


def run(a):
    path=io.inside(a.request);r,rows,expected,stat=registration(path,a.request_sha256)
    require(sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
        set(os.sched_getaffinity(0))==set(r['CPU_slots']) and str(Path(sys.executable))==r['inputs']['metric_runtime_python'],
        'Frozen metric Python, hidden CUDA and exactly2CPU required')
    require(np.__version__==r['inputs']['metric_actual_runtime']['numpy'] and
        platform.python_version()==r['inputs']['metric_actual_runtime']['python'],'Actual statistics NumPy/Python must match the frozen metric runtime')
    out=OUT/'run';require(not out.exists(),'Statistics already attempted; preserve previous paid calls without replay');out.mkdir()
    started=time.monotonic();rp=pin(path);write(out/'claim.json',dict(request=rp,pid=os.getpid(),started_unix=time.time(),CPU_slots=r['CPU_slots']))
    def guard():require(time.monotonic()-started<r['max_seconds'] and time.time()<r['deadline_unix'] and not (OUT/'STOP').exists() and not (io.BASE/'STOP').exists(),
        'Statistics deadline or STOP; preserve paid work')
    ledger=CallLedger(out/'calls',guard)
    try:
        guard();result=core.summarize(rows,expected,stat,ledger.event);counts=ledger.summary()
        require(counts['reserved']==counts['completed']==result['actual_calls'] and counts['unresolved']==0 and counts['completed']['draw_matrix']==1,
            'Every actual statistics call must close once within the original caps')
        outputs={}
        for name,key in (('summary.csv','summary'),('paired.csv','paired'),('source_means.csv','source_means'),('source_paired_differences.csv','source_paired_differences')):
            published.csv_write(out/name,result[key]);outputs[name]=pin(out/name)
        outputs['summary.json']=write(out/'summary.json',result['summary']);outputs['paired.json']=write(out/'paired.json',result['paired'])
        outputs['points.json']=write(out/'points.json',r['summaries']);outputs['pairs.json']=write(out/'pairs.json',r['pairs'])
        outputs['fairness_notes.json']=write(out/'fairness_notes.json',dict(source_count=100,noise_count=3,noise_seeds=r['noise_seeds'],
            source_pairing='same fixed source across all four methods; three noises averaged first with numpy.mean float64',
            comparisons='partial minus whole separately within raw and entropy representations; not added',
            bootstrap=dict(seed=core.SEED,replicates=core.REPLICATES,draw_matrix_calls=1,quantiles=[2.5,97.5],unit='source',multiple_comparison_adjustment=False),
            original_function_source_sha256=r['original_function_source_sha256'],LPIPS_sign_unchanged=True,
            visual_startup_provenance=r['inputs']['visual_closed'],
            ConvNeXt='source-prediction agreement, not classification accuracy; paired display is percentage points',
            all_failure_rows_retained=True,old500_mean_subtraction=False,policy_selection=False))
        text=['# Independent new100 N1024 four-arm confirmation','',
            'All four frozen methods use the same 100 previously selected sources at 4, 10 and 19 dB, with seeds 9301, 9302 and 9303. Every actual failure is retained. This table does not subtract old 500-source means.',
            '', 'First average three noises separately per source using the original numpy.mean float64 expression. The unchanged published draw and interval functions use 10000 source bootstrap replicates, seed 2026100701 and pointwise 2.5/97.5 percentiles. One draw matrix serves 48 marginal summaries and 24 predeclared paired comparisons. Exact zero vectors and identical vectors are retained with explicit interval provenance. There is no multiplicity adjustment.',
            '', 'Paired differences are partial minus whole separately for raw and entropy methods; the two contrasts are not added. LPIPS signs are unchanged and negative paired differences favor partial. ConvNeXt measures source-prediction agreement, not accuracy; fraction estimates and interval endpoints are all multiplied by 100 for display in percent or paired percentage points.',
            '', 'Policies were frozen before this population was selected. No source, policy, model, channel, reconstruction or score was rerun by this statistics stage. Original metric files and earlier confidence intervals remain unchanged.']
        with (out/'REPORT.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(text)+'\n')
        outputs['REPORT.md']=pin(out/'REPORT.md');guard()
        require(source_bindings()==r['source_bindings'] and unchanged(r['inputs']['metric_rows']),
            'Closed scientific inputs changed during statistics')
        done=dict(schema=SCHEMA,status=PASS,request=rp,metric_closed=r['metric_closed'],inputs=r['inputs'],outputs=outputs,
            counts=counts,summary_rows=48,paired_rows=24,source_mean_rows=4800,source_delta_rows=2400,logical_frames=3600,
            source_count=100,noise_count=3,draw_matrix_calls=1,bootstrap_seed=core.SEED,bootstrap_replicates=core.REPLICATES,
            original_function_source_sha256=r['original_function_source_sha256'],original_statistics_unchanged=True,
            LPIPS_sign_unchanged=True,old500_mean_subtraction=False,policy_selection=False,new_model_calls=0,new_image_scores=0,new_PHY_calls=0,
            old_intervals_modified=False,automatic_retry=False,automatic_successor=False,elapsed_seconds=time.monotonic()-started,
            numpy=np.__version__,python=platform.python_version())
        return write(out/'completion.json',done)
    except BaseException as error:
        write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),counts=ledger.summary()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('prepare')
    for name in ('metric-completion-sha256','metric-owner-wait-sha256','statistics-module','out'):q.add_argument('--'+name,required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=900)
    q.add_argument('--cpu-slots',type=lambda x:[int(v) for v in x.split(',')],required=True)
    q=sub.add_parser('run');q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
    a=p.parse_args();print(json.dumps(prepare(a) if a.command=='prepare' else run(a)))


if __name__=='__main__':main()
