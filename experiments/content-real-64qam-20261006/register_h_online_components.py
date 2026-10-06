"""Independent metadata-only registration of the frozen fixed16 H timing job."""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import h_online_components_driver as d

REGISTERED='H_ONLINE_COMPONENTS_REGISTERED_NOT_LAUNCHED'
QUALIFIED='H_ONLINE_COMPONENTS_CPU_INTERFACE_QUALIFICATION_PASS'
P_DONE='P600_RECEIVED_LATENT_CONVNEXT_COMPLETE'


def normal_p600_closed(spec,ctx):
    """Original owner/worker receipt; verify sealed historical graph, not current enumeration."""
    d.require(set(spec)=={'config','owner_config','registration','launch','completion'},'Exact P600 predecessor required')
    cfg=d.read(spec['config']);reg=d.read(spec['registration'])
    bound=d.merge(reg['source_bindings'],reg['input_bindings'])
    for p in (spec['config'],spec['owner_config'],cfg['visual_source_closure']):
        d.require(bound.get(p)==d.sha(p),'P600 dependency/config unbound: '+p)
    d.require(cfg['schema']=='P600_RECEIVED_REPLAY_CONFIG_V1' and cfg['registration']==spec['registration']
        and cfg['visual_owner_config']==spec['owner_config'] and cfg['metric_batch']==ctx['cfg']['metric_batch'],
        'P600 must descend from the same completed H18 metric job')
    for k in ('root','H_out','owner_module','wait_module','cpu_driver_module','ledger',
              'protocol','budget_registration','phase_limits','stop_file'):
        d.require(cfg[k]==ctx['cfg'][k],'P600/H timing execution mismatch: '+k)
    expected=dict(status=P_DONE,source_count=100,frame_count=600,P_policy_snr_points=2,N=1024,
        snrs_db=[13,19],noise_seeds=[2001,2002,2003],RGB_exact_parity=True,legacy_scalars_preserved=True,
        ConvNeXt_scored=True,new_packet_decodes=0,new_noise_draws=0,budget_writes=0,
        policy_selection=False,development_used=True,holdout_used=False,online_latency_measured=False)
    pc=ctx['prior']['context']
    closed=pc['cpu'].closed_batch(ctx['owner'],pc['wait'],
        dict(config=spec['owner_config'],registration=spec['registration'],launch=spec['launch'],completion=spec['completion']),
        cfg['owner_module'],expected,ctx['owner'].raw_process_state)
    old=closed['owner'];done=closed['done']
    d.require(len(old['stages'])==1 and old['stages'][0]['id']=='development'
        and old['stages'][0]['resource']=='gpu' and len(old['stages'][0]['jobs'])==1,'Single original P600 GPU owner required')
    job=old['stages'][0]['jobs'][0];metric_job=ctx['prior']['closed']['owner']['stages'][0]['jobs'][0]
    d.require(job['id']=='p600_received_replay' and job['argv']==
        [metric_job['argv'][0],'-B',cfg['replay_driver_module'],'--config',spec['config']],
        'P600 original command/interpreter differs')
    graph=d.read(cfg['visual_source_closure'])
    d.require(graph['status']=='EXACT_SOURCE_CLOSURE_MATCH' and done['visual_source_bindings']==graph['source_bindings']
        and all(bound.get(p)==s for p,s in graph['source_bindings'].items()),'P600 historical source graph changed')
    d.require(done['source_ids']==ctx['source_ids'] and done['budget_before']==done['budget_after']
        ==closed['owner_done']['budget']==ctx['before']
        and done['metric_predecessor_completion_sha256']==d.sha(ctx['cfg']['metric_batch']['completion']),
        'P600 normal receipt population/budget/predecessor differs')
    return dict(spec=dict(spec),cfg=cfg,closed=closed,before=ctx['before'],
        bindings=d.merge(bound,closed['bindings'],done['outputs'],d.bind(spec.values())))


def validate_request(r):
    d.require(r['schema']=='H_ONLINE_COMPONENTS_REGISTRATION_REQUEST_V1','Wrong independent timing request')
    cfg=copy.deepcopy(r['config']);e=Path(r['execution_dir']);out=Path(cfg['out']);h=Path(cfg['H_out'])
    d.require(e.is_absolute() and out.is_absolute() and h in e.parents and h in out.parents
        and e!=out and e not in out.parents and out not in e.parents,'Fresh independent timing directories required')
    d.require(cfg['schema']=='H_ONLINE_COMPONENTS_CONFIG_V1'
        and cfg['registration']==str(e/'execution_registration.json')
        and cfg['visual_owner_config']==str(e/'owner_config.json'),'Timing planned paths differ')
    d.require(type(cfg['max_seconds']) is int and 0<cfg['max_seconds']<=14400
        and cfg['overall_deadline_unix']==d.DEADLINE,'Original timing deadline/cap required')
    d.require(not e.exists() and not out.exists(),'Prior timing attempt preserved; no retry')
    return cfg,e,out


def make_owner(cfg,ctx,e):
    old=ctx['prior']['closed']['owner'];owner=copy.deepcopy(old)
    d.require(old['gpu_threads']==6 and len(old['stages'])==1 and old['stages'][0]['resource']=='gpu'
        and len(old['stages'][0]['jobs'])==1,'Original single metric GPU owner required')
    python=old['stages'][0]['jobs'][0]['argv'][0]
    owner.update(owner_out=str(e),registration=cfg['registration'])
    expected=dict(source_count=16,component_case_count=288,H_policy_snr_points=18,noise_seed=6201,
        warmup_repetitions=1,measured_repetitions=3,batch_size=1,exact_TX_parity=True,exact_RX_parity=True,
        per_component_direction_repetitions=1152,budget_writes=0,new_packet_decodes=0,new_noise_draws=0,
        new_quality_samples=0,new_metric_calls=0,policy_selection=False,development_used=True,holdout_used=False,
        GPU_used=True,online_source_visual_components_measured=True,exclusive_PHY_latency_measured=False,
        PHY_encode_measured=False,end_to_end_latency_measured=False,
        instrumented_uncached_cost_not_optimized_production_latency=True,
        MAIN_complete=False,overall_development_complete=False,H_full_delivery_claimed=False,
        own_owner_success_not_yet_certified=True)
    owner['stages']=[dict(id='development',resource='gpu',requires=[],max_seconds=cfg['max_seconds'],jobs=[
        dict(id='h_online_components',argv=[python,'-B',cfg['timing_driver_module'],'--config',str(e/'timing_config.json')],
            cwd=cfg['root'],out=cfg['out'],completion=str(Path(cfg['out'])/'completion.json'),
            accepted_statuses=[d.DONE],receipt_expect=expected)])]
    return owner,python


def budget_guard(ctx):
    cfg=ctx['cfg'];now=ctx['owner'].budget_snapshot(cfg['ledger'],d.sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
    d.require(now==ctx['before'] and now['charged']==164760 and now['phase_charged']['development']==10800
        and now['development_remaining']==2400 and now['failed']==now['unresolved']==0,'H budget changed; no PHY may overlap timing')
    d.require(not Path(cfg['stop_file']).exists() and time.time()<d.DEADLINE,'STOP/deadline blocks timing')


def qualification(q,sources,inputs,python,cfg):
    d.require(q['status']==QUALIFIED and q['python']==python and q['GPU_used'] is False
        and q['models_constructed'] is False and q['new_packet_decodes']==q['new_noise_draws']==0
        and q['results'] and all(v['exit_code']==0 and v['test_count']>0 for v in q['results']),
        'Actual CPU-only timing qualification required')
    for key in ('source_bindings','input_bindings','outputs'):d.verify(q[key])
    for p,s in q['source_bindings'].items():d.require(sources.get(p)==s,'Qualified timing dependency omitted')
    for p,s in d.merge(q['input_bindings'],q['outputs']).items():d.require(inputs.get(p)==s,'Qualification input/evidence omitted')
    for p in (str(Path(__file__).absolute()),cfg['timing_driver_module'],cfg['timing_core_module']):
        d.require(q['source_bindings'].get(p)==d.sha(p),'Timing registrar/core/driver not qualified')


def register(request_path):
    d.require(sys.platform.startswith('linux'),'Actual Linux predecessor/process admission required')
    request_path=Path(request_path).absolute();r=d.read(request_path);cfg,e,out=validate_request(r)
    sources=d.merge(r['source_bindings'],d.bind((Path(__file__).absolute(),Path(d.__file__).absolute())))
    inputs=d.merge(r['input_bindings'],d.bind((request_path,)));qp=r['prepared_qualification']
    d.require(inputs.get(qp)==d.sha(qp),'Timing qualification must be explicitly pinned')
    ctx=d.inspect_inputs(cfg,d.merge(sources,inputs))
    p600=normal_p600_closed(r['p600_batch'],ctx)
    d.require(all(d.merge(sources,inputs).get(p)==s for p,s in p600['bindings'].items()),
        'Complete normally closed P600 lineage must be explicitly bound')
    owner,python=make_owner(cfg,ctx,e);qualification(d.read(qp),sources,inputs,python,cfg)
    d.require(set(owner['gpu_affinity'])<=os.sched_getaffinity(0)
        and shutil.disk_usage(cfg['H_out']).free>=d.MAX_OUTPUT_BYTES+8*1024**3,
        'Original GPU affinity/storage headroom unavailable')
    budget_guard(ctx);e.mkdir()
    try:
        cp,op,rp=e/'timing_config.json',e/'owner_config.json',e/'execution_registration.json'
        gate=e/'predecessor_closure.json'
        d.save(gate,dict(status='H_METRICS_AND_P600_NORMALLY_CLOSED_FOR_TIMING',
            metric_bindings=ctx['prior']['bindings'],P600_bindings=p600['bindings'],
            metric_batch=cfg['metric_batch'],p600_batch=r['p600_batch'],budget=ctx['before'],
            original_owner_success=True,original_workers_exit_zero=True,GPU_used=False,new_packet_decodes=0))
        d.save(cp,cfg);d.save(op,owner);inputs=d.merge(inputs,d.bind((cp,op,gate)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=e.name,
            source_stage_scope=d.SCOPE,allowed_stage_ids=['development'],owner_config_sha256=d.sha(op),
            phase_limits=cfg['phase_limits'],budget_registration_sha256=d.sha(cfg['budget_registration']),
            scientific_protocol_sha256=d.sha(cfg['protocol']),source_bindings=sources,input_bindings=inputs,
            budget_before=ctx['before'],timing_design=ctx['scope'],p600_batch=r['p600_batch'],
            source_count=16,component_case_count=288,H_policy_snr_points=18,noise_seed=6201,
            warmup_repetitions=1,measured_repetitions=3,batch_size=1,GPU_jobs=1,GPU_threads=6,
            GPU_affinity=owner['gpu_affinity'],new_packet_decodes=0,new_noise_draws=0,new_metric_calls=0,
            new_quality_samples=0,budget_writes=0,policy_selection=False,holdout_started=False,
            MAIN_started=False,C_started=False,H_full_delivery_claimed=False,future_stage_automatic=False,
            scientific_protocol_modified=False,created_unix=time.time(),
            shared_ledger_exclusion='No new physical decoder or other visual worker while the timing owner is active.',
            historical_PHY_scope='Already-paid concurrent decoder+bookkeeping windows; not isolated PHY or end-to-end latency.')
        ctx['owner'].validate_config(owner,reg,d.sha(op))
        d.verify(sources);d.verify(inputs);budget_guard(ctx);d.save(rp,reg)
        result=dict(status=REGISTERED,registration_sha256=d.sha(rp),owner_config_sha256=d.sha(op),
            outputs=d.bind((cp,op,rp,gate)),workers_started=False,GPU_used=False,models_constructed=False,
            new_packet_decodes=0,new_noise_draws=0,budget=ctx['before'],future_stage_automatic=False,
            required_launch_condition='Exact successful registration, normal H18 metric and P600 closure, unchanged budget, no STOP, original exclusive GPU admission.')
        d.save(e/'registration_completion.json',result);return result
    except BaseException:
        d.save(e/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True)
    print(register(p.parse_args().request)['status'])
