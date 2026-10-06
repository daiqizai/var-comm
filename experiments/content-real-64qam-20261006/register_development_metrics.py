"""Seal one metric-only H18 GPU stage after the real renderer normally exits.

This registrar loads no model, scores no image and starts no process. Historical
render source bindings are verified as sealed, without re-enumerating new code.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import h_development_metrics_driver as d

PINS = ('metric_driver_module', 'metric_adapter_module', 'prepared_qualification', 'metric_assets')
BATCH_KEYS = ('config', 'owner_config', 'registration', 'launch', 'completion')
REGISTERED = 'H_DEVELOPMENT_METRICS_REGISTERED_NOT_LAUNCHED'


def pin(value):
    p = Path(value['path'])
    d.require(p.is_absolute() and p.is_file() and not p.is_symlink()
              and d.sha(p) == value['sha256'], 'Pinned metric input missing or changed: ' + str(p))
    return p


def qualification(value, paths, python):
    d.require(value['status'] == 'H_DEVELOPMENT_METRICS_CPU_QUALIFICATION_PASS'
              and value['GPU_used'] is False and value['new_packet_decodes'] == 0
              and value['real_metric_calls'] == 0 and value['python'] == python
              and value['results'] and all(x['exit_code'] == 0 for x in value['results']),
              'Closed CPU implementation qualification in the actual metric interpreter required')
    for key in ('source_bindings', 'input_bindings', 'outputs'): d.verify(value[key])
    for p in paths:
        d.require(value['source_bindings'].get(str(p)) == d.sha(p), 'Qualification misses actual metric entry: ' + str(p))


def inspect_inputs(request, paths):
    d.require(set(request['render_batch']) == set(BATCH_KEYS), 'Exact render predecessor pins required')
    spec = {k: str(pin(request['render_batch'][k])) for k in BATCH_KEYS}
    ctx = d.normal_render_closed(spec)
    old = ctx['old']; prior = ctx['render_cfg']; before = ctx['before']
    d.require(before['charged'] == 164760 and before['phase_charged']['development'] == 10800
              and before['development_remaining'] == 2400 and before['unresolved'] == before['failed'] == 0,
              'Completed H18 budget and untouched MAIN reserve required')
    python = old['stages'][0]['jobs'][0]['argv'][0]
    # absolute(), never resolve(): the venv interpreter may itself be a symlink.
    d.require(request['python'] == python and Path(python).is_absolute() and Path(python).is_file(),
              'Use the actual completed render GPU interpreter, preserving its venv path')
    ma, mb, flags = d.metric_assets(paths['metric_assets'])
    d.require(old['gpu_threads'] == flags['threads'] == 6 and flags['interop_threads'] == 2,
              'Original metric numerical thread settings differ')
    q = d.read(paths['prepared_qualification'])
    qualification(q, (Path(__file__).absolute(), paths['metric_driver_module'], paths['metric_adapter_module']), python)
    d.require(q['input_bindings'].get(str(paths['metric_assets'])) == d.sha(paths['metric_assets']),
              'CPU qualification must bind the exact original metric-asset manifest')
    d.require(str(paths['metric_driver_module']) == str(Path(d.__file__).absolute()),
              'Registrar imported a different metric implementation')
    d.require(len(ctx['source_ids']) == len(set(ctx['source_ids'])) == 100 and len(ctx['schedule']) == 18,
              'The fixed original100 and eighteen H declarations are required')
    ctx.update(spec=spec,metric_assets=ma,metric_bindings=mb,metric_flags=flags,qualification=q,
               render_registration=d.read(spec['registration']),python=python)
    return ctx


def make_configs(request, ctx, paths):
    prior=ctx['render_cfg']; e=Path(request['execution_dir'])
    cp=e/'metrics_config.json'; rp=e/'execution_registration.json'; op=e/'owner_config.json'
    cfg={k:copy.deepcopy(prior[k]) for k in ('root','H_out','owner_module','wait_module','cpu_driver_module',
        'ledger','budget_registration','phase_limits','stop_file','protocol')}
    cfg.update(schema='H_DEVELOPMENT_METRICS_CONFIG_V1',out=request['metrics_out'],registration=str(rp),
        visual_owner_config=str(op),render_driver_module=prior['visual_driver_module'],
        metric_driver_module=str(paths['metric_driver_module']),metric_adapter_module=str(paths['metric_adapter_module']),
        metric_assets=str(paths['metric_assets']),render_batch=copy.deepcopy(ctx['spec']),
        max_seconds=request['max_seconds'],overall_deadline_unix=d.u.DEADLINE,**copy.deepcopy(ctx['metric_assets']['paths']))
    owner=copy.deepcopy(ctx['old']);owner.update(owner_out=str(e),registration=str(rp))
    expected=dict(source_count=100,frame_count=5400,H_policy_snr_points=18,MAIN_frames=0,
        new_packet_decodes=0,policy_selection=False,development_used=True,holdout_used=False,
        unified_neural_metrics_run=True,statistical_aggregation_run=False,
        MAIN_complete=False,P_complete=False,overall_development_complete=False,H_full_delivery_claimed=False,
        online_timing_measured=False)
    # The original owner has no 'metrics' stage. Its existing development GPU
    # stage schedules only this explicitly bound scoring entry and receipt.
    owner['stages']=[dict(id='development',resource='gpu',requires=[],max_seconds=request['max_seconds'],jobs=[
        dict(id='development_metrics',argv=[request['python'],'-B',str(paths['metric_driver_module']),'--config',str(cp)],
             cwd=prior['root'],out=request['metrics_out'],completion=str(Path(request['metrics_out'])/'completion.json'),
             accepted_statuses=[d.DONE],receipt_expect=expected)])]
    return cfg,owner


def register(request_path):
    request_path=Path(request_path).absolute(); r=d.read(request_path)
    d.require(r['schema']=='H_DEVELOPMENT_METRICS_REGISTRATION_REQUEST_V1','Wrong H18 metric registration request')
    d.require(sys.platform.startswith('linux'),'Registration requires actual Linux process-exit evidence')
    d.require(time.time()<d.u.DEADLINE,'Original H deadline expired')
    paths={k:pin(r[k]) for k in PINS};ctx=inspect_inputs(r,paths)
    before=ctx['before'];old=ctx['old'];a=ctx['owner'];prior=ctx['render_cfg'];q=ctx['qualification']
    sources=d.merge(ctx['render_registration']['source_bindings'],ctx['metric_assets']['source_bindings'],
        q['source_bindings'],d.bind((Path(__file__).absolute(),paths['metric_driver_module'],paths['metric_adapter_module'])))
    inputs=d.merge(ctx['render_registration']['input_bindings'],ctx['bindings'],ctx['metric_bindings'],
        q['input_bindings'],q['outputs'],d.bind((request_path,*paths.values())))
    d.verify(sources);d.verify(inputs)
    e,out,h=Path(r['execution_dir']),Path(r['metrics_out']),Path(prior['H_out'])
    for p in (e,out):
        d.require(p.is_absolute() and h in p.parents and not p.exists(), 'Fresh independent H execution/output required')
    d.require(e!=out and e not in out.parents and out not in e.parents,'Metric execution and output paths overlap')
    d.require(type(r['max_seconds']) is int and 0<r['max_seconds']<=86400,'Finite metric-stage time cap required')
    d.require(old['gpu_threads']==6 and set(old['gpu_affinity'])<=os.sched_getaffinity(0),
              'Original GPU CPU-affinity/threads unavailable')
    d.require(shutil.disk_usage(h).free>=1024**3,'Insufficient headroom for metric-only text outputs')
    cfg,owner=make_configs(r,ctx,paths)
    bound=d.merge(inputs,sources)
    for key in d.REQUIRED:
        if key=='visual_owner_config':continue
        d.require(bound.get(cfg[key])==d.sha(cfg[key]),'Metric execution dependency unbound: '+key)
    e.mkdir()
    try:
        cp,op,rp=e/'metrics_config.json',e/'owner_config.json',e/'execution_registration.json'
        gate=e/'predecessor_closure.json'
        d.save(gate,dict(status='H18_RENDER_NORMALLY_CLOSED_FOR_METRIC_SCORING',bindings=ctx['bindings'],
            budget=before,source_count=100,frame_count=5400,H_policy_snr_points=18,
            historical_graph_validation='SEALED_SOURCE_BINDINGS_EXACT_NO_REENUMERATION',
            new_packet_decodes=0,GPU_used=False,MAIN_complete=False))
        d.save(cp,cfg);d.save(op,owner);inputs=d.merge(inputs,d.bind((cp,op,gate)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=e.name,
            source_stage_scope=d.SCOPE,allowed_stage_ids=['development'],owner_config_sha256=d.sha(op),
            phase_limits=cfg['phase_limits'],budget_registration_sha256=d.sha(cfg['budget_registration']),
            scientific_protocol_sha256=d.sha(cfg['protocol']),source_bindings=sources,input_bindings=inputs,budget_before=before,
            source_count=100,frame_count=5400,H_policy_snr_points=18,MAIN_frames=0,noise_seeds=[6201,6202,6203],
            GPU_jobs=1,GPU_threads=6,GPU_affinity=old['gpu_affinity'],metric_batch_size=1,numerical_runtime=ctx['metric_flags'],
            new_packet_decodes=0,new_image_renders=0,policy_selection=False,development_used=True,holdout_started=False,
            H_full_delivery_claimed=False,P_complete=False,MAIN_started=False,C_started=False,
            scientific_protocol_modified=False,future_stage_automatic=False,created_unix=time.time(),
            shared_ledger_exclusion='No shared-ledger job until this metric owner and worker normally close.',
            metric_scope='Original H18 actual float-RGB only; original metrics and independent ConvNeXt; no P seed relabeling, policy selection or statistical aggregation.')
        a.validate_config(owner,reg,d.sha(op));d.verify(sources);d.verify(inputs)
        # Recheck normal predecessors and ledger immediately before sealing.
        refreshed=d.normal_render_closed(ctx['spec'])
        d.require(refreshed['bindings']==ctx['bindings'] and refreshed['before']==before,
                  'Normal predecessor evidence changed during registration')
        d.require(not Path(cfg['stop_file']).exists() and time.time()<d.u.DEADLINE,'STOP/deadline changed')
        current=a.budget_snapshot(cfg['ledger'],d.sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
        d.require(current==before,'Paid budget changed during metric registration')
        d.save(rp,reg)
        checked=d.load_registered(cp)  # Metadata/SHA only; no model, image extraction, PHY or process.
        d.require(checked['before']==before and checked['source_ids']==ctx['source_ids']
                  and checked['schedule']==ctx['schedule'],'Read-only metric preflight differs')
        d.require(a.budget_snapshot(cfg['ledger'],d.sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)==before,
                  'Paid budget changed after metric preflight')
        done=dict(status=REGISTERED,registration_sha256=d.sha(rp),owner_config_sha256=d.sha(op),
            outputs=d.bind((cp,op,rp,gate)),source_count=100,frame_count=5400,H_policy_snr_points=18,
            workers_started=False,GPU_used=False,new_packet_decodes=0,real_metric_calls=0,budget=before,
            policy_selection=False,MAIN_started=False,P_complete=False,future_stage_automatic=False,
            read_only_driver_preflight='PASS',required_launch_condition='Valid registration_completion with all hashes; no registration_failure; original render and CPU owners/workers normally exited; STOP absent and budget unchanged. One exclusive metric GPU job only.')
        d.save(e/'registration_completion.json',done);return done
    except BaseException:
        d.save(e/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True)
    print(register(parser.parse_args().request)['status'])
