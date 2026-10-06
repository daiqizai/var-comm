"""Register the bounded P600 received-latent replay after H18 metrics close.

This entry loads no model and starts no process. The caller supplies exact
metadata/SHA and a CPU qualification; the driver validates normal predecessors.
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

import p600_received_replay_driver as d

REGISTERED = 'P600_RECEIVED_REPLAY_REGISTERED_NOT_LAUNCHED'
QUALIFIED = 'P600_CPU_INTERFACE_QUALIFICATION_PASS'


def qualification(value, sources, inputs, python):
    d.require(value['status'] == QUALIFIED and value['GPU_used'] is False
              and value['models_constructed'] is False and value['new_packet_decodes'] == 0
              and value['new_noise_draws'] == 0 and value['python'] == python
              and value['results'] and all(r['exit_code'] == 0 and r['test_count'] > 0 for r in value['results']),
              'Actual CPU interface qualification required')
    for key in ('source_bindings','input_bindings','outputs'): d.verify(value[key])
    for p,s in value['source_bindings'].items(): d.require(sources.get(p) == s, 'CPU-qualified source missing from new graph')
    for p,s in d.merge(value['input_bindings'],value['outputs']).items():
        d.require(inputs.get(p) == s, 'Qualification evidence missing from new graph')
    for p in (str(Path(__file__).absolute()),str(Path(d.__file__).absolute())):
        d.require(value['source_bindings'].get(p) == d.sha(p), 'Registrar/driver not covered by qualification')


def validate_request(r):
    d.require(r['schema'] == 'P600_RECEIVED_REPLAY_REGISTRATION_REQUEST_V1', 'Wrong P600 request')
    cfg = copy.deepcopy(r['config']); e = Path(r['execution_dir']); out = Path(cfg['out']); h = Path(cfg['H_out'])
    d.require(e.is_absolute() and out.is_absolute() and h in e.parents and h in out.parents
              and e != out and e not in out.parents and out not in e.parents,
              'Fresh independent P execution/output directories required')
    d.require(cfg['registration'] == str(e/'execution_registration.json')
              and cfg['visual_owner_config'] == str(e/'owner_config.json')
              and cfg['schema'] == 'P600_RECEIVED_REPLAY_CONFIG_V1', 'Planned own paths differ')
    d.require(type(cfg['max_seconds']) is int and 0 < cfg['max_seconds'] <= 7200
              and cfg['overall_deadline_unix'] == d.DEADLINE
              and type(cfg['max_archive_bytes']) is int
              and d.RGB_BYTES < cfg['max_archive_bytes'] <= d.MAX_ARCHIVE_BYTES,
              'Bounded original P replay resources required')
    d.require(not e.exists() and not out.exists(), 'Prior P replay attempt preserved; no retry')
    return cfg,e,out


def make_owner(cfg, ctx, execution):
    old = ctx['prior']['closed']['owner']; owner = copy.deepcopy(old)
    d.require(old['gpu_threads'] == 6 and len(old['stages']) == 1
              and old['stages'][0]['resource'] == 'gpu' and len(old['stages'][0]['jobs']) == 1,
              'Original single metric GPU owner required')
    python = old['stages'][0]['jobs'][0]['argv'][0]
    owner.update(owner_out=str(execution),registration=cfg['registration'])
    expected = dict(source_count=100,frame_count=600,P_policy_snr_points=2,N=1024,
        snrs_db=[13,19],noise_seeds=[2001,2002,2003],RGB_exact_parity=True,legacy_scalars_preserved=True,
        ConvNeXt_scored=True,source_predictions=100,reconstruction_predictions=600,Dc_batch_size=3,
        classifier_batch_size=1,budget_writes=0,new_packet_decodes=0,new_noise_draws=0,
        GPU_used=True,development_used=True,holdout_used=False,policy_selection=False,training_updates=0,
        statistical_aggregation_run=False,MAIN_complete=False,overall_development_complete=False,
        H_full_delivery_claimed=False,online_latency_measured=False)
    owner['stages'] = [dict(id='development',resource='gpu',requires=[],max_seconds=cfg['max_seconds'],jobs=[
        dict(id='p600_received_replay',argv=[python,'-B',cfg['replay_driver_module'],'--config',str(execution/'replay_config.json')],
             cwd=cfg['root'],out=cfg['out'],completion=str(Path(cfg['out'])/'completion.json'),
             accepted_statuses=[d.DONE],receipt_expect=expected)])]
    return owner,python


def budget_guard(ctx):
    c = ctx['cfg']; before = ctx['before']
    now = ctx['owner'].budget_snapshot(c['ledger'],d.sha(c['budget_registration']),c['phase_limits'],quiescent=True)
    d.require(now == before and now['charged'] == 164760 and now['phase_charged']['development'] == 10800
              and now['development_remaining'] == 2400 and now['failed'] == now['unresolved'] == 0,
              'Original quiescent H budget must remain unchanged')
    d.require(not Path(c['stop_file']).exists() and time.time() < d.DEADLINE, 'STOP/deadline blocks P replay')


def register(request_path):
    request_path = Path(request_path).absolute(); r = d.read(request_path)
    d.require(sys.platform.startswith('linux'), 'Actual Linux predecessor/process evidence required')
    cfg,e,out = validate_request(r)
    sources = d.merge(r['source_bindings'],d.bind((Path(__file__).absolute(),Path(d.__file__).absolute())))
    inputs = d.merge(r['input_bindings'],d.bind((request_path,)))
    # Qualification pins are supplied, never discovered from a mutable latest pointer.
    qp = r['prepared_qualification']
    d.require(inputs.get(qp) == d.sha(qp), 'CPU qualification completion must be explicitly pinned')
    q = d.read(qp); bound = d.merge(sources,inputs)
    ctx = d.inspect_inputs(cfg,bound)
    owner,python = make_owner(cfg,ctx,e)
    qualification(q,sources,inputs,python)
    d.require(q['source_bindings'].get(cfg['core_module']) == d.sha(cfg['core_module']),
              'Actual P core not CPU-qualified')
    d.require(set(owner['gpu_affinity']) <= os.sched_getaffinity(0)
              and shutil.disk_usage(cfg['H_out']).free >= cfg['max_archive_bytes']+d.STORAGE_RESERVE_BYTES,
              'Original GPU affinity or storage headroom unavailable')
    budget_guard(ctx)
    e.mkdir()
    try:
        cp,op,rp = e/'replay_config.json',e/'owner_config.json',e/'execution_registration.json'
        gate = e/'predecessor_closure.json'
        d.save(gate,dict(status='H18_METRICS_NORMALLY_CLOSED_FOR_P600_REPLAY',
            bindings=ctx['prior']['bindings'],budget=ctx['before'],metric_sources=100,metric_frames=5400,
            new_packet_decodes=0,new_noise_draws=0,GPU_used=False,
            closure='Original normal owner/worker receipt and frozen graph validated; own P worker has not started.'))
        d.save(cp,cfg); d.save(op,owner)
        inputs = d.merge(inputs,d.bind((cp,op,gate)))
        reg = dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=e.name,
            source_stage_scope=d.SCOPE,allowed_stage_ids=['development'],owner_config_sha256=d.sha(op),
            phase_limits=cfg['phase_limits'],budget_registration_sha256=d.sha(cfg['budget_registration']),
            scientific_protocol_sha256=d.sha(cfg['protocol']),source_bindings=sources,input_bindings=inputs,
            budget_before=ctx['before'],source_count=100,frame_count=600,P_policy_snr_points=2,
            snrs_db=[13,19],noise_seeds=[2001,2002,2003],GPU_jobs=1,GPU_threads=6,GPU_affinity=owner['gpu_affinity'],
            Dc_batch_size=3,classifier_batch_size=1,new_packet_decodes=0,new_noise_draws=0,budget_writes=0,
            training_updates=0,policy_selection=False,holdout_started=False,scientific_protocol_modified=False,
            MAIN_started=False,C_started=False,H_full_delivery_claimed=False,future_stage_automatic=False,
            max_archive_bytes=cfg['max_archive_bytes'],created_unix=time.time(),
            scope='Replay original received Z only, exact old RGB parity, retain legacy scalar metrics and original P noises; independent ConvNeXt added.',
            shared_ledger_exclusion='No shared-ledger job until this P replay owner and worker normally close.')
        ctx['owner'].validate_config(owner,reg,d.sha(op))
        # Recheck file content at the seal boundary. The already-closed process
        # proof is reused; no second recursive historical-context construction.
        d.verify(sources);d.verify(inputs);budget_guard(ctx)
        d.save(rp,reg)
        result = dict(status=REGISTERED,registration_sha256=d.sha(rp),owner_config_sha256=d.sha(op),
            outputs=d.bind((cp,op,rp,gate)),source_count=100,frame_count=600,workers_started=False,
            GPU_used=False,models_constructed=False,new_packet_decodes=0,new_noise_draws=0,
            real_metric_calls=0,budget=ctx['before'],future_stage_automatic=False,
            metadata_preflight='PASS',required_launch_condition='All sealed pins and original normal H18 metric receipts; STOP absent, unchanged budget, original single-GPU admission, fresh P launch only.')
        d.save(e/'registration_completion.json',result); return result
    except BaseException:
        d.save(e/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True)
    print(register(parser.parse_args().request)['status'])
