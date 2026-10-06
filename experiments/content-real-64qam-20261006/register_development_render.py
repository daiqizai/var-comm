"""Seal one H18 development actual-RX GPU batch; never start it.

Normal CPU/source/selector closures, the immutable development100 population,
qualified visual implementation and unchanged packet budget are prerequisites.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import h_development_render_driver as d

REQUIRED_PINS=('cpu_driver_module','rx_adapter_module','visual_driver_module','prepared_qualification')
CPU_KEYS=('config','owner_config','registration','launch','completion')
REGISTERED='H_DEVELOPMENT_RENDER_REGISTERED_NOT_LAUNCHED'


def pin(value):
    p=Path(value['path']);d.require(p.is_absolute() and d.sha(p)==value['sha256'],'Pinned input missing/changed: '+str(p));return p


def qualification(q,paths):
    d.require(q['status']=='H_DEVELOPMENT_RENDER_CPU_QUALIFICATION_PASS' and q['GPU_used'] is False
        and q['new_packet_decodes']==0 and q['results'] and all(r['exit_code']==0 for r in q['results']),
        'Independent synthetic visual implementation qualification required')
    d.verify(q['source_bindings']);d.verify(q['input_bindings']);d.verify(q['outputs'])
    for p in paths:d.require(q['source_bindings'].get(str(p))==d.sha(p),'Qualification misses actual new entry: '+str(p))


def inspect_inputs(r,paths):
    spec={k:str(pin(r['cpu_batch'][k])) for k in CPU_KEYS}
    d.require(set(r['cpu_batch'])==set(CPU_KEYS),'Unexpected CPU predecessor schema')
    reg=d.read(spec['registration']);d.verify(reg['source_bindings']);d.verify(reg['input_bindings'])
    bound=d.merge(reg['source_bindings'],reg['input_bindings']);physical=d.read(spec['config'])
    d.require(physical['registration']==spec['registration'] and physical['owner_config']==spec['owner_config']
        and bound.get(spec['config'])==d.sha(spec['config']) and bound.get(spec['owner_config'])==d.sha(spec['owner_config']),
        'Actual CPU configuration is not sealed')
    d.require(reg['source_bindings'].get(str(paths['cpu_driver_module']))==d.sha(paths['cpu_driver_module'])
        and str(paths['cpu_driver_module'])==str(Path(physical['runtime_dir'])/'h_development_driver.py'),
        'Use the actual completed H18 CPU implementation')
    for k in ('owner_module','wait_module'):
        d.require(bound.get(physical[k])==d.sha(physical[k]),'Unbound normal predecessor verifier: '+k)
    owner=d.module(physical['owner_module'],'development_registration_owner');wait=d.module(physical['wait_module'],'development_registration_wait')
    cpu=d.module(paths['cpu_driver_module'],'development_registration_cpu');adapter=d.module(paths['rx_adapter_module'],'development_registration_rx')
    group=adapter.verify_cpu_closed(spec,cpu,owner,wait)
    source_cfg,source,source_cfg_path=d.original_source_config(group)
    old=group['closure']['owner'];before=group['before']
    d.assert_budget(before,before,group)
    for k in ('calibration_registration','numerical_reference','static_closure_module','source_driver_module'):
        d.require(bound.get(source_cfg[k])==d.sha(source_cfg[k]),'Original visual dependency unbound: '+k)
    numerical=d.read(source_cfg['numerical_reference']);cal=d.read(source_cfg['calibration_registration'])
    flags=d.visual_identity(source_cfg,source['done'],group['population'],numerical,cal)
    graph_path=Path(source['owner']['owner_out'])/'visual_source_closure.json'
    d.require(source['registration']['input_bindings'].get(str(graph_path))==d.sha(graph_path),'Original source graph receipt unbound')
    reference=d.read(graph_path)
    d.require(reference['status']=='EXACT_SOURCE_CLOSURE_MATCH'
        and reference['source_bindings']==source['done']['visual_source_bindings'],'Original actual source graph differs')
    helper=d.module(source_cfg['static_closure_module'],'development_registration_visual_graph')
    actual=helper.collect_bindings(source_cfg['root'],source_cfg['native_runtime'],source_cfg['var_source'],source_cfg['dino_source'],source_cfg['uep_runtime'])
    for p,s in reference['source_bindings'].items():d.require(actual.get(p)==s,'Previously executed visual source changed: '+p)
    for p in (Path(__file__).absolute(),paths['rx_adapter_module'],paths['visual_driver_module']):
        d.require(actual.get(str(p))==d.sha(p),'New visual source must be in the complete tracked graph before registration: '+str(p))
    python=source['owner']['stages'][0]['jobs'][0]['argv'][0]
    d.require(r['python']==python and Path(python).is_absolute() and Path(python).is_file(),
        'Use actual development-source GPU interpreter, without resolving its venv symlink')
    return dict(spec=spec,reg=reg,bound=bound,physical=physical,group=group,source_cfg=source_cfg,source=source,
        source_cfg_path=source_cfg_path,old=old,before=before,owner=owner,wait=wait,flags=flags,
        helper=helper,actual=actual,reference=reference,graph_path=graph_path)


def make_configs(r,ctx,paths):
    source=ctx['source_cfg'];e=Path(r['execution_dir']);rp=e/'execution_registration.json';cp=e/'render_config.json'
    cfg={k:copy.deepcopy(source[k]) for k in ('root','H_out','runtime_dir','uep_runtime','native_runtime','var_source','dino_source',
        'calibration_registration','numerical_reference','source_driver_module','static_closure_module','ledger',
        'budget_registration','phase_limits','stop_file','owner_module','wait_module','protocol')}
    cfg.update(schema='H_DEVELOPMENT_RX_CONFIG_V1',out=r['render_out'],registration=str(rp),visual_owner_config=str(e/'owner_config.json'),
        cpu_driver_module=str(paths['cpu_driver_module']),cpu_batch=copy.deepcopy(ctx['spec']),
        rx_adapter_module=str(paths['rx_adapter_module']),visual_driver_module=str(paths['visual_driver_module']),
        receiver_module=str(Path(source['runtime_dir'])/'h_payload_rx.py'),cache_module=str(Path(source['runtime_dir'])/'h_payload_render_driver.py'),
        visual_source_closure=str(e/'visual_source_closure.json'),max_seconds=r['max_seconds'],
        overall_deadline_unix=d.DEADLINE,max_archive_bytes=r['max_archive_bytes'])
    owner=copy.deepcopy(ctx['old']);owner.update(owner_out=str(e),registration=str(rp))
    owner['stages']=[dict(id='render',resource='gpu',requires=[],max_seconds=r['max_seconds'],jobs=[dict(id='development_render',
        argv=[r['python'],'-B',str(paths['visual_driver_module']),'--config',str(cp)],cwd=source['root'],out=r['render_out'],
        completion=str(Path(r['render_out'])/'completion.json'),accepted_statuses=[d.DONE],
        receipt_expect=dict(source_count=100,frame_count=5400,phase_frame_counts={'development':5400},H_policy_snr_points=18,MAIN_frames=0,
            images_scored=True,source_decode_complete=True,arithmetic_source_decode_complete=True,new_packet_decodes=0,
            policy_selection=False,development_used=True,holdout_used=False,MAIN_complete=False,overall_development_complete=False,
            H_full_delivery_claimed=False,unified_neural_metrics_run=False))])]
    return cfg,owner


def register(request_path):
    request_path=Path(request_path).absolute();r=d.read(request_path)
    d.require(r['schema']=='H_DEVELOPMENT_RENDER_REGISTRATION_REQUEST_V1','Wrong development render request')
    d.require(sys.platform.startswith('linux'),'Actual registration requires Linux process evidence')
    d.require(time.time()<d.DEADLINE,'Original H deadline expired')
    paths={k:pin(r[k]) for k in REQUIRED_PINS};q=d.read(paths['prepared_qualification'])
    qualification(q,(Path(__file__).absolute(),paths['rx_adapter_module'],paths['visual_driver_module']))
    ctx=inspect_inputs(r,paths);source=ctx['source_cfg'];old=ctx['old'];a=ctx['owner'];before=ctx['before']
    sources=d.merge(ctx['reg']['source_bindings'],q['source_bindings'],ctx['actual'],d.bind((Path(__file__).absolute(),)))
    inputs=d.merge(ctx['reg']['input_bindings'],ctx['group']['bindings'],ctx['group']['done']['outputs'],
        d.bind((request_path,*paths.values(),ctx['source_cfg_path'],ctx['graph_path'])))
    d.verify(sources);d.verify(inputs)
    e,out,h=Path(r['execution_dir']),Path(r['render_out']),Path(old['out'])
    for p in (e,out):d.require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent H execution/output required')
    d.require(e!=out and e not in out.parents and out not in e.parents,'Execution/science paths overlap')
    d.require(type(r['max_seconds'])is int and 0<r['max_seconds']<=86400
        and type(r['max_archive_bytes'])is int and 0<r['max_archive_bytes']<=d.MAX_ARCHIVE_BYTES,'Finite visual time/storage cap required')
    d.require(old['gpu_threads']==6 and set(old['gpu_affinity'])<=os.sched_getaffinity(0),'Original GPU affinity/threads unavailable')
    d.require(shutil.disk_usage(h).free>=r['max_archive_bytes']+d.STORAGE_RESERVE_BYTES,'Insufficient float RGB storage headroom')
    cfg,owner=make_configs(r,ctx,paths)
    for k in d.REQUIRED:
        if k in ('visual_owner_config','visual_source_closure'):continue
        d.require(d.merge(inputs,sources).get(cfg[k])==d.sha(cfg[k]),'Visual execution dependency unbound: '+k)
    e.mkdir()
    try:
        cp,op,rp=e/'render_config.json',e/'owner_config.json',e/'execution_registration.json'
        gate,graph=e/'predecessor_closure.json',e/'visual_source_closure.json'
        d.save(gate,dict(status='H18_CPU_SOURCE_AND_SELECTOR_NORMALLY_CLOSED',bindings=ctx['group']['bindings'],
            budget=before,source_count=100,frame_count=5400,new_packet_decodes=0,GPU_used=False,MAIN_complete=False))
        d.save(graph,dict(status='EXACT_SOURCE_CLOSURE_MATCH',source_bindings=ctx['actual'],inputs=ctx['reference']['inputs'],
            previous_graph=dict(path=str(ctx['graph_path']),sha256=d.sha(ctx['graph_path'])),
            newly_bound_sources={p:s for p,s in ctx['actual'].items() if p not in ctx['reference']['source_bindings']},
            original_registered_files_changed=False,source_enumeration_only=True,GPU_used=False,new_packet_decodes=0))
        d.save(cp,cfg);d.save(op,owner);inputs=d.merge(inputs,d.bind((cp,op,gate,graph)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=e.name,source_stage_scope=d.SCOPE,allowed_stage_ids=['render'],
            owner_config_sha256=d.sha(op),phase_limits=cfg['phase_limits'],budget_registration_sha256=d.sha(cfg['budget_registration']),
            scientific_protocol_sha256=d.sha(cfg['protocol']),source_bindings=sources,input_bindings=inputs,budget_before=before,
            source_count=100,frame_count=5400,H_policy_snr_points=18,MAIN_frames=0,noise_seeds=[6201,6202,6203],
            GPU_jobs=1,GPU_threads=6,GPU_affinity=old['gpu_affinity'],max_archive_bytes=r['max_archive_bytes'],new_packet_decodes=0,
            policy_selection=False,development_used=True,holdout_started=False,H_full_delivery_claimed=False,C_started=False,
            MAIN_started=False,scientific_protocol_modified=False,future_stage_automatic=False,created_unix=time.time(),
            shared_ledger_exclusion='No shared-ledger job until this visual owner and worker normally close.',
            receiver_rule='Actual paid header/payload only; target pixels enter separate MSE/PSNR scoring; canonical source errors follow original protocol.')
        a.validate_config(owner,reg,d.sha(op));d.verify(sources);d.verify(inputs)
        d.require(ctx['helper'].collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])==ctx['actual'],
            'Complete tracked source graph changed while sealing')
        d.require(not Path(cfg['stop_file']).exists() and time.time()<d.DEADLINE,'STOP/deadline changed')
        current=a.budget_snapshot(cfg['ledger'],d.sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
        d.assert_budget(current,before,ctx['group']);d.save(rp,reg)
        checked=d.load_registered(cp)  # No model, pixel extraction, decoder or launch.
        d.require(checked['before']==before and len(checked['source_ids'])==100 and len(checked['schedule'])==18,
            'Read-only visual preflight differs')
        d.assert_budget(a.budget_snapshot(cfg['ledger'],d.sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True),
                        before,ctx['group'])
        done=dict(status=REGISTERED,registration_sha256=d.sha(rp),owner_config_sha256=d.sha(op),outputs=d.bind((cp,op,rp,gate,graph)),
            source_count=100,frame_count=5400,workers_started=False,GPU_used=False,new_packet_decodes=0,budget=before,
            policy_selection=False,MAIN_started=False,future_stage_automatic=False,read_only_driver_preflight='PASS',
            required_launch_condition='Successful registration_completion outputs, no registration_failure, normal CPU/source/selector exits, STOP absent and budget unchanged. One exclusive GPU worker; no automatic next stage.')
        d.save(e/'registration_completion.json',done);return done
    except BaseException:
        d.save(e/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);a=p.parse_args()
    done=register(a.request);print(done['status'])


if __name__=='__main__':main()
