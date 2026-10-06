"""Prepare one immutable H18 development CPU batch; never launch it.

Missing real finalized policies, normal source100 closure or source qualification
is a hard stop. The four MAIN points remain pending and their2400 calls reserved.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import h_development_driver as d


def save(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def pin(value):
    p=Path(value['path']);d.require(p.is_absolute() and d.sha(p)==value['sha256'],'Pinned input absent/changed: '+str(p));return p


def make_contract(ctx):
    core=ctx['core'];c=dict(status='H_DEVELOPMENT_CPU_ENGINEERING_SEALED',engineering_choices=core.ENGINEERING_CHOICES,
        public_schedule=ctx['rows'],engineering_choices_canonical_sha256=d.digest(core.ENGINEERING_CHOICES),
        public_schedule_canonical_sha256=d.digest(ctx['rows']),MAIN_reserved_packet_calls=2400,
        MAIN_executable=False,scientific_protocol_modified=False)
    for k,v in ctx['args'].items():c[k+'_canonical_sha256']=d.digest(v)
    for k,p in dict(core=core.__file__,initial_core=core.initial.__file__,calibrated_core=core.calibrated.__file__,
                   fixed_control_rules=core.selection_rules.__file__,phy=core.phy.__file__).items():
        c[k+'_source_sha256']=d.sha(p)
    core.prepare(**ctx['args'],contract=dict(c,execution_registration_sha256='0'*64))
    return c


def make_configs(request,draft,oldowner,rows):
    execution,out=Path(request['execution_dir']),Path(request['payload_out'])
    cp,rp,op=execution/'payload_config.json',execution/'execution_registration.json',execution/'owner_config.json'
    cfg=copy.deepcopy(draft);cfg.update(schema='H_DEVELOPMENT_CPU_CONFIG_V1',out=str(out),registration=str(rp),owner_config=str(op),
        engineering_contract=str(execution/'engineering_contract.json'),workers=2,
        cpu_affinities=oldowner['cpu_affinities'],phase_limits=d.PHASES,max_worker_seconds=request['max_worker_seconds'],
        overall_deadline_unix=d.DEADLINE)
    owner=copy.deepcopy(oldowner);owner.update(owner_out=str(execution),registration=str(rp))
    ready=sum(x['status']=='FROZEN_POLICY_READY' for x in rows);driver=str(HERE/'h_development_driver.py')
    common=dict(workers=2,declared_policy_snr_points=18,executable_policy_snr_points=ready,maximum_H_packet_calls=10800,
        MAIN_reserved_packet_calls=2400,images_scored=False,source_decode_complete=False,arithmetic_source_decode_complete=False,
        GPU_used=False,development_used=True,holdout_used=False,policy_selection=False,MAIN_complete=False,
        overall_development_complete=False,H_full_delivery_claimed=False,visual_stage_started=False)
    jobs=[]
    for i in (0,1):
        folder=out/f'worker_{i}'
        jobs.append(dict(id=f'h_dev_worker_{i}',argv=[request['python'],'-B',driver,'--config',str(cp),'--stage','worker','--worker-index',str(i)],
            cwd=cfg['root'],out=str(folder),completion=str(folder/'completion.json'),accepted_statuses=['H_DEVELOPMENT_CPU_WORKER_COMPLETE'],
            receipt_expect=dict(common,worker_index=i,source_count=50,frame_count=ready*150)))
    owner['stages']=[dict(id='development',resource='cpu',requires=[],max_seconds=request['max_worker_seconds'],jobs=jobs),
        dict(id='report',resource='cpu',requires=['development'],max_seconds=request['merge_seconds'],jobs=[dict(id='h_dev_merge',
            argv=[request['python'],'-B',driver,'--config',str(cp),'--stage','merge'],cwd=cfg['root'],out=str(out),
            completion=str(out/'completion.json'),accepted_statuses=['H_DEVELOPMENT_CPU_RECEIVE_COMPLETE'],
            receipt_expect=dict(common,worker_index=None,source_count=100,frame_count=ready*300))])]
    return cfg,owner


def qualification(q,sourcepaths):
    d.require(q['status']=='H_DEVELOPMENT_CPU_QUALIFICATION_PASS' and q['GPU_used'] is False
        and q['new_packet_decodes']==0 and q['results'] and all(x['exit_code']==0 for x in q['results']),
        'Prepared CPU implementation qualification missing/failed')
    d.verify(q['source_bindings'])
    for p in sourcepaths:d.require(q['source_bindings'].get(str(p))==d.sha(p),'Qualification does not bind current new code: '+str(p))


def register(request_path):
    request_path=Path(request_path).absolute();r=d.read(request_path)
    d.require(r['schema']=='H_DEVELOPMENT_CPU_REGISTRATION_REQUEST_V1','Wrong development request')
    d.require(sys.platform.startswith('linux'),'Actual process-exit registration requires Linux')
    d.require(time.time()<d.DEADLINE,'Original H deadline expired')
    draftpath=pin(r['draft_config']);qpath=pin(r['prepared_qualification']);draft=d.read(draftpath);q=d.read(qpath)
    ownfiles=[HERE/name for name in ('h_development_cpu.py','h_development_driver.py','register_h_development_cpu.py')]
    qualification(q,ownfiles)
    # Input pins are read-only and cannot be replaced by plausible completion names.
    pins={str(pin(x)):x['sha256'] for x in r['additional_pins']}
    pins=d.merged(pins,d.bind((request_path,draftpath,qpath)))
    for k in d.REQUIRED:d.require(pins.get(draft[k])==d.sha(draft[k]),'Required request pin missing: '+k)
    sources=d.merged(q['source_bindings'],d.bind(ownfiles));inputs=dict(pins)
    d.require(set(draft['prerequisites'])=={'selection','source'},'Actual selection/source prerequisites missing')
    for spec in draft['prerequisites'].values():
        for p in spec.values():d.require(pins.get(p)==d.sha(p),'Actual prerequisite file not pinned')
        rr=d.read(spec['registration']);d.verify(rr['source_bindings']);d.verify(rr['input_bindings'])
        sources=d.merged(sources,rr['source_bindings']);inputs=d.merged(inputs,rr['input_bindings'])
    ctx=d.prepare_context(draft,d.merged(sources,inputs));api=ctx['api'];old=ctx['closures']['source']['owner']
    d.require(draft['H_out']==old['out'] and draft['root']==old['root'] and draft['cpu_affinities']==old['cpu_affinities']
        and draft['stop_file']==str(Path(old['out'])/'STOP'),'New draft differs from original H workspace/resources')
    for c in ctx['closures'].values():
        inputs=d.merged(inputs,c['bindings'],c['done']['outputs']);sources=d.merged(sources,c['registration']['source_bindings'])
    for value in (ctx['source'],ctx['assets']):inputs=d.merged(inputs,value['outputs'],value['input_bindings']);sources=d.merged(sources,value['source_bindings'])
    current=api.budget_snapshot(draft['ledger'],d.sha(draft['budget_registration']),d.PHASES,quiescent=True)
    d.budget_admission(current,current)
    for c in ctx['closures'].values():
        d.require(c['done']['budget_before']==c['done']['budget_after']==c['owner_done']['budget']==current,
                  'H budget changed after policy/source completion')
    execution,out,h=Path(r['execution_dir']),Path(r['payload_out']),Path(old['out'])
    for p in (execution,out):d.require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent H output required')
    d.require(execution!=out and execution not in out.parents and out not in execution.parents,'Execution/science paths overlap')
    d.require(Path(r['python']).is_absolute() and Path(r['python']).is_file()
        and type(r['max_worker_seconds'])is int and 0<r['max_worker_seconds']<=21600
        and type(r['merge_seconds'])is int and 0<r['merge_seconds']<=r['max_worker_seconds'],'Invalid interpreter/finite work cap')
    d.require(set(sum(old['cpu_affinities'],[]))<=os.sched_getaffinity(0),'Registered CPU cores unavailable')
    contract=make_contract(ctx);cfg,owner=make_configs(r,draft,old,ctx['rows']);execution.mkdir()
    try:
        cp,op,ep,rp=(execution/n for n in ('payload_config.json','owner_config.json','engineering_contract.json','execution_registration.json'))
        gate=execution/'predecessor_closure.json';save(gate,dict(status='DEVELOPMENT100_SOURCE_AND_FULL1000_SELECTOR_NORMALLY_CLOSED',
            closures=ctx['closures'],budget=current,MAIN_complete=False,development_results_read=False))
        save(cp,cfg);save(op,owner);save(ep,contract);inputs=d.merged(inputs,d.bind((cp,op,ep,gate)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,
            source_stage_scope='H_DEVELOPMENT_18_POINTS_CPU_ONLY',allowed_stage_ids=['development','report'],
            owner_config_sha256=d.sha(op),phase_limits=d.PHASES,budget_before=current,
            budget_registration_sha256=d.sha(cfg['budget_registration']),source_bindings=sources,input_bindings=inputs,
            source_count=100,declared_H_policy_snr_points=18,MAIN_reserved_policy_snr_points=4,
            H_maximum_frames=5400,H_maximum_packet_calls=10800,MAIN_reserved_packet_calls=2400,
            noise_seeds=[6201,6202,6203],snrs_db=[13,19],CPU_workers=2,threads_per_worker=2,nice=15,GPU_jobs=0,
            noise_stage='development',full_public_catalogue=True,policy_selection=False,MAIN_started=False,
            development_results_used_for_selection=False,holdout_started=False,H_full_delivery_claimed=False,
            scientific_protocol_modified=False,future_stage_automatic=False,created_unix=time.time())
        api.validate_config(owner,reg,d.sha(op));d.verify(sources);d.verify(inputs)
        d.require(not Path(cfg['stop_file']).exists() and time.time()<d.DEADLINE,'STOP/deadline changed')
        d.require(api.budget_snapshot(cfg['ledger'],d.sha(cfg['budget_registration']),d.PHASES,quiescent=True)==current,'Budget changed while sealing')
        save(rp,reg);d.load_registered(cp)  # Complete read-only preflight; no backend constructed.
        complete=dict(status='H_DEVELOPMENT_CPU_REGISTERED_NOT_LAUNCHED',registration_sha256=d.sha(rp),owner_config_sha256=d.sha(op),
            outputs=d.bind((cp,op,ep,rp,gate)),source_count=100,H_policy_snr_points=18,H_maximum_packet_calls=10800,
            MAIN_reserved_packet_calls=2400,MAIN_started=False,workers_started=False,new_packet_decodes=0,GPU_used=False,
            future_stage_automatic=False,required_launch_condition='This complete receipt and all outputs must verify, registration_failure absent, predecessors normally reaped and budget/STOP unchanged. Registration alone never launches.')
        save(execution/'registration_completion.json',complete);return complete
    except BaseException as error:
        save(execution/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);a=p.parse_args()
    result=register(a.request);print(json.dumps({k:result[k] for k in ('status','registration_sha256')}))


if __name__=='__main__':main()
