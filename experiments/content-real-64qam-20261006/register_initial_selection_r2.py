"""Independently register/run H initial200 selection after certified exit race.

No decoder, model construction, GPU, subprocess launch, or new policy search.
Registration requires a separately registered truthful closeout certificate.
The original six partial candidates are references, not newly selected results.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

DEADLINE = 1791564605.9549868
PHASES = {'qualification':2000,'coarse':49152,'refine':12288,'initial_true200':20000,
          'whole_calibration':48000,'partial_calibration':36000,'development':13200,'engineering_reserve':19360}
EXPECTED_CHARGES = dict.fromkeys(PHASES, 0)
EXPECTED_CHARGES.update(qualification=1608,coarse=49152,initial_true200=19200)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
def bind(paths): return {str(p):sha(p) for p in paths}
def verify(bindings):
    for p,s in bindings.items(): require(Path(p).is_absolute() and sha(p)==s,'Changed immutable binding: '+p)
def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting binding: '+p);result[p]=s
    return result
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def pin(record):
    p=Path(record['path']);require(p.is_absolute() and sha(p)==record['sha256'],'Pinned input changed: '+str(p));return p


def check_budget(snapshot):
    require(snapshot.get('created') is True and snapshot['charged']==69960
            and snapshot['phase_charged']==EXPECTED_CHARGES and snapshot['unresolved']==0
            and snapshot.get('failed',0)==0,'Original 69960-packet quiescent ledger differs')


def require_fresh_output(path):
    path=Path(path)
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
            'Existing selection output: preserve, do not retry')


def verify_render_closed(owner,prior,owner_path,reg_path,launch_path,render_path,state_reader,closeout_paths):
    """Admit a truthful registered closeout, never invent old owner success."""
    cfg,reg,launch=read(owner_path),read(reg_path),read(launch_path)
    owner.validate_config(cfg,reg,sha(owner_path))
    require(cfg['registration']==str(reg_path) and reg['allowed_stage_ids']==['render']
            and len(cfg['stages'])==1,'Wrong predecessor render scope')
    stage=cfg['stages'][0]
    require(stage['id']=='render' and stage['resource']=='gpu' and len(stage['jobs'])==1,
            'Expected one exclusive render worker')
    job=stage['jobs'][0];argv=job['argv']
    require(argv.count('--config')==1 and argv[argv.index('--config')+1]==str(render_path),
            'Predecessor render configuration differs')
    ident=launch['identity'];expected=[launch['argv'][0],'-B',str(Path(owner.__file__).resolve()),'--config',str(owner_path)]
    require(ident['argv']==launch['argv']==expected and launch['registration_sha256']==sha(reg_path)
            and launch['owner_config_sha256']==sha(owner_path),'Render owner launch identity differs')
    identity_path=Path(cfg['owner_out'])/'owner_identity.json';identity=read(identity_path)
    require(all(identity[k]==ident[k] for k in ('pid','uid','start_ticks','argv'))
            and identity['registration_sha256']==sha(reg_path) and identity['config_sha256']==sha(owner_path),
            'Recorded render owner identity differs from actual launch')
    require(prior.exited(ident,state_reader),'Render owner still live or unreaped')
    base=Path(cfg['owner_out']);require(not (base/'completion.json').exists(),
            'The failed old owner must not acquire a fabricated success receipt')
    cpreg=closeout_paths['closeout_registration'];certificate=closeout_paths['closeout_certificate']
    release_path=closeout_paths['stop_release'];completed_path=closeout_paths['closeout_completion']
    creg,cert,release,completed=read(cpreg),read(certificate),read(release_path),read(completed_path)
    require(creg['status']=='H_RENDER_CLOSEOUT_READ_ONLY_REGISTERED','Independent closeout is not registered')
    for field in ('source_bindings','input_bindings'):verify(creg[field])
    require(completed['status']=='H_RENDER_CLOSEOUT_COMPLETE_STOP_RELEASED'
            and completed['closeout_registration_sha256']==sha(cpreg)
            and completed['original_owner_success'] is False and completed['scientific_outputs_complete'] is True
            and completed['original_owner_completion_written'] is False and completed['new_packet_decodes']==0
            and completed['new_visual_inference']==0 and completed['GPU_used'] is False,
            'Closeout/release sequence is incomplete')
    verify(completed['outputs'])
    require(all(completed['outputs'].get(str(p))==sha(p) for p in (certificate,release_path)),
            'Closeout terminal receipt does not bind certificate and release')
    require(cert['status']=='H_RENDER_OUTPUTS_AND_EXIT_CERTIFIED_AFTER_OWNER_RACE'
            and cert['original_owner_success'] is False and cert['scientific_outputs_complete'] is True
            and cert['original_registration_sha256']==sha(reg_path)
            and cert['closeout_registration_sha256']==sha(cpreg) and cert['source_count']==200
            and cert['frame_count']==9600 and cert['new_packet_decodes']==0 and cert['GPU_used'] is False,
            'Closeout certificate changed original science/supervisor meaning')
    verify(cert['bindings']);verify(cert['outputs']);verify(cert['source_bindings']);check_budget(cert['budget'])
    require(cert['bindings'].items()<=creg['input_bindings'].items()
            and cert['source_bindings'].items()<=creg['source_bindings'].items()
            and cert['budget']==creg['budget_before']==completed['budget']
            and completed['original_registration_sha256']==sha(reg_path),
            'Closeout certificate changed registered evidence/budget')
    for field,path in (('original_owner_config',owner_path),('original_launch',launch_path),
                       ('render_completion',Path(job['completion'])),('original_failure',base/'failure.json')):
        require(cert[field]=={'path':str(path),'sha256':sha(path)},'Closeout certificate links another predecessor: '+field)
    require(owner.same_identity(cert['owner_identity'],ident),'Certified owner differs')
    failure=base/'failure.json';require(creg['input_bindings'].get(str(failure))==sha(failure),
            'Original failed owner evidence is not bound before closeout')
    require(not (base/'STOP').exists(),'Unexpected owner STOP blocks selection')
    global_stop=Path(cfg['out'])/'STOP';render_stop=Path(job['out'])/'STOP'
    require(not global_stop.exists() and not render_stop.exists(),'New/current global or renderer STOP blocks selection')
    require(release['status']=='H_RENDER_CLOSEOUT_EXACT_STOP_ARCHIVED'
            and release['certificate_sha256']==sha(certificate) and release['closeout_registration_sha256']==sha(cpreg)
            and release['original_path']==str(global_stop) and release['released'] is True
            and release['original_owner_success'] is False
            and release['old_owner_and_worker_exited'] is True and release['budget_unchanged'] is True
            and creg['expected_stop']=={'path':release['original_path'],'sha256':release['sha256']},
            'Global STOP was not released by this exact certificate')
    markers=release['released_markers'];expected_stops=creg['expected_stops']
    require(len(markers)==len(expected_stops)==2
            and {r['original_path'] for r in markers}=={str(global_stop),str(render_stop)}
            and {r['path'] for r in expected_stops}=={str(global_stop),str(render_stop)},
            'Both exact registered STOP markers must be released')
    expected_map={r['path']:r['sha256'] for r in expected_stops}
    require(len({r['preserved_path'] for r in markers})==2,'STOP archives collide')
    for marker in markers:
        preserved=Path(marker['preserved_path'])
        require(preserved.is_absolute() and sha(preserved)==marker['sha256']==expected_map[marker['original_path']]
                and completed['outputs'].get(str(preserved))==sha(preserved),'Released STOP archive is unsealed')
    global_marker=next(r for r in markers if r['original_path']==str(global_stop))
    require(all(global_marker[k]==release[k] for k in ('original_path','preserved_path','sha256')),
            'Global STOP compatibility fields differ')
    close_request=pin(creg['request']);require(creg['input_bindings'].get(str(close_request))==sha(close_request),
            'Closeout request unbound')
    supplement_path=pin(read(close_request)['renderer_stop_evidence']);supplement=read(supplement_path)
    require(creg['input_bindings'].get(str(supplement_path))==sha(supplement_path)
            and supplement['status']=='RENDER_LOCAL_STOP_EVIDENCE_PRESERVED'
            and supplement['original_path']==str(render_stop) and supplement['sha256']==expected_map[str(render_stop)]
            and supplement['owner_failure_sha256']==sha(failure)
            and creg['input_bindings'].get(supplement['preserved_path'])==supplement['sha256'],
            'Supplemental renderer STOP evidence is absent or changed')
    worker_dir=base/'stages/render/workers'/job['id']
    exit_path=worker_dir/'exit_receipt.json';worker_launch=worker_dir/'launch.json';log=worker_dir/'worker.log'
    event,wlaunch=read(exit_path),read(worker_launch);child=event['identity']
    require(cert['worker_exit_receipt']=={'path':str(exit_path),'sha256':sha(exit_path)}
            and owner.same_identity(cert['worker_identity'],child) and event['exit_code']==0
            and event['registration_sha256']==sha(reg_path) and event['completion']==job['completion']
            and event['completion_sha256']==sha(job['completion']) and event['closed_log_sha256']==sha(log),
            'Real worker exit0 or sealed completion/log is absent')
    require(wlaunch['identity']==child and child['argv']==job['argv'] and child['uid']==ident['uid']
            and wlaunch['registration_sha256']==sha(reg_path) and prior.exited(child,state_reader),
            'Real worker launch/exit is unproven or process still present')
    require(not (Path(job['out'])/'failure.json').exists(),'A new receiver failure cannot be normalized as owner race')
    required=bind((owner_path,reg_path,launch_path,render_path,identity_path,failure,exit_path,worker_launch,log,Path(job['completion'])))
    require(all(cert['bindings'].get(p)==s for p,s in required.items()),'Certified evidence is incomplete')
    done_path=Path(job['completion']);done=read(done_path);rcfg=read(render_path)
    require(done_path==Path(rcfg['out'])/'completion.json'
            and rcfg['registration']==str(reg_path) and rcfg['render_owner_config']==str(owner_path)
            and cert['bindings'].get(str(done_path))==sha(done_path),'Actual render completion is unbound')
    owner.receipt(done_path,job['accepted_statuses'],sha(reg_path),job.get('receipt_expect'),job['out'])
    require(done['status']=='H_INITIAL_TRUE200_RX_COMPLETE' and done['registration_sha256']==sha(reg_path)
            and done['config_sha256']==sha(render_path) and done['source_count']==200 and done['frame_count']==9600
            and done['images_scored'] is True and done['source_decode_complete'] is True
            and done['new_packet_decodes']==0 and done['budget_unchanged'] is True
            and done['policy_selection'] is False and done['development_used'] is False
            and done['holdout_used'] is False,'Incomplete or wrong-scope render result')
    for key in ('outputs','input_bindings','source_bindings','predecessor_closure_bindings','visual_launch_bindings'):
        verify(done[key])
    require(done['cpu_completion_sha256']==sha(rcfg['cpu_completion'])
            and done['shortlist_sha256']==sha(rcfg['shortlist']),'Predecessor CPU/shortlist changed')
    require(all(cert['outputs'].get(p)==s for p,s in done['outputs'].items()),'Closeout omitted scientific outputs')
    require(not (certificate.parent/'failure.json').exists(),'Closeout attempt failed')
    return dict(status='H_RENDER_SCIENCE_CERTIFIED_AFTER_FAILED_OWNER',original_owner_success=False,owner_identity=ident,
        child_identities=[child],render_completion=str(done_path),closeout_certificate_sha256=sha(certificate),
        bindings=merge(cert['bindings'],cert['outputs'],completed['outputs'],creg['source_bindings'],creg['input_bindings'],
                       required,bind(closeout_paths.values())))


def audit_source_receipts(render_cfg,done,ids,rows):
    """Check all200 source receipts against the single sealed9600-row table."""
    require(len(ids)==len(set(ids))==200 and len(rows)==9600,'Original200/9600 scope differs')
    grouped={i:[] for i in range(200)}
    for row in rows:
        i=row['source_index'];require(type(i) is int and i in grouped and row['source_id']==ids[i],'Source row differs')
        grouped[i].append(row)
    inputs={};outputs=done['outputs'];out=Path(render_cfg['out'])
    for i,sid in enumerate(ids):
        cp=out/'source_checkpoints'/f'{i:04d}.json';metrics=out/'sources'/f'{i:04d}.json';archive=out/'images'/f'{i:04d}.npz'
        require(all(outputs.get(str(p))==sha(p) for p in (cp,metrics,archive)),'Source files absent or unsealed')
        receipt=read(cp)
        require(receipt['status']=='H_INITIAL_RX_SOURCE_COMPLETE' and receipt['source_id']==sid and receipt['source_index']==i
                and receipt['frame_count']==len(grouped[i])==48 and receipt['images_scored'] is True
                and receipt['source_decode_complete'] is True and receipt['new_packet_decodes']==0
                and receipt['registration_sha256']==done['registration_sha256']
                and receipt['config_sha256']==done['config_sha256'],'Source receipt scope differs')
        require(receipt['outputs']==bind((archive,metrics)) and read(metrics)==grouped[i],
                'Source metrics differ from combined actual frame metrics')
        require(all(row['image_archive']==str(archive) for row in grouped[i]),'Image reference crosses source archive')
        verify(receipt['input_bindings']);inputs=merge(inputs,receipt['input_bindings'])
    return dict(status='ALL_200_SOURCE_RECEIPTS_AND_9600_ROWS_MATCH',source_count=200,frame_count=9600,
                source_input_bindings=inputs,RGB_arrays_redecoded=False)


def predecessor(request):
    paths={k:pin(request[k]) for k in ('render_owner_config','render_registration','render_owner_launch','render_config',
                                      'selector_module','interpretation','closeout_registration','closeout_certificate',
                                      'stop_release','closeout_completion')}
    cfg,reg=read(paths['render_config']),read(paths['render_registration'])
    bound=merge(reg['input_bindings'],reg['source_bindings']);verify(bound)
    for key in ('owner_module','wait_module'):
        require(bound.get(cfg[key])==sha(cfg[key]),'Verifier not in frozen predecessor: '+key)
    for key in ('render_owner_config','render_config','selector_module','interpretation'):
        require(bound.get(str(paths[key]))==sha(paths[key]),'Frozen predecessor dependency absent: '+key)
    entry=Path(cfg['runtime_dir'])/'h_payload_render_driver.py'
    require(bound.get(str(entry))==sha(entry),'Frozen render verifier missing')
    a=module(cfg['owner_module'],'selection_bound_owner');p=module(cfg['wait_module'],'selection_bound_prior')
    closure=verify_render_closed(a,p,paths['render_owner_config'],paths['render_registration'],paths['render_owner_launch'],
                                 paths['render_config'],a.raw_process_state,
                                 {k:paths[k] for k in ('closeout_registration','closeout_certificate','stop_release','closeout_completion')})
    # This original loader verifies CPU traces, paid ledger results, S1 assets,
    # source/model bindings and CPU exit evidence. It constructs no model/backend.
    renderer=module(entry,'selection_bound_render_verifier');native_ctx=renderer.load_registered(str(paths['render_config']))
    before=a.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),PHASES,quiescent=True);check_budget(before)
    require(cfg['phase_limits']==PHASES and native_ctx['before']==before,'Budget changed during predecessor audit')
    done=read(closure['render_completion']);shortlist=read(cfg['shortlist'])
    require(len(shortlist['whole_candidates'])==16 and len(shortlist['partial_candidates'])==6,'Frozen shortlist shape differs')
    metrics=Path(cfg['out'])/'frame_metrics.json'
    require(done['outputs'].get(str(metrics))==sha(metrics),'Sealed whole metrics absent')
    audit=audit_source_receipts(cfg,done,shortlist['source_ids'],read(metrics))
    require(a.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),PHASES,quiescent=True)==before,
            'Shared ledger changed during receipt audit')
    return dict(paths=paths,cfg=cfg,reg=reg,owner=a,prior=p,closure=closure,done=done,shortlist=shortlist,
                metrics=metrics,audit=audit,before=before)


def make_configs(request,previous,render_cfg,execution):
    regpath=execution/'execution_registration.json';config=execution/'selection_config.json'
    cfg=dict(schema='H_INITIAL_SELECTION_CONFIG_R2',registration=str(regpath),owner_config=str(execution/'owner_config.json'),
        request_path=request['request_path'],out=request['selection_out'],root=render_cfg['root'],H_out=previous['out'],
        stop_file=render_cfg['stop_file'],max_seconds=request['max_seconds'],overall_deadline_unix=DEADLINE)
    owner=copy.deepcopy(previous);owner.update(owner_out=str(execution),registration=str(regpath))
    owner['stages']=[dict(id='freeze',resource='cpu',requires=[],max_seconds=request['max_seconds'],jobs=[
        dict(id='initial_selection',argv=[request['python'],'-B',str(Path(__file__).resolve()),'--stage','run','--config',str(config)],
            cwd=cfg['root'],out=cfg['out'],completion=str(Path(cfg['out'])/'completion.json'),
            accepted_statuses=['H_INITIAL_TRUE200_WHOLE_SELECTION_COMPLETE'],
            receipt_expect=dict(selected_count=8,source_count=200,measured_frames=9600,new_packet_decodes=0,
                GPU_used=False,development_used=False,holdout_used=False,full1000_calibration_complete=False))])]
    return cfg,owner


def register(request_path):
    request_path=Path(request_path).resolve();r=read(request_path)
    require(r['schema']=='H_INITIAL_SELECTION_REGISTRATION_REQUEST_R2','Wrong request schema')
    require(sys.platform.startswith('linux'),'Actual registration requires Linux process evidence')
    require(time.time()<DEADLINE,'Original H deadline expired')
    ctx=predecessor(r);a=ctx['owner'];old=read(ctx['paths']['render_owner_config'])
    qualified_path=pin(r['prepared_qualification']);qualified=read(qualified_path)
    require(qualified['status']=='H_INITIAL_SELECTION_R2_CPU_QUALIFICATION_PASS' and qualified['GPU_used'] is False
            and qualified['new_packet_decodes']==0 and qualified['results']
            and all(x['exit_code']==0 for x in qualified['results']),'New helper CPU qualification not PASS')
    verify(qualified['source_bindings'])
    require(qualified['source_bindings'].get(str(Path(__file__).resolve()))==sha(__file__)
            and qualified['source_bindings'].get(str(ctx['paths']['selector_module']))==sha(ctx['paths']['selector_module']),
            'Qualification does not bind current selector/helper')
    execution,out,h=Path(r['execution_dir']),Path(r['selection_out']),Path(old['out'])
    for p in (execution,out):require(p.is_absolute() and h in p.parents and not p.exists(),'New independent H path required: '+str(p))
    require(execution!=out and execution not in out.parents and out not in execution.parents,'Evidence/results overlap')
    require(Path(r['python']).is_absolute() and Path(r['python']).is_file() and 0<r['max_seconds']<=3600,
            'Missing interpreter or invalid finite CPU time cap')
    sources=merge(ctx['reg']['source_bindings'],qualified['source_bindings'],bind((Path(__file__).resolve(),)))
    inputs=merge(ctx['reg']['input_bindings'],ctx['closure']['bindings'],ctx['done']['outputs'],ctx['audit']['source_input_bindings'],
                 bind((request_path,qualified_path,*ctx['paths'].values())))
    verify(sources);verify(inputs);r=dict(r,request_path=str(request_path))
    cfg,owner=make_configs(r,old,ctx['cfg'],execution)
    execution.mkdir()
    try:
        gate=execution/'predecessor_closure.json';save(gate,dict(ctx['closure'],source_audit=ctx['audit'],budget=ctx['before']))
        cp,op,rp=execution/'selection_config.json',execution/'owner_config.json',execution/'execution_registration.json'
        save(cp,cfg);save(op,owner);inputs=merge(inputs,bind((gate,cp,op)))
        new=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,
            owner_config_sha256=sha(op),allowed_stage_ids=['freeze'],phase_limits=PHASES,
            source_bindings=sources,input_bindings=inputs,render_completion_sha256=sha(ctx['closure']['render_completion']),
            budget_registration_sha256=sha(ctx['cfg']['budget_registration']),budget_before=ctx['before'],
            scientific_protocol_sha256=sha(ctx['cfg']['protocol']),selector_sha256=sha(ctx['paths']['selector_module']),
            source_count=200,measured_frames=9600,selected_count=8,partial_reselected=False,original_render_owner_success=False,
            predecessor_closeout_certificate_sha256=sha(ctx['paths']['closeout_certificate']),
            selection_rule='Original frozen h_payload_select.select_initial; mean of source-wise three-noise PSNR; original epsilon/tie',
            GPU_jobs=0,new_packet_decodes=0,new_visual_inference=0,development_used=False,holdout_started=False,
            C_started=False,H_full_delivery_claimed=False,scientific_protocol_modified=False,
            full1000_calibration_complete=False,future_stage_automatic=False,created_unix=time.time())
        a.validate_config(owner,new,sha(op));verify(sources);verify(inputs)
        require(a.budget_snapshot(ctx['cfg']['ledger'],sha(ctx['cfg']['budget_registration']),PHASES,quiescent=True)==ctx['before'],
                'Shared ledger changed before sealing')
        save(rp,new)
        result=dict(status='H_INITIAL_SELECTION_R2_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(rp),owner_config_sha256=sha(op),
            outputs=bind((gate,cp,op,rp)),workers_started=False,GPU_used=False,new_packet_decodes=0,
            scope='Eight whole winners only; original six partial shortlist entries remain unmeasured references',
            future_stage_automatic=False)
        save(execution/'registration_completion.json',result);return result
    except BaseException as error:
        save(execution/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),traceback=traceback.format_exc()))
        raise


def verify_live_cpu_owner(ctx,cfg,reg,config_path):
    a=ctx['owner'];ocfg=read(cfg['owner_config']);a.validate_config(ocfg,reg,sha(cfg['owner_config']))
    require(reg['allowed_stage_ids']==['freeze'] and len(ocfg['stages'])==1,'Wrong live selector scope')
    stage=ocfg['stages'][0];require(stage['resource']=='cpu' and len(stage['jobs'])==1,'Only one CPU selector may run')
    job=stage['jobs'][0];expected=[job['argv'][0],'-B',str(Path(__file__).resolve()),'--stage','run','--config',str(config_path)]
    require(job['argv']==expected and job['out']==cfg['out'],'Current selector not the registered command')
    base=Path(ocfg['owner_out']);owner_id=read(base/'owner_identity.json')
    require(a.same_identity(owner_id,a.identity(os.getppid())) and owner_id['registration_sha256']==sha(cfg['registration'])
            and owner_id['config_sha256']==sha(cfg['owner_config']),'Parent is not the registered live owner')
    lp=base/'stages/freeze/workers/initial_selection/launch.json';launch=read(lp)
    require(a.same_identity(launch['identity'],a.identity(os.getpid())) and launch['argv']==expected
            and launch['registration_sha256']==sha(cfg['registration']) and launch['resource']=='cpu'
            and launch['threads']==2 and set(launch['affinity'])==set(ocfg['cpu_affinities'][0])
            and set(os.sched_getaffinity(0))==set(launch['affinity']),'CPU selector identity/affinity differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU-only environment differs')
    return bind((base/'owner_identity.json',lp))


def run(config_path):
    config_path=Path(config_path).resolve();cfg=read(config_path);reg=read(cfg['registration'])
    require(cfg['schema']=='H_INITIAL_SELECTION_CONFIG_R2' and reg['status']=='H_EXECUTION_REVISION_REGISTERED'
            and reg['allowed_stage_ids']==['freeze'],'Unregistered selection')
    verify(reg['source_bindings']);verify(reg['input_bindings']);bound=merge(reg['source_bindings'],reg['input_bindings'])
    for p in (config_path,Path(__file__).resolve(),Path(cfg['request_path']),Path(cfg['owner_config'])):
        require(bound.get(str(p))==sha(p),'Execution dependency unbound: '+str(p))
    require_fresh_output(cfg['out'])
    request=read(cfg['request_path']);ctx=predecessor(request)
    require(ctx['before']==reg['budget_before'],'Original budget changed before selection')
    launch_bindings=verify_live_cpu_owner(ctx,cfg,reg,config_path)
    out=Path(cfg['out']);out.mkdir(exist_ok=True);started=time.monotonic()
    save(out/'attempt.json',dict(status='STARTED',registration_sha256=sha(cfg['registration']),new_packet_decodes=0,GPU_used=False))
    try:
        require(not Path(cfg['stop_file']).exists() and time.time()<DEADLINE,'STOP/deadline blocks selection')
        selector=module(ctx['paths']['selector_module'],'selection_bound_pure_selector')
        result=selector.select_initial(Path(ctx['cfg']['shortlist']).read_bytes(),ctx['metrics'].read_bytes(),
            frame_metrics_path=str(ctx['metrics']),render_completion=ctx['done'],protocol=read(ctx['cfg']['protocol']),
            interpretation=read(ctx['paths']['interpretation']))
        require(result['selected_count']==8 and result['measured_frames']==9600,'Whole selection result scope differs')
        selected=out/'selected_whole.json';save(selected,result)
        partial=out/'partial_shortlist_reference.json';save(partial,dict(status='FROZEN_PRESCREEN_REFERENCE_ONLY_NOT_RESELECTED',
            original_shortlist_path=ctx['cfg']['shortlist'],original_shortlist_sha256=sha(ctx['cfg']['shortlist']),
            partial_candidates=ctx['shortlist']['partial_candidates'],candidate_count=6,
            source='Original frozen prescreen shortlist; this whole selector does not evaluate partial candidates',
            actual_partial_calibration_complete=False,partial_reselected=False))
        summary=out/'whole_summary.csv';fields=['candidate_id','snr_db','arm','slot','source_count','frames',
            'mean_per_source_psnr_db','mean_overall_mse','psnr_of_mean_mse_db','mean_attempted_tx_probability_scales']
        with summary.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
            w.writerows({k:row[k] for k in fields} for row in result['all_candidate_summaries'])
        require(time.monotonic()-started<cfg['max_seconds'] and time.time()<DEADLINE
                and not Path(cfg['stop_file']).exists(),'Selector STOP/deadline exceeded')
        verify_live_cpu_owner(ctx,cfg,reg,config_path);verify(reg['source_bindings']);verify(reg['input_bindings'])
        after=ctx['owner'].budget_snapshot(ctx['cfg']['ledger'],sha(ctx['cfg']['budget_registration']),PHASES,quiescent=True)
        require(after==ctx['before'],'CPU selection changed packet ledger')
        done=dict(status='H_INITIAL_TRUE200_WHOLE_SELECTION_COMPLETE',registration_sha256=sha(cfg['registration']),
            config_sha256=sha(config_path),source_count=200,measured_frames=9600,candidate_count=16,selected_count=8,
            outputs=bind((selected,partial,summary)),source_bindings=reg['source_bindings'],input_bindings=reg['input_bindings'],
            predecessor_closure_bindings=ctx['closure']['bindings'],live_launch_bindings=launch_bindings,budget=after,
            new_packet_decodes=0,new_visual_inference=0,GPU_used=False,development_used=False,holdout_used=False,
            partial_reselected=False,full1000_calibration_complete=False,H_full_delivery_claimed=False,
            future_stage_automatic=False,original_render_owner_success=False,
            predecessor_closeout_certificate_sha256=sha(ctx['paths']['closeout_certificate']),
            elapsed_seconds=time.monotonic()-started)
        save(out/'completion.json',done);return done
    except BaseException as error:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=sha(cfg['registration']),
            error=repr(error),traceback=traceback.format_exc(),new_packet_decodes=0,GPU_used=False));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('register','run'),required=True)
    parser.add_argument('--config',required=True);args=parser.parse_args()
    result=register(args.config) if args.stage=='register' else run(args.config)
    print(json.dumps({k:result[k] for k in ('status','registration_sha256')}))


if __name__=='__main__':main()
