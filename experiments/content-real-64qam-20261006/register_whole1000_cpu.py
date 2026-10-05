"""Seal only whole1000 CPU reception after the source1000 owner has closed.

No models, decoders, jobs, GPU work, policy search or subsequent queue are run.
The eight original winners and the full paid-header catalogue remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

DEADLINE=1791564605.9549868
REQUEST_NAMES=('assets_registrar_module','owner_module','wait_module','driver_module','core_module',
    'source_module','source_config','source_registration','source_owner_config','source_launch',
    'source_owner_completion','source_completion','initial_payload_config',
    'selection_owner_config','selection_registration','selection_launch','selection_owner_completion',
    'selection_config','selection_completion','selected','partial_reference','prepared_qualification')
INHERITED_KEYS=('root','runtime_dir','legacy_runtime','ledger','ledger_module','protocol','budget_registration',
    'budget_registration_sha256','catalogue','reference_qualification','qualification_completion',
    'shortlist','prescreen_completion','stop_file')


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def bind(paths):return {str(p):sha(p) for p in paths}
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Changed immutable binding: '+p)
def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting binding: '+p);result[p]=s
    return result
def pin(record):
    p=Path(record['path']);require(p.is_absolute() and sha(p)==record['sha256'],'Changed pinned input: '+str(p));return p
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def verify_source_closed(a,w,t,paths,state_reader):
    oc,reg,cfg=read(paths['source_owner_config']),read(paths['source_registration']),read(paths['source_config'])
    a.validate_config(oc,reg,sha(paths['source_owner_config']))
    require(oc['registration']==str(paths['source_registration']) and reg['allowed_stage_ids']==['source']
        and reg['source_stage_scope']=='FULL1000_SOURCE_CODEC_ONLY' and oc['phase_limits']==t.PHASES
        and oc['overall_deadline_unix']==DEADLINE,'Wrong source1000 predecessor scope')
    require(len(oc['stages'])==1 and oc['stages'][0]['id']=='source' and oc['stages'][0]['resource']=='gpu'
        and len(oc['stages'][0]['jobs'])==1,'Exactly one completed GPU source job required')
    job=oc['stages'][0]['jobs'][0];base=Path(oc['owner_out'])
    require(job['id']=='full1000_source' and job['argv']==[job['argv'][0],'-B',str(paths['source_module']),
        '--config',str(paths['source_config'])] and job['out']==cfg['out']
        and job['completion']==str(paths['source_completion']) and cfg['source_owner_config']==str(paths['source_owner_config'])
        and cfg['registration']==str(paths['source_registration']) and cfg['root']==oc['root']
        and cfg['H_out']==oc['out'] and paths['source_owner_completion']==base/'completion.json',
        'Source1000 original command/config differs')
    for p in (Path(oc['out'])/'STOP',base/'STOP',Path(cfg['out'])/'STOP'):
        require(not p.exists(),'STOP blocks whole1000 registration: '+str(p))
    for p in (base/'failure.json',base/'registration_failure.json',Path(cfg['out'])/'failure.json'):
        require(not p.exists(),'A failed source owner cannot be promoted to success: '+str(p))
    launch=read(paths['source_launch']);ident=launch['identity'];ip=base/'owner_identity.json';actual=read(ip)
    require(ident['argv']==launch['argv']==[launch['argv'][0],'-B',str(paths['owner_module']),
        '--config',str(paths['source_owner_config'])] and launch['registration_sha256']==sha(paths['source_registration'])
        and launch['owner_config_sha256']==sha(paths['source_owner_config']),'Source owner launch differs')
    require(all(actual[k]==ident[k] for k in ('pid','uid','start_ticks','argv'))
        and actual['registration_sha256']==sha(paths['source_registration'])
        and actual['config_sha256']==sha(paths['source_owner_config']),'Source owner identity differs')
    require(w.exited(ident,state_reader),'Source owner still live or unreaped')
    closed=w.verify_batch(a,paths['source_owner_config'],paths['source_registration'],ident['uid'],state_reader)
    done=read(paths['source_completion'])
    expected=dict(status='H_FULL1000_SOURCE_CODEC_COMPLETE',source_count=1000,source1000_codec_complete=True,
        reused_codec_sources=200,newly_encoded_sources=800,pending_source_encoding_count=0,
        independent_roundtrip=True,new_canonical_roundtrips=3200,new_packet_decodes=0,new_image_renders=0,
        new_metric_calls=0,development_used=False,holdout_used=False,training_updates=0,
        full1000_calibration_complete=False,GPU_used=True)
    for k,v in expected.items():require(done.get(k)==v,'Source1000 completion scope differs: '+k)
    require(done['registration_sha256']==sha(paths['source_registration']) and done['config_sha256']==sha(paths['source_config'])
        and closed['bindings'].get(str(paths['source_completion']))==sha(paths['source_completion']),
        'Source1000 successful completion not sealed')
    require(done['budget_before']==done['budget_after']==reg['budget_before'],'Source changed the registered packet budget')
    t.check_budget(done['budget_before']);t.check_budget(read(paths['source_owner_completion'])['budget'])
    for values in (reg['source_bindings'],reg['input_bindings'],done['source_bindings'],done['input_bindings'],done['outputs']):verify(values)
    bound=merge(reg['source_bindings'],reg['input_bindings'])
    for name in ('source_module','source_config','owner_module','wait_module','assets_registrar_module'):
        require(bound.get(str(paths[name]))==sha(paths[name]),'Source predecessor misses bound dependency: '+name)
    closed['bindings']=merge(closed['bindings'],bind((ip,*[paths[k] for k in ('source_config','source_registration',
        'source_owner_config','source_launch','source_owner_completion','source_completion')])))
    closed.update(owner_identity=ident,owner_exited=True,source_owner_success=True,original_render_owner_success=False)
    return dict(old=oc,source_reg=reg,source_cfg=cfg,source_done=done,closure=closed)


def make_contract(core,protocol,catalogue,selected,partial,shortlist,ids):
    args=dict(protocol=protocol,catalogue=catalogue,selected=selected,partial_reference=partial,shortlist=shortlist,source_ids=ids)
    schedule=core.schedule(selected,partial,shortlist,catalogue,ids)
    contract=dict(status='H_FULL_PAYLOAD_CPU_ENGINEERING_SEALED',engineering_choices=core.ENGINEERING_CHOICES,
        public_schedule=schedule,core_source_sha256=sha(core.__file__),initial_core_source_sha256=sha(core.initial.__file__),
        engineering_choices_canonical_sha256=core.digest(core.ENGINEERING_CHOICES),
        public_schedule_canonical_sha256=core.digest(schedule),
        execution_scope='ONLY whole_calibration; partial slots preserved for a later independent registration',
        complete_public_receive_catalogue=True,noise_stage='full_calibration',scientific_protocol_modified=False)
    for k,v in args.items():contract[k+'_canonical_sha256']=core.digest(v)
    # Pure contract verification only. No frame, backend or noise is constructed.
    core.prepare(**args,contract=dict(contract,execution_registration_sha256='0'*64))
    return contract


def make_configs(r,ctx):
    paths,old,initial=ctx['paths'],ctx['old'],ctx['initial'];execution=Path(r['execution_dir']);out=Path(r['payload_out'])
    cp,rp=execution/'payload_config.json',execution/'execution_registration.json'
    cfg={k:copy.deepcopy(initial[k]) for k in INHERITED_KEYS}
    cfg.update(schema='H_FULL_PAYLOAD_CONFIG_V1',phase='whole_calibration',out=str(out),registration=str(rp),
        owner_config=str(execution/'owner_config.json'),owner_module=str(paths['owner_module']),wait_module=str(paths['wait_module']),
        payload_dir=initial['runtime_dir'],engineering_contract=str(execution/'engineering_contract.json'),workers=2,
        cpu_affinities=copy.deepcopy(old['cpu_affinities']),phase_limits=copy.deepcopy(old['phase_limits']),
        max_worker_seconds=r['max_worker_seconds'],overall_deadline_unix=DEADLINE,
        source_completion=str(paths['source_completion']),asset_completion=ctx['source_cfg']['assets_completion'],
        asset_manifest=ctx['source_cfg']['assets_manifest'],predecessor_batches={
            'source':dict(config=str(paths['source_owner_config']),registration=str(paths['source_registration']),launch=str(paths['source_launch'])),
            'selection':dict(config=str(paths['selection_owner_config']),registration=str(paths['selection_registration']),launch=str(paths['selection_launch']))})
    for key in ('selected','partial_reference','selection_completion'):cfg[key]=str(paths[key])
    oc=copy.deepcopy(old);oc.update(owner_out=str(execution),registration=str(rp))
    jobs=[]
    common=dict(phase='whole_calibration',workers=2,images_scored=False,source_decode_complete=False,
        arithmetic_source_decode_complete=False,GPU_used=False,development_used=False,holdout_used=False,
        policy_selection=False,full1000_calibration_complete=False,maximum_packet_calls=48000,visual_stage_started=False)
    for i in (0,1):
        folder=out/f'worker_{i}'
        jobs.append(dict(id=f'whole_worker_{i}',argv=[r['python'],'-B',str(paths['driver_module']),'--config',str(cp),
            '--stage','worker','--worker-index',str(i)],cwd=old['root'],out=str(folder),completion=str(folder/'completion.json'),
            accepted_statuses=['H_FULL1000_CPU_WORKER_COMPLETE'],receipt_expect=dict(common,worker_index=i,source_count=500,frame_count=12000)))
    oc['stages']=[dict(id='whole_calibration',resource='cpu',requires=[],max_seconds=r['max_worker_seconds'],jobs=jobs),
        dict(id='report',resource='cpu',requires=['whole_calibration'],max_seconds=r['merge_seconds'],jobs=[dict(
            id='whole_merge',argv=[r['python'],'-B',str(paths['driver_module']),'--config',str(cp),'--stage','merge'],
            cwd=old['root'],out=str(out),completion=str(out/'completion.json'),accepted_statuses=['H_FULL1000_CPU_RECEIVE_COMPLETE'],
            receipt_expect=dict(common,worker_index=None,source_count=1000,frame_count=24000))])]
    return cfg,oc


def register(request_path):
    request_path=Path(request_path).absolute();r=read(request_path)
    require(r['schema']=='H_WHOLE1000_CPU_REGISTRATION_REQUEST_V1','Wrong whole1000 request')
    require(sys.platform.startswith('linux'),'Actual registration requires Linux process evidence')
    require(time.time()<DEADLINE,'Original H deadline expired')
    paths={k:pin(r[k]) for k in REQUEST_NAMES}
    t=module(paths['assets_registrar_module'],'whole1000_assets_registration_toolkit')
    a=module(paths['owner_module'],'whole1000_original_owner');w=module(paths['wait_module'],'whole1000_original_wait')
    ctx=verify_source_closed(a,w,t,paths,a.raw_process_state)
    selection=t.verify_selection(a,w,paths,a.raw_process_state)
    old,sreg,scfg=ctx['old'],ctx['source_reg'],ctx['source_cfg'];oldbound=merge(sreg['source_bindings'],sreg['input_bindings'])
    require(selection['owner_config']['root']==old['root'] and selection['owner_config']['out']==old['out'],
        'Selection and source use different H workspaces')
    for k in ('initial_payload_config','selection_registration','selection_owner_config','selection_config','selected','partial_reference'):
        require(oldbound.get(str(paths[k]))==sha(paths[k]),'Current source provenance does not bind original '+k)
    initial=read(paths['initial_payload_config']);ireg=read(initial['registration']);verify(ireg['source_bindings']);verify(ireg['input_bindings'])
    require(oldbound.get(initial['registration'])==sha(initial['registration']) and ireg['input_bindings'].get(str(paths['initial_payload_config']))==sha(paths['initial_payload_config']),
        'Initial physical config was not registered')
    require(initial['root']==old['root'] and initial['ledger']==old['budget_path'] and initial['phase_limits']==old['phase_limits']==t.PHASES
        and initial['protocol']==scfg['protocol'] and initial['budget_registration']==old['budget_registration']
        and initial['budget_registration_sha256']==old['budget_registration_sha256']
        and initial['stop_file']==str(Path(old['out'])/'STOP'),'Original physical protocol/budget differs')
    for k in INHERITED_KEYS:
        if k not in ('root','runtime_dir','legacy_runtime','ledger','budget_registration_sha256','stop_file'):
            require(oldbound.get(initial[k])==sha(initial[k]),'Unbound original physical dependency: '+k)
    for k in ('selected','partial_reference'):
        require(selection['completion']['outputs'].get(str(paths[k]))==sha(paths[k]),'Winner/reference not sealed by actual selector')
    q=read(paths['prepared_qualification']);verify(q['source_bindings'])
    require(q['status']=='H_WHOLE1000_CPU_QUALIFICATION_PASS' and q['GPU_used'] is False and q['new_packet_decodes']==0
        and q['results'] and all(x['exit_code']==0 for x in q['results']),'New CPU implementation not qualified')
    for p in (Path(__file__).absolute(),paths['core_module'],paths['driver_module']):
        require(q['source_bindings'].get(str(p))==sha(p),'Qualification misses current new source: '+str(p))
    require(paths['driver_module'].parent==paths['core_module'].parent and paths['driver_module'].name=='h_full_payload_driver.py'
        and paths['core_module'].name=='h_full_payload_cpu.py','Exact prepared entry/core placement required')
    sources=merge(sreg['source_bindings'],q['source_bindings'],bind((Path(__file__).absolute(),)))
    inputs=merge(sreg['input_bindings'],ctx['source_done']['outputs'],ctx['source_done']['input_bindings'],
        ctx['closure']['bindings'],selection['closure']['bindings'],selection['completion']['outputs'],bind((request_path,*paths.values())))
    verify(sources);verify(inputs)
    driver=module(paths['driver_module'],'whole1000_bound_cpu_driver')
    assets=read(scfg['assets_completion']);manifest=read(scfg['assets_manifest'])
    ids=driver.completed_sources(ctx['source_done'],assets,manifest)
    for p in (initial['runtime_dir'],str(paths['core_module'].parent)):sys.path.insert(0,p)
    core=module(paths['core_module'],'whole1000_bound_cpu_core')
    contract=make_contract(core,read(initial['protocol']),read(initial['catalogue']),read(paths['selected']),
        read(paths['partial_reference']),read(initial['shortlist']),ids)
    before=a.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],t.PHASES,quiescent=True);t.check_budget(before)
    require(before==ctx['source_done']['budget_after'],'Ledger changed after completed source1000')
    execution,out,h=Path(r['execution_dir']),Path(r['payload_out']),Path(old['out'])
    for p in (execution,out):require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent H directory required: '+str(p))
    require(execution!=out and execution not in out.parents and out not in execution.parents,'New paths overlap')
    require(type(r['max_worker_seconds']) is int and 0<r['max_worker_seconds']<=21600
        and type(r['merge_seconds']) is int and 0<r['merge_seconds']<=r['max_worker_seconds']
        and Path(r['python']).is_absolute() and Path(r['python']).is_file(),'Invalid finite CPU cap/interpreter')
    require(set(sum(old['cpu_affinities'],[]))<=set(os.sched_getaffinity(0)),'Registered CPU affinity unavailable')
    ctx.update(paths=paths,initial=initial);cfg,oc=make_configs(r,ctx)
    execution.mkdir()
    try:
        cp,op,rp=execution/'payload_config.json',execution/'owner_config.json',execution/'execution_registration.json'
        gate=execution/'predecessor_closure.json';save(gate,dict(status='SOURCE1000_AND_SELECTOR_OWNERS_CLOSED',
            source=ctx['closure'],selection=selection['closure'],budget=before,original_render_owner_success=False))
        ep=execution/'engineering_contract.json';save(ep,contract);save(cp,cfg);save(op,oc)
        inputs=merge(inputs,bind((cp,op,ep,gate)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=execution.name,
            source_stage_scope='WHOLE1000_ACTUAL_CPU_RECEIVE_ONLY',allowed_stage_ids=['whole_calibration','report'],
            owner_config_sha256=sha(op),phase_limits=t.PHASES,budget_registration_sha256=old['budget_registration_sha256'],
            scientific_protocol_sha256=sha(cfg['protocol']),source_bindings=sources,input_bindings=inputs,budget_before=before,
            source_count=1000,selected_whole_candidates=8,noise_seeds=[6101,6102,6103],snrs_db=[13,19],
            frame_count=24000,maximum_packet_calls=48000,CPU_workers=2,threads_per_worker=2,nice=15,
            cpu_affinities=old['cpu_affinities'],GPU_jobs=0,noise_stage='full_calibration',full_public_catalogue=True,
            partial_started=False,visual_stage_started=False,policy_selection=False,development_used=False,holdout_started=False,
            full1000_calibration_complete=False,H_full_delivery_claimed=False,C_started=False,
            scientific_protocol_modified=False,future_stage_automatic=False,created_unix=time.time())
        a.validate_config(oc,reg,sha(op));verify(sources);verify(inputs)
        require(not (h/'STOP').exists() and time.time()<DEADLINE,'STOP/deadline changed')
        verify_source_closed(a,w,t,paths,a.raw_process_state);t.verify_selection(a,w,paths,a.raw_process_state)
        require(a.budget_snapshot(old['budget_path'],old['budget_registration_sha256'],t.PHASES,quiescent=True)==before,
            'Shared packet ledger changed during registration')
        save(rp,reg);driver.load_registered(cp)  # Pure receipt, contract and source hashes; never construct a decoder.
        done=dict(status='H_WHOLE1000_CPU_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(rp),owner_config_sha256=sha(op),
            outputs=bind((cp,op,rp,gate,ep)),source_count=1000,selected_whole_candidates=8,frame_count=24000,
            maximum_packet_calls=48000,workers_started=False,new_packet_decodes=0,GPU_used=False,
            partial_started=False,visual_stage_started=False,future_stage_automatic=False,
            required_launch_condition='This completion must verify and registration_failure.json must be absent; predecessor exits/STOP/ledger rechecked before launching.')
        save(execution/'registration_completion.json',done);return done
    except BaseException as error:
        save(execution/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',error=repr(error),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);a=p.parse_args()
    result=register(a.request);print(json.dumps({k:result[k] for k in ('status','registration_sha256')}))


if __name__=='__main__':main()
