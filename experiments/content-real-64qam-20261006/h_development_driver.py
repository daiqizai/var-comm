"""Prepared H-only development CPU worker/merge, gated by actual dependencies.

The development100 source exporter is a prerequisite, not supplied by this file.
No registration, decoder, image model or job is invoked merely by importing it.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import sys
import time
import traceback

COUNT=100
DEADLINE=1791564605.9549868
PHASES={'qualification':2000,'coarse':49152,'refine':12288,'initial_true200':20000,
        'whole_calibration':48000,'partial_calibration':36000,'development':13200,'engineering_reserve':19360}
STOP=False
REQUIRED=('protocol','catalogue','finalized','selected','partial_reference','shortlist','development_registration',
          'source_completion','source_qualification','asset_completion','asset_manifest','selection_completion','original_payload_config',
          'qualification_completion','reference_qualification','budget_registration','ledger_module','owner_module',
          'wait_module','calibration_driver_module')


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Changed bound file: '+p)
def bind(paths):return {str(p):sha(p) for p in paths}
def merged(*values):
    result={}
    for valueset in values:
        for p,s in valueset.items():
            require(p not in result or result[p]==s,'Conflicting SHA binding: '+p);result[p]=s
    return result
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def indices(worker):
    require(type(worker)is int and worker in (0,1),'Exactly two workers required');return list(range(worker,COUNT,2))
def stopped(*_):
    global STOP
    STOP=True


def closed_batch(api,wait,spec,owner_module,science_expected,state_reader):
    """Normal completion only, including a development-source preparation batch.

    Frozen wait.verify_batch forbids *any* development read because it served
    the earlier calibration pipeline. Here the exact source receipt explicitly
    allows preparation of the original100; no generic failure exemption exists.
    """
    oc,reg,launch=read(spec['config']),read(spec['registration']),read(spec['launch'])
    api.validate_config(oc,reg,sha(spec['config']));base=Path(oc['owner_out']);ident=launch['identity']
    require(oc['registration']==spec['registration'] and launch['registration_sha256']==sha(spec['registration'])
        and launch['owner_config_sha256']==sha(spec['config']) and launch['argv']==ident['argv']
        and ident['argv'][1:]==['-B',owner_module,'--config',spec['config']],'Predecessor launch differs')
    ip=base/'owner_identity.json';actual=read(ip)
    require(api.same_identity(actual,ident) and actual['config_sha256']==sha(spec['config'])
        and actual['registration_sha256']==sha(spec['registration']),'Predecessor owner identity differs')
    require(wait.exited(ident,state_reader),'Predecessor owner still live or unreaped')
    for p in (Path(oc['out'])/'STOP',base/'STOP',base/'failure.json',base/'registration_failure.json'):
        require(not p.exists(),'Failed/stopped predecessor: '+str(p))
    done=read(base/'completion.json')
    require(done['status']=='REGISTERED_H_STAGE_BATCH_COMPLETE' and done['registration_sha256']==sha(spec['registration'])
        and done['allowed_stage_ids']==reg['allowed_stage_ids'] and done['qualification_passed'] is True
        and all(done[k] is False for k in ('future_stages_started','H_full_delivery_claimed','C_started','holdout_started'))
        and done['budget']['unresolved']==done['budget']['failed']==0,'Predecessor did not close normally')
    require([x['stage'] for x in done['completed']]==[s['id'] for s in oc['stages']],'Predecessor stage coverage differs')
    bound=bind((spec['config'],spec['registration'],spec['launch'],ip,base/'completion.json'));children=[];target=None
    for stage,link in zip(oc['stages'],done['completed']):
        d=base/'stages'/stage['id'];cp=d/'completion.json'
        require(link['completion']==str(cp) and link['sha256']==sha(cp),'Changed predecessor stage receipt')
        sr=read(cp);require(sr['status']=='REGISTERED_STAGE_COMPLETE' and sr['stage']==stage['id']
            and sr['registration_sha256']==sha(spec['registration']) and sr['budget']['unresolved']==sr['budget']['failed']==0,
            'Predecessor stage incomplete')
        jobs={j['id']:j for j in stage['jobs']}
        require(len(sr['jobs'])==len(jobs) and {e['job_id'] for e in sr['jobs']}==set(jobs),'Predecessor child coverage differs')
        bound.update(bind((cp,)))
        for event in sr['jobs']:
            j=jobs[event['job_id']];wd=d/'workers'/j['id'];ep=wd/'exit_receipt.json';lp=wd/'launch.json';log=wd/'worker.log'
            require(read(ep)==event and event['exit_code']==0 and event['registration_sha256']==sha(spec['registration'])
                and event['identity']==read(lp)['identity'] and event['identity']['argv']==j['argv']
                and read(lp)['argv']==j['argv'] and read(lp)['registration_sha256']==sha(spec['registration'])
                and int(event['identity']['uid'])==int(ident['uid']) and wait.exited(event['identity'],state_reader),
                'Predecessor child is failed/live/unreaped')
            require(event['closed_log_sha256']==sha(log) and event['completion']==j['completion']
                and event['completion_sha256']==sha(j['completion']),'Changed closed child evidence')
            for p in (Path(j['out'])/'failure.json',Path(j['out'])/'STOP'):
                require(not p.exists(),'Child failure/STOP blocks development')
            value=api.receipt(j['completion'],j['accepted_statuses'],sha(j.get('receipt_registration',spec['registration'])),
                              j.get('receipt_expect'),j['out'])
            require(value.get('holdout_used',False) is False,'Holdout use forbidden')
            if j['completion']==spec['completion']:
                require(target is None,'Duplicate expected science completion')
                for k,v in science_expected.items():require(value.get(k)==v,'Wrong prerequisite science: '+k)
                target=value
            children.append(event['identity']);bound.update(bind((ep,lp,log,j['completion'])));bound=merged(bound,value['outputs'])
    require(target is not None,'Required actual scientific completion missing from owner')
    verify(reg['source_bindings']);verify(reg['input_bindings'])
    return dict(owner=oc,registration=reg,done=target,owner_done=done,bindings=bound,
                owner_identity=ident,children=children,normal_owner_success=True)


def validate_sources(pop,source,assets,manifest):
    ids=pop['source_ids'];require(len(ids)==len(set(ids))==100,'Original100 population required')
    expected=dict(status='H_DEVELOPMENT100_SOURCE_CODEC_COMPLETE',source_count=100,source_codec_complete=True,
        pending_source_encoding_count=0,independent_roundtrip=True,population='development',development_used=True,
        holdout_used=False,new_packet_decodes=0,new_image_renders=0,new_metric_calls=0,training_updates=0)
    for k,v in expected.items():require(source.get(k)==v,'Required development100 codec prerequisite absent: '+k)
    require(assets['status']==manifest['status']=='H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'
        and assets['source_count']==100 and manifest['source_ids']==source['source_ids']==ids
        and manifest['preprocessing_ids']==pop['preprocessing_ids']
        and manifest['original_data_bindings']==pop['data_bindings'] and manifest['visual_identity']==pop['identity'],
        'Development assets do not match original100 metadata; calibration substitution forbidden')
    require(len(source['records'])==len(manifest['records'])==100,'Development100 checkpoint coverage incomplete')
    for i,(c,a) in enumerate(zip(source['records'],manifest['records'])):
        require(c['source_index']==a['source_index']==i and c['source_id']==a['source_id']==ids[i]
            and a['preprocessing_id']==pop['preprocessing_ids'][i]
            and source['outputs'].get(c['checkpoint'])==c['sha256']
            and assets['outputs'].get(a['checkpoint'])==a['checkpoint_sha256'], 'Development source record unsealed/misindexed')
    return ids


def source_qualification_covers_entries(qualification,owner,api):
    require(isinstance(qualification.get('source_bindings'),dict) and qualification['source_bindings'],
            'Source qualification has no bound implementation')
    for stage in owner['stages']:
        for job in stage['jobs']:
            entry=str(api.command_entry(job['argv']))
            require(qualification['source_bindings'].get(entry)==sha(entry),
                    'Source qualification did not cover actual executed entry: '+entry)


def budget_admission(before,current,phase_events=()):
    require(before['unresolved']==before['failed']==0 and before['phase_charged']['development']==0
        and before['development_remaining']==13200,'H development batch requires unused full original reserve')
    require(current['failed']==0 and all(current['phase_charged'][k]==v for k,v in before['phase_charged'].items() if k!='development'),
        'Unrelated H phase changed during development')
    n=current['phase_charged']['development']
    require(0<=n<=10800 and current['charged']==before['charged']+n and current['development_remaining']==13200-n>=2400,
        'H consumed reserved MAIN calls or counters changed')
    for eid in phase_events:
        match=re.fullmatch(r'H:development:Hslot(\d{2}):src(\d{4}):seed(620[123]):(header|body)',eid)
        require(match is not None and int(match[1])<18 and int(match[2])<100,'Unregistered development/MAIN event before H batch closure')


def prepare_context(cfg,bindings):
    for key in REQUIRED:require(bindings.get(cfg[key])==sha(cfg[key]),'Required original/new input unbound: '+key)
    for directory,names in ((cfg['runtime_dir'],('h64_catalog.py','h64_phy.py','h64_backend.py','h64_source.py')),
        (cfg['payload_dir'],('h_payload_cpu.py',)),(cfg['calibration_dir'],('h_full_payload_cpu.py','h_full_calibration_select.py')),
        (cfg['legacy_runtime'],('ldpc_backend.py','uep_phy.py','uep_common.py')),
        (str(Path(cfg['root'])/'src/var_comm'),('scale_channel.py','token_trellis.cpp'))):
        for name in names:
            p=str(Path(directory)/name);require(bindings.get(p)==sha(p),'Unbound original runtime: '+p)
    require(cfg['workers']==2 and cfg['phase_limits']==PHASES and cfg['overall_deadline_unix']==DEADLINE
        and 0<cfg['max_worker_seconds']<=21600 and len(cfg['cpu_affinities'])==2
        and all(len(x)==len(set(x))==2 for x in cfg['cpu_affinities'])
        and len(set(sum(cfg['cpu_affinities'],[])))==4,'CPU resource or deadline scope differs')
    initial=read(cfg['original_payload_config']);ireg=read(initial['registration'])
    verify(ireg['source_bindings']);verify(ireg['input_bindings'])
    require(bindings.get(initial['registration'])==sha(initial['registration'])
        and ireg['input_bindings'].get(cfg['original_payload_config'])==sha(cfg['original_payload_config']), 'Original PHY config lineage absent')
    for k in ('root','legacy_runtime','ledger','ledger_module','protocol','budget_registration','catalogue',
              'qualification_completion','reference_qualification','shortlist','stop_file'):
        require(cfg[k]==initial[k],'Original H physical prerequisite changed: '+k)
    budget=read(cfg['budget_registration'])
    require(budget['status']=='FROZEN' and budget['branch']=='H' and budget['total_cap']==200000
        and budget['phase_limits']==PHASES and Path(cfg['ledger']).is_file(),'Original independent H ledger required')
    api=module(cfg['owner_module'],'dev_bound_owner');wait=module(cfg['wait_module'],'dev_bound_wait')
    require(set(cfg['prerequisites'])=={'selection','source'},'Both actual selection/source normal batches required')
    closures={}
    for name,expected in (('selection',dict(status='H_FULL1000_CALIBRATION_SELECTION_COMPLETE',source_count=1000,
        measured_frames=42000,whole_count=8,selected_partial_count=2,whole_policy_reselected=False,
        development_used=False,new_packet_decodes=0)),('source',dict(status='H_DEVELOPMENT100_SOURCE_CODEC_COMPLETE',source_count=100))):
        spec=cfg['prerequisites'][name]
        for path in spec.values():require(bindings.get(path)==sha(path),'Unbound prerequisite '+name)
        require(spec['completion']==cfg[name+'_completion'],'Prerequisite completion path differs')
        closures[name]=closed_batch(api,wait,spec,cfg['owner_module'],expected,api.raw_process_state)
        require(closures[name]['owner']['root']==cfg['root'] and closures[name]['owner']['out']==cfg['H_out']
            and closures[name]['owner']['budget_path']==cfg['ledger'],'Prerequisite workspace/budget differs')
    selection,source=closures['selection']['done'],closures['source']['done']
    require(selection['outputs'].get(cfg['finalized'])==sha(cfg['finalized']),'Full1000 finalized policies not actually sealed')
    assets,manifest,pop=read(cfg['asset_completion']),read(cfg['asset_manifest']),read(cfg['development_registration'])
    for done in (source,assets):
        verify(done['outputs']);verify(done['source_bindings']);verify(done['input_bindings'])
        require(done['input_bindings'].get(cfg['development_registration'])==sha(cfg['development_registration']),
            'Development population not bound by source preparation')
    require(assets['outputs'].get(cfg['asset_manifest'])==sha(cfg['asset_manifest'])
        and source['input_bindings'].get(cfg['asset_completion'])==sha(cfg['asset_completion'])
        and source['input_bindings'].get(cfg['asset_manifest'])==sha(cfg['asset_manifest']), 'Codec/assets receipt chain differs')
    sourcequal=read(cfg['source_qualification']);verify(sourcequal['source_bindings'])
    require(sourcequal['status']=='H_DEVELOPMENT100_SOURCE_QUALIFICATION_PASS' and sourcequal['new_packet_decodes']==0
        and sourcequal['results'] and all(x['exit_code']==0 for x in sourcequal['results'])
        and source['input_bindings'].get(cfg['source_qualification'])==sha(cfg['source_qualification']),
        'Development100 source implementation lacks its own bound qualification')
    source_qualification_covers_entries(sourcequal,closures['source']['owner'],api)
    ids=validate_sources(pop,source,assets,manifest)
    qual=read(cfg['qualification_completion']);verify(qual['outputs']);verify(qual['source_bindings'])
    require(qual['status']=='H_PHY_QUALIFICATION_PASS' and qual['outputs'].get(cfg['catalogue'])==sha(cfg['catalogue'])
        and qual['budget_registration_sha256']==sha(cfg['budget_registration']), 'Qualified actual catalogue differs')
    for p in (cfg['runtime_dir'],cfg['payload_dir'],cfg['calibration_dir'],str(Path(__file__).absolute().parent)):sys.path.insert(0,p)
    import h_development_cpu as core
    args={k:read(cfg[k]) for k in ('protocol','catalogue','finalized','selected','partial_reference','shortlist','development_registration')}
    rows=core.schedule(args['finalized'],args['selected'],args['partial_reference'],args['shortlist'],args['catalogue'],args['protocol'])
    core.population(pop)
    for name,proof in args['finalized']['input_proofs'].items():
        if name in ('protocol','catalogue','shortlist','partial_reference'):
            require(proof=={'path':cfg[name],'sha256':sha(cfg[name])},'Final selection provenance differs: '+name)
    require(args['finalized']['input_proofs']['selected_whole']=={'path':cfg['selected'],'sha256':sha(cfg['selected'])},
        'Preserved whole input differs')
    return dict(cfg=cfg,source=source,assets=assets,manifest=manifest,ids=ids,api=api,wait=wait,
        closures=closures,budget=budget,qualification=qual,core=core,args=args,rows=rows)


def load_registered(config_path):
    cfg=read(config_path);reg=read(cfg['registration'])
    require(cfg['schema']=='H_DEVELOPMENT_CPU_CONFIG_V1' and reg['status']=='H_EXECUTION_REVISION_REGISTERED'
        and reg['branch']=='H' and reg['allowed_stage_ids']==['development','report']
        and reg['source_stage_scope']=='H_DEVELOPMENT_18_POINTS_CPU_ONLY','Separate H development-only registration required')
    verify(reg['source_bindings']);verify(reg['input_bindings']);bound=merged(reg['source_bindings'],reg['input_bindings'])
    for p in (str(Path(config_path).absolute()),str(Path(__file__).absolute()),str(Path(__file__).with_name('h_development_cpu.py').absolute()),
              cfg['engineering_contract'],cfg['owner_config']):require(bound.get(p)==sha(p),'New entry/config unbound')
    ctx=prepare_context(cfg,bound);contract=read(cfg['engineering_contract'])
    require('execution_registration_sha256' not in contract,'Contract/registration hash cycle forbidden')
    ctx['context']=ctx['core'].prepare(**ctx['args'],contract=dict(contract,execution_registration_sha256=sha(cfg['registration'])))
    ctx['reg']=reg;ctx['source_module']=module(cfg['calibration_driver_module'],'dev_frozen_asset_reader')
    current=ctx['api'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),PHASES,quiescent=False)
    with contextlib.closing(sqlite3.connect(cfg['ledger'])) as db:events=[x[0] for x in db.execute("SELECT event_id FROM events WHERE phase='development'")]
    budget_admission(reg['budget_before'],current,events)
    return ctx


def load_source(ctx,index):
    require(type(index)is int and 0<=index<100,'Development source index outside original100')
    # This frozen reader reads the supplied CP/NPZ only; its full1000 count lives
    # in the old loader, which is never called. The100 population was checked above.
    value=ctx['source_module'].load_source(ctx,index)
    acp=read(ctx['manifest']['records'][index]['checkpoint']);meta=ctx['args']['development_registration']
    require(acp['preprocessing_id']==meta['preprocessing_ids'][index]
        and acp['original_development_data_binding']==meta['data_bindings'][index], 'Source is not exact original development cache')
    return value


def runtime(ctx):
    # Reuse the qualified construction only, not the frozen full1000 runner.
    return ctx['source_module'].build_runtime(ctx)


def live_owner(ctx,config_path,worker):
    cfg=ctx['cfg'];api=ctx['api'];oc=read(cfg['owner_config']);api.validate_config(oc,ctx['reg'],sha(cfg['owner_config']))
    require([s['id'] for s in oc['stages']]==['development','report'] and all(s['resource']=='cpu' for s in oc['stages']),
        'Only this CPU phase/merge may run')
    base=Path(oc['owner_out']);ip=base/'owner_identity.json';ident=read(ip)
    require(api.same_identity(ident,api.identity(os.getppid())) and ident['registration_sha256']==sha(cfg['registration'])
        and ident['config_sha256']==sha(cfg['owner_config']),'Actual CPU parent is not registered owner')
    stage='development' if worker is not None else 'report';suffix=['--stage','merge'] if worker is None else ['--stage','worker','--worker-index',str(worker)]
    jobs=next(s['jobs'] for s in oc['stages'] if s['id']==stage)
    matches=[j for j in jobs if j['argv'][1:]==['-B',str(Path(__file__).absolute()),'--config',str(Path(config_path).absolute()),*suffix]]
    require(len(matches)==1,'Current CPU command not registered');job=matches[0]
    lp=base/'stages'/stage/'workers'/job['id']/'launch.json';started=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-started<10 and api.same_identity(ident,api.identity(os.getppid())),'Owner did not seal worker launch');time.sleep(.1)
    launch=read(lp);i=0 if worker is None else worker
    require(api.same_identity(launch['identity'],api.identity(os.getpid())) and launch['argv']==job['argv']
        and launch['registration_sha256']==sha(cfg['registration']) and launch['resource']=='cpu' and launch['threads']==2
        and set(launch['affinity'])==set(cfg['cpu_affinities'][i])==set(os.sched_getaffinity(0))
        and os.getpriority(os.PRIO_PROCESS,0)==15 and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Worker identity/resources differ')
    require(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
        'CPU thread limits differ')
    return bind((ip,lp))


def trace_inventory(ctx,paths,worker=None):
    core=ctx['core'];wanted=set(indices(worker) if worker is not None else range(100));seen=set();events={};frames=0;gray=pending=0
    entries={r['development_slot']:r for r in ctx['rows'] if r['status']=='FROZEN_POLICY_READY'}
    for p in paths:
        doc=read(p);i=doc['source_index'];require(i in wanted and i not in seen,'Wrong/duplicate source trace');seen.add(i)
        require(doc['status']=='H_DEVELOPMENT_CPU_SOURCE_TRACES' and doc['source_id']==ctx['ids'][i]
            and doc['registration_sha256']==sha(ctx['cfg']['registration']),'Development source trace identity differs')
        verify(doc['source_bindings']);keys=set()
        for f in doc['frames']:
            s,n=f['development_slot'],f['noise_seed'];require(s in entries and n in core.SEEDS and (s,n) not in keys,'Wrong/duplicate frame')
            keys.add((s,n));entry=entries[s];c=entry['candidate'];event=core.frame_event(s,i,n)
            require(f['status']=='H_DEVELOPMENT_CPU_FRAME_TRACE' and f['stage']==f['noise_stage']=='development'
                and f['event_id']==event and f['source_id']==ctx['ids'][i] and f['source_index']==i
                and f['role']==entry['role'] and f['candidate_id']==c['candidate_id'] and f['arm']==c['arm'] and f['snr_db']==c['snr_db']
                and f['public_frame_counter']==core.frame_counter(s,i,n) and f['scrambling_session']==core.SESSION
                and f['execution_registration_sha256']==sha(ctx['cfg']['registration']) and f['total_symbols']==1024
                and f['frame_normalized'] is False and f['arithmetic_source_decode_run'] is False
                and f['neural_metrics_run'] is False and f['policy_selection'] is False,'Development frame contract differs')
            rx=f['rx'];parsed=bool(rx['body'] and rx['body'].get('parser_accepted'))
            rp=ctx['context']['book'][str(rx['header']['profile_id'])] if rx['header']['header_ok'] else None
            require(f['rx_profile']==rp and (not parsed or rp is not None),'RX profile differs from paid header')
            if not parsed:state,g,tokens='WIRE_REJECT_GRAY',True,None
            elif rp['mode']=='raw':state,g,tokens='RAW_SOURCE_DECODED',False,core.phy.raw_tokens(rx['body']['payload'],rp).tolist()
            else:state,g,tokens='ARITHMETIC_CANONICAL_RX_REQUIRED',None,None
            require((f['rx_source_status'],f['gray'],f['received_raw_tokens'])==(state,g,tokens),'Unresolved/incorrect source parse claim')
            kinds=['header']+(['body'] if rx['body'] is not None else [])
            require(f['packet_event_ids']==[event+':'+k for k in kinds] and f['logical_packet_events']==len(kinds),'Metering references differ')
            for k in kinds:
                eid=event+':'+k;require(eid not in events,'Duplicate event');events[eid]={'kind':k,'result_sha256':digest(rx[k])}
            frames+=1;gray+=int(g is True);pending+=int(g is None)
        require(keys=={(s,n) for s in entries for n in core.SEEDS},'Missing policy/source/noise frame')
    require(seen==wanted,'Missing source traces')
    return events,dict(source_count=len(wanted),frame_count=frames,logical_packet_events=len(events),wire_gray_frames=gray,
        arithmetic_pending_frames=pending,declared_policy_snr_points=18,executable_policy_snr_points=len(entries))


def audit_ledger(path,expected,worker=None):
    seen=set()
    with contextlib.closing(sqlite3.connect(str(path))) as db:
        for eid,kind,status,result,checksum in db.execute("SELECT event_id,kind,status,result,result_sha FROM events WHERE phase='development'"):
            m=re.fullmatch(r'H:development:Hslot(\d{2}):src(\d{4}):seed(620[123]):(header|body)',eid)
            require(m is not None and int(m[1])<18 and int(m[2])<100,'Foreign/MAIN event in H-only batch')
            if worker is not None and int(m[2])%2!=worker:continue
            require(eid in expected and eid not in seen and status=='COMPLETE' and kind==expected[eid]['kind']
                and checksum==expected[eid]['result_sha256']==digest(json.loads(result)),'Actual prepaid result missing/changed')
            seen.add(eid)
        require(seen==set(expected),'Missing charged H events')
        if worker is None:
            n=db.execute("SELECT charged FROM counters WHERE phase='development'").fetchone()
            require(n and n[0]==len(expected)<=10800,'Development counter exceeds H allocation')
    return dict(status='H_DEVELOPMENT_CPU_TRACE_LEDGER_MATCH',paid_events=len(seen),event_ids_sha256=digest(sorted(seen)),all_complete=True)


def boundary(cfg,started,out):
    require(not STOP and not Path(cfg['stop_file']).exists() and not (Path(out)/'STOP').exists(),'STOP at safe source boundary')
    require(time.time()<DEADLINE and time.monotonic()-started<cfg['max_worker_seconds'],'Original deadline/CPU stage cap reached')


def run(config_path,worker):
    ctx=load_registered(config_path);cfg=ctx['cfg'];out=Path(cfg['out'])/(f'worker_{worker}' if worker is not None else '')
    out.mkdir(parents=True,exist_ok=True);require(not (out/'attempt.json').exists() and not (out/'failure.json').exists(),'Prior attempt preserved; no retry')
    import fcntl
    with (out/'execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (out/'attempt.json').exists(),'Concurrent/prior attempt')
        regsha=sha(cfg['registration']);started=time.monotonic()
        try:
            launch=live_owner(ctx,config_path,worker);boundary(cfg,started,out)
            m=module(cfg['ledger_module'],'development_bound_ledger');ledger=m.BudgetLedger(cfg['ledger'],sha(cfg['budget_registration']),'H',PHASES)
            atomic(out/'attempt.json',dict(status='STARTED',registration_sha256=regsha,worker_index=worker,identity=ledger.worker,automatic_retry=False))
            paths=[];outputs={};identities=[]
            if worker is not None:
                backend,header=runtime(ctx);entries=[r for r in ctx['rows'] if r['status']=='FROZEN_POLICY_READY']
                with (out/'traces.journal.jsonl').open('x',encoding='utf-8',newline='\n') as journal:
                    for i in indices(worker):
                        boundary(cfg,started,out);assets=load_source(ctx,i);frames=[]
                        for e in entries:
                            for seed in ctx['core'].SEEDS:
                                f=ctx['core'].run_frame(ctx=ctx['context'],development_slot=e['development_slot'],source_id=ctx['ids'][i],
                                    source_index=i,noise_seed=seed,scales=assets['scales'],arithmetic_bits=assets['arithmetic_bits'],backend=backend,header=header,ledger=ledger)
                                journal.write(canonical(f)+'\n');journal.flush();os.fsync(journal.fileno());frames.append(f)
                        p=out/'traces'/f'{i:04d}.json';atomic(p,dict(status='H_DEVELOPMENT_CPU_SOURCE_TRACES',registration_sha256=regsha,
                            source_index=i,source_id=ctx['ids'][i],frames=frames,source_bindings=assets['source_bindings']))
                        cp=out/'source_checkpoints'/f'{i:04d}.json';atomic(cp,dict(status='H_DEVELOPMENT_CPU_SOURCE_COMPLETE',registration_sha256=regsha,
                            source_index=i,source_id=ctx['ids'][i],frame_count=len(frames),outputs={str(p):sha(p)}))
                        outputs.update(bind((p,cp)));paths.append(p)
                        atomic(out/'status.json',dict(status='RUNNING',completed_sources=len(paths),total_sources=50,registration_sha256=regsha))
                outputs.update(bind((out/'traces.journal.jsonl',)));identities=[ledger.worker]
            else:
                for w in (0,1):
                    folder=Path(cfg['out'])/f'worker_{w}';p=folder/'completion.json';done=read(p);verify(done['outputs'])
                    require(not (folder/'failure.json').exists() and done['status']=='H_DEVELOPMENT_CPU_WORKER_COMPLETE'
                        and done['registration_sha256']==regsha and done['driver_config_sha256']==sha(config_path)
                        and done['worker_index']==w and done['source_indices']==indices(w),'Wrong/incomplete worker receipt')
                    for identity in done['worker_identities']:require(ctx['wait'].exited(identity,ctx['api'].raw_process_state),'Worker not reaped before merge')
                    for i in indices(w):
                        trace=folder/'traces'/f'{i:04d}.json';require(done['outputs'].get(str(trace))==sha(trace),'Unsealed worker trace');paths.append(trace)
                    outputs=merged(outputs,done['outputs'],bind((p,)));identities.extend(done['worker_identities'])
            expected,summary=trace_inventory(ctx,paths,worker);audit=audit_ledger(cfg['ledger'],expected,worker)
            verify(ctx['reg']['source_bindings']);verify(ctx['reg']['input_bindings']);boundary(cfg,started,out)
            done=dict(status='H_DEVELOPMENT_CPU_WORKER_COMPLETE' if worker is not None else 'H_DEVELOPMENT_CPU_RECEIVE_COMPLETE',
                registration_sha256=regsha,driver_config_sha256=sha(config_path),worker_index=worker,workers=2,
                source_indices=indices(worker) if worker is not None else list(range(100)),worker_identities=identities,
                source_ids=list(ctx['ids']),maximum_H_packet_calls=10800,MAIN_reserved_packet_calls=2400,
                outputs=outputs,**summary,ledger_audit=audit,source_bindings=ctx['reg']['source_bindings'],input_bindings=ctx['reg']['input_bindings'],
                predecessor_closure_bindings=merged(*(x['bindings'] for x in ctx['closures'].values())),live_launch_bindings=launch,
                source_completion_sha256=sha(cfg['source_completion']),finalized_policies_sha256=sha(cfg['finalized']),
                budget_registration_sha256=sha(cfg['budget_registration']),images_scored=False,source_decode_complete=False,
                new_packet_decodes=summary['logical_packet_events'],
                arithmetic_source_decode_complete=False,GPU_used=False,development_used=True,holdout_used=False,
                policy_selection=False,MAIN_complete=False,overall_development_complete=False,H_full_delivery_claimed=False,
                visual_stage_started=False,requires_owner_worker_exit_receipt_before_visual_stage=True)
            if worker is None:
                snapshot=ctx['api'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),PHASES,quiescent=True)
                budget_admission(ctx['reg']['budget_before'],snapshot,expected);require(snapshot['phase_charged']['development']==len(expected),'Ledger changed')
                done.update(budget_before=ctx['reg']['budget_before'],budget_after=snapshot)
            atomic(out/'completion.json',done);return done
        except BaseException as error:
            if not (out/'failure.json').exists():atomic(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,
                error=repr(error),traceback=traceback.format_exc(),ledger_never_refunded=True))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage',choices=('worker','merge'),required=True)
    p.add_argument('--worker-index',type=int);a=p.parse_args();signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    require((a.stage=='worker' and a.worker_index in (0,1)) or (a.stage=='merge' and a.worker_index is None),'Wrong worker/merge arguments')
    done=run(a.config,a.worker_index);print(json.dumps({k:done[k] for k in ('status','registration_sha256','frame_count')}))


if __name__=='__main__':main()
