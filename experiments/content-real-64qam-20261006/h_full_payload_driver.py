"""Prepared, independently registered full1000 CPU receive worker/merge.

No source encoding, canonical arithmetic inference, image scoring, selection or
job launch occurs here. Incomplete1000 assets are a hard stop before PHY import.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
import traceback

COUNT=1000
DEADLINE=1791564605.9549868
PHASE_LIMITS={'qualification':2000,'coarse':49152,'refine':12288,'initial_true200':20000,
              'whole_calibration':48000,'partial_calibration':36000,'development':13200,'engineering_reserve':19360}
STOP=False


def require(ok,message):
    if not ok: raise RuntimeError(message)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def verify(bindings):
    for p,s in bindings.items():require(Path(p).is_absolute() and sha(p)==s,'Changed bound file: '+p)
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def stop(*_):
    global STOP
    STOP=True
def indices(worker):
    require(type(worker)is int and worker in (0,1),'Two workers only');return list(range(worker,COUNT,2))
def checked(p,status):
    r=read(p);require(r['status']==status,'Wrong completion: '+str(p));verify(r['outputs']);return r
def token_sha(a):
    import numpy as np
    return hashlib.sha256(b'int64:680\0'+np.asarray(a,dtype='<i8').tobytes()).hexdigest()


def completed_sources(source,assets,manifest):
    """Validate the future source-codec completion, never accept CPU staging."""
    require(source['status']=='H_FULL1000_SOURCE_CODEC_COMPLETE' and source['source_count']==1000
            and source['source1000_codec_complete'] is True and source['pending_source_encoding_count']==0
            and source['reused_codec_sources']==200 and source['newly_encoded_sources']==800
            and source['development_used'] is False and source['holdout_used'] is False
            and source['new_packet_decodes']==0,'Full1000 source codec is incomplete or wrong population')
    require(assets['status']=='H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'
            and assets['source_count']==1000 and manifest['status']==assets['status'], 'Original full1000 CPU asset scope differs')
    ids=manifest['source_ids']
    require(len(ids)==len(set(ids))==1000 and source['source_ids']==ids,'Original full1000 source order changed')
    require(len(manifest['records'])==len(source['records'])==1000,'Source checkpoint count differs')
    for i,(a,c) in enumerate(zip(manifest['records'],source['records'])):
        require(a['source_index']==c['source_index']==i and a['source_id']==c['source_id']==ids[i], 'Source record identity differs')
        require(assets['outputs'].get(a['checkpoint'])==a['checkpoint_sha256']
                and source['outputs'].get(c['checkpoint'])==c['sha256'],'Source checkpoint not in completed scope')
    return ids


def verify_predecessors(cfg,reg,bound):
    """Require real successful and reaped selector/source owners before calls.

    The earlier failed render owner is certified through the unchanged selector
    provenance. It is not accepted as a successful predecessor here.
    """
    a=module(cfg['owner_module'],'full_bound_owner');w=module(cfg['wait_module'],'full_bound_wait')
    expected={'selection','source'}|({'whole'} if cfg['phase']=='partial_calibration' else set())
    require(set(cfg['predecessor_batches'])==expected,'Explicit closed prerequisite batches required')
    allbind={}
    for kind,spec in cfg['predecessor_batches'].items():
        for key in ('config','registration','launch'):
            require(bound.get(spec[key])==sha(spec[key]),'Unbound predecessor '+kind+'/'+key)
        oc,rr,launch=read(spec['config']),read(spec['registration']),read(spec['launch'])
        a.validate_config(oc,rr,sha(spec['config']));ident=launch['identity']
        require(oc['registration']==spec['registration'] and launch['argv']==ident['argv']
                and ident['argv'][1:]==['-B',cfg['owner_module'],'--config',spec['config']]
                and launch['registration_sha256']==sha(spec['registration'])
                and launch['owner_config_sha256']==sha(spec['config']),'Predecessor owner launch differs')
        require(w.exited(ident,a.raw_process_state),'Predecessor owner still live/unreaped')
        evidence=w.verify_batch(a,spec['config'],spec['registration'],ident['uid'],a.raw_process_state)
        science=cfg[{'selection':'selection_completion','source':'source_completion','whole':'whole_cpu_completion'}[kind]]
        require(evidence['bindings'].get(science)==sha(science),'Wrong predecessor science completion')
        if kind=='whole':
            done=read(science)
            require(done['status']=='H_FULL1000_CPU_RECEIVE_COMPLETE' and done['phase']=='whole_calibration'
                    and done['source_count']==1000 and done['frame_count']==24000
                    and done['selected_sha256']==sha(cfg['selected'])
                    and done['source_completion_sha256']==sha(cfg['source_completion'])
                    and done['source_decode_complete'] is False,'Wrong preceding whole CPU calibration')
        allbind.update(evidence['bindings'])
    return a,allbind


def load_registered(config_path):
    cfg=read(config_path);reg=read(cfg['registration']);phase=cfg['phase']
    require(cfg['schema']=='H_FULL_PAYLOAD_CONFIG_V1' and phase in ('whole_calibration','partial_calibration')
            and reg['status']=='H_EXECUTION_REVISION_REGISTERED' and reg['branch']=='H'
            and phase in reg['allowed_stage_ids'],'Independent whole/partial execution registration required')
    verify(reg['source_bindings']);verify(reg['input_bindings']);bound=dict(reg['input_bindings'],**reg['source_bindings'])
    for p in (str(Path(config_path).absolute()),str(Path(__file__).absolute()),
              str(Path(__file__).absolute().with_name('h_full_payload_cpu.py'))):
        require(bound.get(p)==sha(p),'Unbound new entry/config: '+p)
    keys=('protocol','engineering_contract','budget_registration','ledger_module','catalogue','reference_qualification',
          'qualification_completion','shortlist','prescreen_completion','selected','partial_reference','selection_completion',
          'source_completion','asset_completion','asset_manifest','owner_module','wait_module','owner_config')
    if phase=='partial_calibration':keys+=('whole_cpu_completion',)
    for key in keys:require(bound.get(cfg[key])==sha(cfg[key]),'Required dependency unbound: '+key)
    for directory,names in ((cfg['runtime_dir'],('h64_catalog.py','h64_phy.py','h64_backend.py','h64_source.py')),
         (cfg['payload_dir'],('h_payload_cpu.py',)),(cfg['legacy_runtime'],('ldpc_backend.py','uep_phy.py','uep_common.py')),
         (str(Path(cfg['root'])/'src/var_comm'),('scale_channel.py','token_trellis.cpp'))):
        for name in names:
            p=str(Path(directory)/name);require(bound.get(p)==sha(p),'Frozen runtime not bound: '+p)
    budget=read(cfg['budget_registration'])
    require(budget['status']=='FROZEN' and budget['branch']=='H' and budget['total_cap']==200000
            and budget['phase_limits']==cfg['phase_limits']==reg['phase_limits']==PHASE_LIMITS
            and cfg['budget_registration_sha256']==sha(cfg['budget_registration'])
            and Path(cfg['ledger']).is_file(),'Original existing immutable H ledger/budget required')
    require(cfg['workers']==2 and len(cfg['cpu_affinities'])==2
            and all(len(set(x))==len(x)==2 for x in cfg['cpu_affinities'])
            and len(set(sum(cfg['cpu_affinities'],[])))==4 and 0<cfg['max_worker_seconds']<=21600
            and cfg['overall_deadline_unix']==DEADLINE,'CPU/deadline resource scope changed')
    source=checked(cfg['source_completion'],'H_FULL1000_SOURCE_CODEC_COMPLETE')
    assets=checked(cfg['asset_completion'],'H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE')
    verify(source['input_bindings']);verify(source['source_bindings'])
    verify(assets['input_bindings']);verify(assets['source_bindings'])
    require(assets['outputs'].get(cfg['asset_manifest'])==sha(cfg['asset_manifest']),'Asset manifest unsealed')
    require(source['input_bindings'].get(cfg['asset_completion'])==sha(cfg['asset_completion'])
            and source['input_bindings'].get(cfg['asset_manifest'])==sha(cfg['asset_manifest']),'Codec did not bind exact assets')
    manifest=read(cfg['asset_manifest']);ids=completed_sources(source,assets,manifest)
    qual=checked(cfg['qualification_completion'],'H_PHY_QUALIFICATION_PASS');verify(qual['source_bindings'])
    require(qual['outputs'].get(cfg['catalogue'])==sha(cfg['catalogue'])
            and qual['budget_registration_sha256']==sha(cfg['budget_registration']),'Qualified catalogue/budget differs')
    prescreen=checked(cfg['prescreen_completion'],'H_PRESCREEN_COMPLETE_FINAL')
    require(prescreen['outputs'].get(cfg['shortlist'])==sha(cfg['shortlist'])
            and prescreen['ready_for_real_calibration'] is True,'Prescreen shortlist changed')
    selection=checked(cfg['selection_completion'],'H_INITIAL_TRUE200_WHOLE_SELECTION_COMPLETE')
    require(selection['source_count']==200 and selection['selected_count']==8 and selection['measured_frames']==9600
            and selection['partial_reselected'] is False and selection['new_packet_decodes']==0
            and selection['development_used'] is False and selection['holdout_used'] is False,
            'Initial whole selection science differs')
    for key in ('selected','partial_reference'):
        require(selection['outputs'].get(cfg[key])==sha(cfg[key]),'Selected/reference file not sealed')
    selected,partial=read(cfg['selected']),read(cfg['partial_reference'])
    require(selected['input_proofs']['shortlist_sha256']==sha(cfg['shortlist'])
            and partial['original_shortlist_path']==cfg['shortlist']
            and partial['original_shortlist_sha256']==sha(cfg['shortlist']),'Frozen prescreen provenance changed')
    a,predecessors=verify_predecessors(cfg,reg,bound)
    for p in (cfg['runtime_dir'],cfg['payload_dir'],str(Path(__file__).absolute().parent)):
        sys.path.insert(0,p)
    import h_full_payload_cpu as core
    contract=read(cfg['engineering_contract'])
    require('execution_registration_sha256' not in contract,'No registration/contract hash cycle')
    context=core.prepare(read(cfg['protocol']),read(cfg['catalogue']),selected,partial,read(cfg['shortlist']),ids,
                         dict(contract,execution_registration_sha256=sha(cfg['registration'])))
    return dict(cfg=cfg,reg=reg,budget=budget,source=source,assets=assets,manifest=manifest,
                context=context,core=core,qualification=qual,api=a,predecessor_bindings=predecessors)


def load_source(ctx,index):
    import numpy as np
    cfg=ctx['cfg'];row=ctx['source']['records'][index];a=ctx['manifest']['records'][index]
    require(sha(row['checkpoint'])==row['sha256'] and sha(a['checkpoint'])==a['checkpoint_sha256'],'Changed source checkpoints')
    cp,acp=read(row['checkpoint']),read(a['checkpoint']);sid=ctx['context']['source_ids'][index]
    require(cp['source_index']==acp['source_index']==index and cp['source_id']==acp['source_id']==sid
            and cp['registration_sha256']==ctx['source']['registration_sha256']
            and cp['independent_roundtrip'] is True and cp['tokens_sha256']==acp['tokens_sha256']==a['tokens_sha256']
            and cp['source_assets_checkpoint']=={'path':a['checkpoint'],'sha256':a['checkpoint_sha256']},
            'Full source codec is not linked to actual source assets')
    verify(cp['outputs']);verify(acp['outputs'])
    archives=list(cp['outputs']);require(len(archives)==1 and Path(archives[0]).suffix=='.npz','Expected one source bit archive')
    archive=archives[0];raw=acp['archive']
    require(raw==a['archive'] and raw in acp['outputs'] and ctx['source']['outputs'].get(archive)==sha(archive),
            'Full source archives not sealed')
    with np.load(raw,allow_pickle=False) as z:
        tokens=z['tokens'];scales=ctx['core'].initial.split_raw_tokens(tokens)
    require(token_sha(tokens)==cp['tokens_sha256'],'Codec refers to another token vector')
    lengths={r['m']:r for r in cp['lengths']}
    require(len(cp['lengths'])==4 and set(lengths)=={6,7,8,9},'Full source lengths incomplete')
    with np.load(archive,allow_pickle=False) as z:
        bits={m:ctx['core'].phy.binary(z[f'm{m}_bits']) for m in (6,7,8,9)}
    for m,b in bits.items():
        require(len(b)>=2 and len(b)==lengths[m]['arithmetic_bits']
                and lengths[m]['raw_bits']==12*ctx['core'].token_count(m)
                and lengths[m]['zero_extension_reads']==30,'Source canonical bit count differs')
    return dict(scales=scales,arithmetic_bits=bits,source_bindings={row['checkpoint']:row['sha256'],
        a['checkpoint']:a['checkpoint_sha256'],archive:sha(archive),raw:sha(raw)})


def make_ledger(ctx):
    cfg=ctx['cfg'];m=module(cfg['ledger_module'],'full_registered_ledger')
    return m.BudgetLedger(cfg['ledger'],sha(cfg['budget_registration']),'H',PHASE_LIMITS)


def configure(ctx,worker):
    cfg=ctx['cfg'];require(sys.platform.startswith('linux'),'Actual workers require Linux')
    cores=cfg['cpu_affinities'][worker];require(set(cores)<=os.sched_getaffinity(0),'Registered affinity unavailable')
    os.sched_setaffinity(0,set(cores));os.setpriority(os.PRIO_PROCESS,0,15);os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'


def build_runtime(ctx):
    cfg=ctx['cfg']
    for p in (cfg['legacy_runtime'],str(Path(cfg['root'])/'src')):sys.path.insert(0,p)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    from h64_backend import H64Backend
    from uep_phy import Header
    b=H64Backend(reference_qualification=read(cfg['reference_qualification']),
                 legacy_backend_path=Path(cfg['legacy_runtime'])/'ldpc_backend.py')
    require(b.identity==ctx['qualification']['backend_identity'],'Backend differs from qualified PHY')
    for bucket in ctx['context']['catalogue']['buckets']:
        require(bucket['admission']=='ADMITTED' and b.plan(bucket['k'],bucket['n'],bucket['q'])==bucket['layout'],
                'Registered actual layout changed')
    return b,Header(cfg['root'])


def verify_live_owner(ctx,config_path,worker):
    cfg=ctx['cfg'];a=ctx['api'];oc=read(cfg['owner_config']);a.validate_config(oc,ctx['reg'],sha(cfg['owner_config']))
    require(oc['registration']==cfg['registration'] and all(s['resource']=='cpu' for s in oc['stages']),
            'Only bounded CPU stages may share this execution revision')
    require([s['id'] for s in oc['stages']]==[cfg['phase'],'report'],
            'One phase then merge per independent owner; no concurrent whole/partial owners')
    base=Path(oc['owner_out']);ident=read(base/'owner_identity.json')
    require(a.same_identity(ident,a.identity(os.getppid())) and ident['registration_sha256']==sha(cfg['registration'])
            and ident['config_sha256']==sha(cfg['owner_config']),'Registered live owner is not parent')
    stageid=cfg['phase'] if worker is not None else 'report'
    stages=[s for s in oc['stages'] if s['id']==stageid];require(len(stages)==1,'Wrong owner stage')
    stage=stages[0];require(len(stage['jobs'])==(2 if worker is not None else 1),'Worker concurrency scope differs')
    suffix=['--stage','worker','--worker-index',str(worker)] if worker is not None else ['--stage','merge']
    matches=[j for j in stage['jobs'] if j['argv'][1:]==['-B',str(Path(__file__).absolute()),'--config',str(Path(config_path).absolute()),*suffix]]
    require(len(matches)==1,'Actual worker argv not registered')
    j=matches[0];lp=base/'stages'/stageid/'workers'/j['id']/'launch.json';launch=read(lp)
    require(a.same_identity(launch['identity'],a.identity(os.getpid())) and launch['registration_sha256']==sha(cfg['registration'])
            and launch['argv']==j['argv'] and launch['resource']=='cpu' and launch['threads']==2,'Live worker identity differs')
    slot=worker if worker is not None else 0
    require(cfg['cpu_affinities']==oc['cpu_affinities'] and set(launch['affinity'])==set(cfg['cpu_affinities'][slot])
            and set(os.sched_getaffinity(0))==set(launch['affinity']) and os.getpriority(os.PRIO_PROCESS,0)==15,
            'Owner/config/launch/actual CPU affinity or priority differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'CPU-only numerical environment differs')
    return {str(base/'owner_identity.json'):sha(base/'owner_identity.json'),str(lp):sha(lp)}


def trace_inventory(ctx,paths,worker=None):
    wanted=set(indices(worker) if worker is not None else range(COUNT));phase=ctx['cfg']['phase'];core=ctx['core']
    entries={r['full_slot']:r for r in ctx['context']['schedule'] if r['phase']==phase}
    seen,events,energies=set(),{},[];gray_count=pending=0
    for p in paths:
        source=read(p);i=source['source_index']
        require(i in wanted and i not in seen and source['status']=='H_FULL_CPU_SOURCE_TRACES'
                and source['phase']==phase and source['source_id']==ctx['context']['source_ids'][i]
                and source['registration_sha256']==sha(ctx['cfg']['registration']),'Source trace identity differs')
        seen.add(i);verify(source['source_bindings']);keys=set()
        for f in source['frames']:
            slot,seed=f['full_slot'],f['noise_seed'];require(slot in entries and seed in core.SEEDS and (slot,seed) not in keys,'Wrong/duplicate frame')
            keys.add((slot,seed));c=entries[slot]['candidate']
            event=f'H:{phase}:{c["candidate_id"]}:snr{c["snr_db"]}:src{i:04d}:seed{seed}'
            require(f['status']=='H_FULL_CPU_FRAME_TRACE' and f['stage']==phase and f['event_id']==event
                and f['source_id']==source['source_id'] and f['source_index']==i and f['candidate_id']==c['candidate_id']
                and f['arm']==c['arm'] and f['snr_db']==c['snr_db'] and f['candidate_slot']==c['slot']
                and f['public_frame_counter']==core.frame_counter(slot,i,seed) and f['scrambling_session']==core.SESSION
                and f['noise_stage']==core.NOISE_STAGE and f['total_symbols']==1024 and f['frame_normalized'] is False
                and f['execution_registration_sha256']==sha(ctx['cfg']['registration']), 'Full frame trace changed')
            rx=f['rx'];parsed=bool(rx['body'] and rx['body'].get('parser_accepted'))
            rp=ctx['context']['book'][str(rx['header']['profile_id'])] if rx['header']['header_ok'] else None
            require(f['rx_profile']==rp and (not parsed or rp is not None),'Receiver profile is not actual paid header')
            if not parsed:status,expected_gray,tokens='WIRE_REJECT_GRAY',True,None
            elif rp['mode']=='raw':status,expected_gray,tokens='RAW_SOURCE_DECODED',False,core.phy.raw_tokens(rx['body']['payload'],rp).tolist()
            else:status,expected_gray,tokens='ARITHMETIC_CANONICAL_RX_REQUIRED',None,None
            require(f['rx_source_status']==status and f['gray'] is expected_gray and f['received_raw_tokens']==tokens
                    and f['arithmetic_source_decode_run'] is False and f['neural_metrics_run'] is False,
                    'CPU trace misstates actual receiver parse/canonical status')
            names=['header']+(['body'] if f['rx']['body'] is not None else [])
            require(f['logical_packet_events']==len(names) and f['packet_event_ids']==[event+':'+k for k in names],'Paid event references differ')
            for k in names:
                eid=event+':'+k;require(eid not in events,'Duplicate paid event');events[eid]=dict(kind=k,result_sha256=digest(f['rx'][k]))
            energies.append(float(f['total_energy']));gray_count+=int(f['gray'] is True)
            pending+=int(f['rx_source_status']=='ARITHMETIC_CANONICAL_RX_REQUIRED')
        require(keys=={(s,n) for s in entries for n in core.SEEDS},'Missing source frames')
    require(seen==wanted,'Missing full calibration sources')
    import numpy as np
    energy=np.asarray(energies);require(np.isfinite(energy).all() and np.all(energy>0),'Invalid actual frame energies')
    return events,dict(source_count=len(wanted),frame_count=len(energies),logical_packet_events=len(events),
        wire_gray_frames=gray_count,arithmetic_pending_frames=pending,energy=dict(mean=float(energy.mean()),
            min=float(energy.min()),max=float(energy.max()),sd_population=float(energy.std())))


def audit_ledger(path,expected,phase,worker=None):
    seen=set()
    with contextlib.closing(sqlite3.connect(str(path),timeout=60)) as db:
        for eid,kind,status,result,checksum in db.execute('SELECT event_id,kind,status,result,result_sha FROM events WHERE phase=?',(phase,)):
            if worker is not None and eid not in expected:
                import re
                match=re.search(r':src(\d{4}):seed\d+:',eid)
                require(match is not None and int(match.group(1))<1000 and int(match.group(1))%2!=worker,'Unexpected worker event');continue
            require(eid in expected and eid not in seen and status=='COMPLETE' and kind==expected[eid]['kind']
                    and checksum==expected[eid]['result_sha256'] and digest(json.loads(result))==checksum,'Paid result differs/incomplete')
            seen.add(eid)
        require(seen==set(expected),'Missing actual paid event')
        if worker is None:
            row=db.execute('SELECT charged FROM counters WHERE phase=?',(phase,)).fetchone()
            require(row and row[0]==len(expected)<=PHASE_LIMITS[phase],'Phase ledger charge differs')
    return dict(status='FULL_CPU_LEDGER_TRACE_MATCH',paid_events=len(seen),event_ids_sha256=digest(sorted(seen)),all_complete=True)


def boundary(cfg,started,out):
    require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP at packet boundary')
    require(time.monotonic()-started<cfg['max_worker_seconds'] and time.time()<DEADLINE,'Original H deadline exhausted')


def run(config_path,worker):
    ctx=load_registered(config_path);cfg=ctx['cfg'];phase=cfg['phase'];out=Path(cfg['out'])/(f'worker_{worker}' if worker is not None else '')
    out.mkdir(parents=True,exist_ok=True);require(not (out/'failure.json').exists(),'Prior failure preserved; no automatic retry')
    import fcntl
    with (out/'execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (out/'attempt.json').exists(),'Prior attempt preserved; no implicit resume')
        regsha=sha(cfg['registration']);started=time.monotonic()
        try:
            configure(ctx,worker if worker is not None else 0)
            launch=verify_live_owner(ctx,config_path,worker);boundary(cfg,started,out)
            ledger=make_ledger(ctx)
            atomic(out/'attempt.json',dict(status='STARTED',phase=phase,registration_sha256=regsha,worker_index=worker,
                                          identity=ledger.worker,automatic_retry=False))
            outputs={};paths=[];identities=[]
            if worker is not None:
                backend,header=build_runtime(ctx)
                entries=[r for r in ctx['context']['schedule'] if r['phase']==phase]
                with (out/'traces.journal.jsonl').open('x',encoding='utf-8',newline='\n') as journal:
                    for i in indices(worker):
                        assets=load_source(ctx,i);sid=ctx['context']['source_ids'][i];frames=[]
                        for entry in entries:
                            for seed in ctx['core'].SEEDS:
                                boundary(cfg,started,out)
                                f=ctx['core'].run_frame(ctx=ctx['context'],full_slot=entry['full_slot'],source_id=sid,
                                    source_index=i,noise_seed=seed,scales=assets['scales'],arithmetic_bits=assets['arithmetic_bits'],
                                    backend=backend,header=header,ledger=ledger)
                                journal.write(canonical(f)+'\n');journal.flush();os.fsync(journal.fileno());frames.append(f)
                        p=out/'traces'/f'{i:04d}.json';atomic(p,dict(status='H_FULL_CPU_SOURCE_TRACES',phase=phase,
                            registration_sha256=regsha,source_index=i,source_id=sid,frames=frames,source_bindings=assets['source_bindings']))
                        cp=out/'source_checkpoints'/f'{i:04d}.json';atomic(cp,dict(status='H_FULL_CPU_SOURCE_COMPLETE',phase=phase,
                            registration_sha256=regsha,source_index=i,source_id=sid,frame_count=len(frames),outputs={str(p):sha(p)},images_scored=False))
                        outputs.update({str(p):sha(p),str(cp):sha(cp)});paths.append(p)
                        atomic(out/'status.json',dict(status='RUNNING',phase=phase,completed_sources=len(paths),total_sources=500,
                            elapsed_seconds=time.monotonic()-started,registration_sha256=regsha))
                outputs[str(out/'traces.journal.jsonl')]=sha(out/'traces.journal.jsonl')
                identities=[ledger.worker]
            else:
                for w in (0,1):
                    folder=Path(cfg['out'])/f'worker_{w}';require(not (folder/'failure.json').exists(),'Failed worker cannot merge')
                    cp=folder/'completion.json';done=checked(cp,'H_FULL1000_CPU_WORKER_COMPLETE')
                    require(done['phase']==phase and done['registration_sha256']==regsha and done['worker_index']==w
                            and done['source_indices']==indices(w) and done['driver_config_sha256']==sha(config_path)
                            and done['budget_registration_sha256']==sha(cfg['budget_registration'])
                            and done['source_decode_complete'] is False,'Worker completion scope differs')
                    for i in indices(w):
                        p=folder/'traces'/f'{i:04d}.json';require(done['outputs'].get(str(p))==sha(p),'Worker trace unsealed');paths.append(p)
                    outputs.update(done['outputs']);outputs[str(cp)]=sha(cp);identities.extend(done['worker_identities'])
            expected,summary=trace_inventory(ctx,paths,worker);audit=audit_ledger(cfg['ledger'],expected,phase,worker)
            verify(ctx['reg']['source_bindings']);verify(ctx['reg']['input_bindings']);boundary(cfg,started,out)
            done=dict(status='H_FULL1000_CPU_WORKER_COMPLETE' if worker is not None else 'H_FULL1000_CPU_RECEIVE_COMPLETE',
                phase=phase,registration_sha256=regsha,driver_config_sha256=sha(config_path),worker_index=worker,workers=2,
                source_indices=indices(worker) if worker is not None else list(range(1000)),worker_identities=identities,
                budget_registration_sha256=sha(cfg['budget_registration']),source_completion_sha256=sha(cfg['source_completion']),
                selected_sha256=sha(cfg['selected']),partial_reference_sha256=sha(cfg['partial_reference']),
                maximum_packet_calls=PHASE_LIMITS[phase],ledger_audit=audit,outputs=outputs,**summary,
                source_bindings=ctx['reg']['source_bindings'],input_bindings=ctx['reg']['input_bindings'],
                predecessor_closure_bindings=ctx['predecessor_bindings'],live_launch_bindings=launch,
                images_scored=False,source_decode_complete=False,arithmetic_source_decode_complete=False,GPU_used=False,
                development_used=False,holdout_used=False,policy_selection=False,full1000_calibration_complete=False,
                requires_owner_worker_exit_receipt_before_visual_stage=True,initial_physical_results_reused=False)
            done['requires_both_cpu_groups_closed_before_visual_stage']=True
            done['visual_stage_started']=False
            if worker is None:
                snapshot=ledger.assert_quiescent();require(snapshot['phase_charged'][phase]==len(expected),'Merge ledger changed')
                done['ledger']=snapshot
            atomic(out/'completion.json',done);return done
        except BaseException as error:
            if not (out/'failure.json').exists():atomic(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',phase=phase,
                registration_sha256=regsha,error=repr(error),traceback=traceback.format_exc(),ledger_never_refunded=True))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=('worker','merge'),required=True);p.add_argument('--worker-index',type=int)
    a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    require((a.stage=='worker' and a.worker_index in (0,1)) or (a.stage=='merge' and a.worker_index is None),'Wrong worker/merge arguments')
    result=run(a.config,a.worker_index);print(json.dumps({k:result[k] for k in ('status','phase','registration_sha256')}))


if __name__=='__main__':main()
