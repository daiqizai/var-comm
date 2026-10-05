"""Read-only closure gate and receiver adapter for the two full1000 CPU groups.

No PHY call, policy choice, model construction or job launch occurs here.
Reconstruction delegates to the unchanged actual-RX receiver and its exact-input
cache. Full target pixels are used only by the separate scoring function.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

PHASES=('whole_calibration','partial_calibration')
SEEDS=(6101,6102,6103)
PHASE_FRAMES={'whole_calibration':24000,'partial_calibration':18000}
FINAL={'RAW_SOURCE_DECODED','ARITHMETIC_SOURCE_DECODED','WIRE_REJECT_GRAY','ARITHMETIC_SOURCE_INVALID_GRAY'}


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Changed bound file: '+p)
def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting SHA: '+p);result[p]=s
    return result
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def verify_batch_closed(spec,phase,driver,owner,wait):
    """Both completed CPU workers and the merge/owner must be actually reaped."""
    ctx=driver.load_registered(spec['config']);cfg=ctx['cfg'];reg=read(spec['registration'])
    oc,launch,done=read(spec['owner_config']),read(spec['launch']),read(spec['completion'])
    owner.validate_config(oc,reg,sha(spec['owner_config']));ident=launch['identity']
    require(cfg['phase']==phase and cfg['registration']==spec['registration'] and cfg['owner_config']==spec['owner_config']
        and oc['registration']==spec['registration'] and [s['id'] for s in oc['stages']]==[phase,'report']
        and all(s['resource']=='cpu' for s in oc['stages']) and Path(cfg['out'])/'completion.json'==Path(spec['completion']),
        'CPU full-calibration predecessor scope differs')
    require(launch['argv']==ident['argv'] and launch['argv'][1:]==['-B',cfg['owner_module'],'--config',spec['owner_config']]
        and launch['registration_sha256']==sha(spec['registration']) and launch['owner_config_sha256']==sha(spec['owner_config']),
        'CPU owner launch identity differs')
    owner_identity=read(Path(oc['owner_out'])/'owner_identity.json')
    require(owner.same_identity(owner_identity,ident) and owner_identity['registration_sha256']==sha(spec['registration'])
        and owner_identity['config_sha256']==sha(spec['owner_config']),'CPU owner identity/config differs')
    for p in (Path(oc['out'])/'STOP',Path(oc['owner_out'])/'STOP',Path(cfg['out'])/'STOP',
              Path(cfg['out'])/'worker_0/STOP',Path(cfg['out'])/'worker_1/STOP'):
        require(not p.exists(),'STOP blocks visual continuation: '+str(p))
    require(not (Path(oc['owner_out'])/'registration_failure.json').exists(),'CPU predecessor registration failed')
    require(wait.exited(ident,owner.raw_process_state),'CPU owner remains live/unreaped')
    closure=wait.verify_batch(owner,spec['owner_config'],spec['registration'],ident['uid'],owner.raw_process_state)
    require(closure['bindings'].get(spec['completion'])==sha(spec['completion']),'CPU merge not sealed by successful owner')
    expected=dict(status='H_FULL1000_CPU_RECEIVE_COMPLETE',phase=phase,source_count=1000,frame_count=PHASE_FRAMES[phase],
        worker_index=None,workers=2,GPU_used=False,images_scored=False,source_decode_complete=False,
        arithmetic_source_decode_complete=False,development_used=False,holdout_used=False,policy_selection=False,
        requires_both_cpu_groups_closed_before_visual_stage=True,visual_stage_started=False,initial_physical_results_reused=False)
    for k,v in expected.items():require(done.get(k)==v,'CPU completion field differs: '+k)
    require(done['registration_sha256']==sha(spec['registration']) and done['driver_config_sha256']==sha(spec['config'])
        and done['budget_registration_sha256']==sha(cfg['budget_registration'])
        and done['source_completion_sha256']==sha(cfg['source_completion'])
        and done['selected_sha256']==sha(cfg['selected']) and done['partial_reference_sha256']==sha(cfg['partial_reference']),
        'CPU completed input binding differs')
    for values in (done['outputs'],done['source_bindings'],done['input_bindings']):verify(values)
    require(len(done['worker_identities'])==2,'Two CPU worker exit proofs required')
    for child in done['worker_identities']:
        require(any(owner.same_identity(child,x) for x in closure['child_identities'])
            and wait.exited(child,owner.raw_process_state),'CPU packet worker missing from closed owner or still present')
    paths=[Path(cfg['out'])/f'worker_{i%2}'/'traces'/f'{i:04d}.json' for i in range(1000)]
    require(all(done['outputs'].get(str(p))==sha(p) for p in paths),'Full1000 trace coverage is not sealed')
    events,inventory=driver.trace_inventory(ctx,paths)
    audit=driver.audit_ledger(cfg['ledger'],events,phase)
    require(audit==done['ledger_audit'] and inventory['frame_count']==PHASE_FRAMES[phase]
        and inventory['source_count']==1000,'Actual paid CPU events changed after completion')
    closure['bindings']=merge(closure['bindings'],{str(Path(oc['owner_out'])/'owner_identity.json'):sha(Path(oc['owner_out'])/'owner_identity.json')})
    return dict(context=ctx,cfg=cfg,done=done,closure=closure,owner_identity=ident,audit=audit)


def load_registered(config_path):
    """Pure CPU gate for a later, separately supervised single-GPU entry point."""
    path=Path(config_path).resolve();cfg=read(path);reg=read(cfg['registration'])
    require(cfg['schema']=='H_FULL1000_RX_CONFIG_V1' and reg['status']=='H_EXECUTION_REVISION_REGISTERED'
        and reg['branch']=='H' and reg['allowed_stage_ids']==['render']
        and reg['source_stage_scope']=='FULL1000_ACTUAL_RX_IMAGES_ONLY','Independent full1000 visual registration required')
    bound=merge(reg['input_bindings'],reg['source_bindings']);verify(bound)
    for key in ('owner_module','wait_module','cpu_driver_module','receiver_module','cache_module','source_driver_module',
                'protocol','budget_registration','calibration_registration','numerical_reference','visual_owner_config',
                'static_closure_module','visual_driver_module'):
        require(bound.get(cfg[key])==sha(cfg[key]),'Required full RX input/source unbound: '+key)
    require(bound.get(str(path))==sha(path) and bound.get(str(Path(__file__).resolve()))==sha(__file__),'Full RX config/adapter unbound')
    require(set(cfg['cpu_batches'])==set(PHASES),'Both full CPU groups must close before visual stage')
    owner=module(cfg['owner_module'],'full_rx_bound_owner');wait=module(cfg['wait_module'],'full_rx_bound_wait')
    driver=module(cfg['cpu_driver_module'],'full_rx_bound_cpu_driver');groups={}
    for phase in PHASES:
        spec=cfg['cpu_batches'][phase]
        require(set(spec)=={'config','registration','owner_config','launch','completion'},'Unexpected predecessor schema')
        for p in spec.values():require(bound.get(p)==sha(p),'CPU predecessor input unbound: '+p)
        groups[phase]=verify_batch_closed(spec,phase,driver,owner,wait)
    first,second=(groups[p] for p in PHASES)
    for key in ('root','runtime_dir','protocol','budget_registration','ledger','phase_limits','source_completion','asset_completion',
                'asset_manifest','selected','partial_reference','catalogue','shortlist','owner_module','wait_module'):
        require(first['cfg'][key]==second['cfg'][key],'Whole/partial inputs differ: '+key)
    for key in ('protocol','budget_registration','ledger','phase_limits','root','runtime_dir','owner_module','wait_module'):
        require(cfg[key]==first['cfg'][key],'Visual and physical registration differ: '+key)
    require(first['context']['context']['schedule']==second['context']['context']['schedule'], 'Whole/partial public policy schedule differs')
    schedule=first['context']['context']['schedule'];require(len(schedule)==14 and [r['full_slot'] for r in schedule]==list(range(14)),
        'Full fourteen-policy schedule incomplete')
    require(cfg['phase_limits']==reg['phase_limits'],'Visual phase quotas differ')
    before=owner.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
    require(before==reg['budget_before'] and before['created'] is True and before['unresolved']==0,
        'Packet ledger changed since full visual registration')
    for phase in PHASES:require(before['phase_charged'][phase]==groups[phase]['done']['ledger_audit']['paid_events'],
        'Final paid phase count differs')
    cal=read(cfg['calibration_registration']);ids=first['context']['context']['source_ids']
    require(list(ids)==cal['source_ids'] and cal['stage']==cal['calibration_or_development']=='m1_calibration',
        'Full visual population must be the original calibration1000')
    for p,s in first['context']['source']['source_bindings'].items():require(bound.get(p)==s,'Frozen source-codec dependency unbound: '+p)
    numerical=read(cfg['numerical_reference']);require(numerical['status']=='REAL_NATIVE_QUALIFICATION_PASS','Original native qualification absent')
    closure=module(cfg['static_closure_module'],'full_rx_static_visual_closure')
    actual=closure.collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])
    require(closure.compare_bindings(actual,bound)['status']=='EXACT_SOURCE_CLOSURE_MATCH','Full static visual source closure differs')
    receiver=module(cfg['receiver_module'],'full_rx_frozen_receiver');cache=module(cfg['cache_module'],'full_rx_frozen_cache')
    return dict(cfg=cfg,reg=reg,bound=bound,owner=owner,wait=wait,cpu=driver,groups=groups,source_ids=list(ids),schedule=schedule,
        public_context=first['context']['context'],core=first['context']['core'],before=before,calibration=cal,
        expected_flags=numerical['numerical_runtime'],static_bindings=actual,rx_api=receiver,cache_api=cache,
        catalogue=first['context']['context']['catalogue'])


def load_source(ctx,index):
    require(type(index) is int and 0<=index<1000,'Only calibration1000 indices allowed')
    sid=ctx['source_ids'][index];sources={};inputs={}
    for phase in PHASES:
        group=ctx['groups'][phase];base=Path(group['cfg']['out'])/f'worker_{index%2}'
        cp_path=base/'source_checkpoints'/f'{index:04d}.json';tp=base/'traces'/f'{index:04d}.json'
        require(group['done']['outputs'].get(str(cp_path))==sha(cp_path) and group['done']['outputs'].get(str(tp))==sha(tp),
            'Actual full CPU source evidence changed')
        cp,source=read(cp_path),read(tp)
        require(cp['status']=='H_FULL_CPU_SOURCE_COMPLETE' and source['status']=='H_FULL_CPU_SOURCE_TRACES'
            and cp['phase']==source['phase']==phase and cp['source_id']==source['source_id']==sid
            and cp['source_index']==source['source_index']==index and cp['registration_sha256']==source['registration_sha256']==group['done']['registration_sha256']
            and cp['outputs']=={str(tp):sha(tp)} and cp['frame_count']==len(source['frames'])==PHASE_FRAMES[phase]//1000,
            'Full CPU trace source/phase/registration identity differs')
        verify(source['source_bindings']);inputs=merge(inputs,source['source_bindings'],{str(p):sha(p) for p in (cp_path,tp)});sources[phase]=source
    group=ctx['groups'][PHASES[0]]['context'];row=group['manifest']['records'][index];ap=Path(row['checkpoint'])
    require(group['assets']['outputs'].get(str(ap))==sha(ap)==row['checkpoint_sha256'],'Target checkpoint unsealed')
    acp=read(ap);require(acp['source_index']==index and acp['source_id']==sid and acp['preprocessing_id']==ctx['calibration']['preprocessing_ids'][index]
        and acp['archive']==row['archive'],'Original target/preprocessing identity differs')
    verify(acp['outputs']);require(all(group['assets']['outputs'].get(p)==s for p,s in acp['outputs'].items()),'Target archive output unsealed')
    with np.load(acp['archive'],allow_pickle=False) as z:target=z['pixels'].copy()
    require(target.dtype==np.uint8 and target.shape==(3,256,256)
        and hashlib.sha256(target.tobytes()).hexdigest()==acp['preprocessing_id'],'Actual original target preprocessing differs')
    inputs=merge(inputs,{str(ap):sha(ap),acp['archive']:sha(acp['archive'])})
    return sources,target,acp['preprocessing_id'],inputs


def validate_frame_grid(sources,schedule,core):
    require(set(sources)==set(PHASES),'Both whole and partial actual received groups are required')
    first=sources[PHASES[0]];sid,index=first['source_id'],first['source_index']
    rows={r['full_slot']:r for r in schedule};require(len(rows)==14 and set(rows)==set(range(14)),'Missing/duplicate full slots')
    expected={(slot,seed) for slot in rows for seed in SEEDS};seen=set();frames=[]
    for phase in PHASES:
        source=sources[phase]
        require(source['phase']==phase and source['source_id']==sid and source['source_index']==index,'Full received source grouping differs')
        for frame in source['frames']:
            slot,seed=frame['full_slot'],frame['noise_seed'];require((slot,seed) in expected and (slot,seed) not in seen,
                'Missing/duplicate/unregistered full frame');seen.add((slot,seed));entry=rows[slot];c=entry['candidate']
            require(frame['status']=='H_FULL_CPU_FRAME_TRACE' and frame['stage']==entry['phase']==phase and frame['source_id']==sid
                and frame['source_index']==index and frame['candidate_id']==c['candidate_id'] and frame['candidate_slot']==c['slot']
                and frame['arm']==c['arm'] and frame['snr_db']==c['snr_db']
                and frame['public_frame_counter']==core.frame_counter(slot,index,seed) and frame['scrambling_session']==core.SESSION
                and frame['noise_stage']==core.NOISE_STAGE and frame['execution_registration_sha256']==source['registration_sha256'],
                'Actual full frame/public scheduling identity differs')
            frames.append(frame)
    require(seen==expected,'Incomplete actual full source grid')
    return sorted(frames,key=lambda f:(f['full_slot'],f['noise_seed']))


def render_source(sources,target,target_sha,schedule,core,receiver,rx_api,cache_api,cache_identity,boundary):
    frames=validate_frame_grid(sources,schedule,core);entries={r['full_slot']:r for r in schedule}
    cache=cache_api.CachedReceiver(receiver,cache_identity);rows=[];images={};bysha={}
    for frame in frames:
        boundary();entry=entries[frame['full_slot']];c=entry['candidate']
        # This is the only input to reconstruction. Every cache hit re-runs
        # Receiver._wire on this actual frame before reusing deterministic output.
        result=cache.reconstruct(rx_api.receiver_view(frame),frame['event_id'])
        s=result['summary'];require(s['source_decode_complete'] is True and s['source_status'] in FINAL,
            'Unresolved actual arithmetic state cannot be scored')
        scores=rx_api.score_reconstruction(result,target,expected_preprocessing_sha256=target_sha)
        diagnostic=rx_api.evaluation_diagnostics(result,frame['evaluation_only'])
        ih=s['image_sha256']
        if ih not in bysha:
            key=f'image_{len(images):04d}';bysha[ih]=key;images[key]=result['image'].copy()
        else:
            key=bysha[ih];require(np.array_equal(images[key],result['image']),'RGB hash collision')
        rows.append(dict(phase=entry['phase'],full_slot=entry['full_slot'],slot=c['slot'],candidate_id=c['candidate_id'],arm=c['arm'],
            target_m=c['target_m'],K=c.get('K',0),q=c['q'],nominal_rate=c['nominal_rate'],snr_db=c['snr_db'],
            source_id=frame['source_id'],source_index=frame['source_index'],noise_seed=frame['noise_seed'],event_id=frame['event_id'],
            public_frame_counter=frame['public_frame_counter'],noise_stage=core.NOISE_STAGE,**scores,
            source_status=s['source_status'],gray=s['gray'],receiver_view_sha256=s['receiver_view_sha256'],image_key=key,
            received_m=s['received_m'],received_K=s['received_K'],received_mode=s['received_mode'],received_profile_id=s['received_profile_id'],
            rx_summary=s,evaluation_only=diagnostic))
    costs=dict(frames=len(rows),unique_images=len(images),receiver_cache_hits=cache.hits,receiver_cache_misses=cache.misses,
        arithmetic_canonical_calls=sum(r['rx_summary']['arithmetic_canonical_executed_this_frame'] for r in rows),
        suffix_renderer_calls=sum(r['rx_summary']['suffix_renderer_executed_this_frame'] for r in rows),gray_frames=sum(r['gray'] for r in rows),
        new_packet_decodes=0,policy_selection=False)
    return rows,images,costs
