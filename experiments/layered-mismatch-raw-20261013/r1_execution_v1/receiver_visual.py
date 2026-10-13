"""R1 CPU receive projection and isolated, finite, received-state-only rendering.

Import, bundle building, projection and synthetic tests perform no model calls.
Only the explicit ``visual`` command may construct frozen visual models.  The
visual registration and model input list contain no source or TX lookup paths.
"""
from __future__ import annotations
import argparse
import ast
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import shared as s

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
OFFSETS = tuple(sum(x*x for x in SIZES[:i]) for i in range(11))
MODELS = dict(vae='6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b',
    var='b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5',
    decoder='bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1')
RUNTIME = dict(python='3.10.12', torch='2.11.0+cu128', numpy='2.2.6', cuda='12.8',
               cudnn=91900, threads=6, interop_threads=2)
MATH = {
    'loader': ('src/var_comm/next_scale_prior.py', '568217086c94ee7607abc256415d705a11702ffcaaed9c79ca04d5e9a62da562'),
    'decoder': ('experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py', 'd556732b86cb83499c8912a2412a8858f599e1ddd58f28418a0f1889d2b705a1'),
    'prior': ('experiments/scale-causal-partial-residual-20261002/partial_receiver.py', '46b676720283e3d7e90ef38bab4f4ea102ae51f38f414913d613ccb4b7e01cdc'),
    'render': ('experiments/content-real-64qam-20261006/h_source_driver.py', 'b38f8451ca2d94a8ceb4381df3647e327dba38e3d4cf969aa15597bc47c342ce')}
WEIGHTS = dict(vae='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c',
    var='4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb',
    decoder='e3d757a5a16206f5f9be345d0e6330b0f1fd6c78bf167670e70378b429d931c4')
DIST_SHA = 'b55322b4ae4c8c2b600b98870dc98b26435a813b72b3468a4315d7d74771e370'
QUALIFIED_REQUEST_SHA = '5e02226b1249131e3984a55540cadba5dad0883d61333e74749cf25fdcebd675'
MODEL_IDENTITY = s.digest(dict(models=MODELS, numeric_flags=s.FLAGS, runtime=RUNTIME,
    prior='unconditional_class1000', completion='deterministic_argmax', batch_size=1,
    math={k:v[1] for k,v in MATH.items()}, input_order='raw_raster'))
MODEL_ROW_FIELDS = {'view_id', 'cache_context_key', 'observation_key', 'actual_rx_key',
    'receiver_state_sha256', 'input_state_sha256', 'state', 'model_identity'}
JOIN_FIELDS = ('source_id', 'source_index', 'selection_slot', 'snr_hat', 'snr_true',
               'noise_seed', 'candidate_id', 'physical_key', 'arms')
HEX = re.compile(r'^[0-9a-f]{64}$')


def require(ok, message):
    s.require(ok, message)


def array_sha(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def same_pin(a,b):
    return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('path','sha256'))


def normalize_state(state):
    """Strip all provenance; never repair accepted errors using clean tokens."""
    require(isinstance(state, dict) and state.get('kind') in ('gray', 'tokens'), 'Unknown actual receiver state')
    m, k = state.get('m'), state.get('K')
    require(type(m) is int and type(k) is int, 'Noninteger received endpoint')
    prefix, partial = state.get('prefix'), state.get('partial_values')
    require(isinstance(prefix, list) and isinstance(partial, list), 'Received arrays must be JSON lists')
    if state['kind'] == 'gray':
        require(m == k == 0 and prefix == partial == [] and state.get('rgb_fill', .5) == .5,
                'Failure output must be empty received state and fixed 0.5')
        require(state.get('order') in (None, 'raster'), 'Unknown gray order')
        return dict(kind='gray', m=0, K=0, order='raster', prefix=[], partial_values=[], rgb_fill=.5)
    require(1 <= m <= 10 and k >= 0 and (k == 0 if m == 10 else k < SIZES[m]**2),
            'Received endpoint outside the complete ten-scale domain')
    require(state.get('order') == 'raster' and len(prefix) == m and len(partial) == k,
            'Received raster shape differs')
    require(all(isinstance(scale, list) and len(scale) == SIZES[i]**2 for i, scale in enumerate(prefix)),
            'Received complete-scale lengths differ')
    flat = [x for scale in prefix for x in scale] + partial
    require(len(flat) == OFFSETS[m]+k and all(type(x) is int and 0 <= x < 4096 for x in flat),
            'Received tokens must be exact integers in [0,4095]')
    return dict(kind='tokens', m=m, K=k, order='raster', prefix=copy.deepcopy(prefix), partial_values=list(partial))


def validate_model_row(row):
    require(set(row) == MODEL_ROW_FIELDS, 'Model-facing row contains extra/missing fields')
    require(all(valid_sha(row[k]) for k in MODEL_ROW_FIELDS-{'state'}), 'Nonopaque model identity')
    require(row['model_identity'] == MODEL_IDENTITY, 'Frozen model or numeric identity changed')
    require(normalize_state(row['state']) == row['state'] and s.digest(row['state']) == row['input_state_sha256'],
            'Model-facing received input hash differs')
    return row


def actual_input_key(row):
    validate_model_row(row)
    # Rule/alias is intentionally absent: identical KEEP/DROP at one actual
    # event reuses an image, without allowing cross-source/event/model reuse.
    return s.digest([row['cache_context_key'], row['observation_key'], row['actual_rx_key'],
                     row['input_state_sha256'], row['model_identity']])


def project_one(logical, frame, protocol_sha):
    require(frame['status'] == 'COMPLETE' and frame['schema'] == 'LAYERED_RAW_CAL200_ACTUAL_PHY_V1', 'Incomplete actual PHY frame')
    identity = frame['input_identity']; source = identity['source']; rule = logical['rule']
    require(rule in ('KEEP', 'DROP') and rule in frame['receiver_views'], 'Missing actual receiver view')
    for key in ('source_id', 'source_index', 'selection_slot'):
        require(logical[key] == source[key], 'Logical/source identity mismatch')
    for key in ('snr_hat', 'snr_true', 'noise_seed', 'physical_key', 'profile_id', 'public_counter'):
        require(logical[key] == identity[key], 'Logical/physical identity mismatch: '+key)
    original = frame['receiver_views'][rule]
    received_sha = s.digest(original)
    require(received_sha == logical['receiver_state_sha256'] == frame['view_sha256'][rule], 'Actual receiver state hash mismatch')
    obs = {k:frame['observation_identity'][k] for k in ('waveform_sha256','standard_noise_sha256','received_sha256')}
    require(all(valid_sha(x) for x in obs.values()) and frame['actual_RX']['received_sha256'] == obs['received_sha256'],
            'Actual received observation mismatch')
    state = normalize_state(original)
    observation = s.digest(obs); rx = s.digest(frame['actual_RX'])
    # Pixel identity is bound opaquely by the CPU projector. The renderer never
    # receives the pixel hash, source path, class or any clean token hash.
    context = s.digest(dict(source_id=source['source_id'], source_index=source['source_index'],
        pixel_identity=source['preprocessing_id'], protocol=protocol_sha, candidate=logical['candidate_id'],
        physical_key=logical['physical_key'], noise_key=identity['noise_key'], counter=identity['public_counter'],
        snr_hat=identity['snr_hat'], snr_true=identity['snr_true'], seed=identity['noise_seed']))
    view_id = s.digest([context, observation, rx, rule])
    model = dict(view_id=view_id, cache_context_key=context, observation_key=observation, actual_rx_key=rx,
        receiver_state_sha256=received_sha, input_state_sha256=s.digest(state), state=state, model_identity=MODEL_IDENTITY)
    validate_model_row(model)
    join = {k:copy.deepcopy(logical[k]) for k in JOIN_FIELDS}
    join.update(view_id=view_id, receiver_rule=rule, observation_key=observation, actual_rx_key=rx,
        receiver_state_sha256=received_sha, input_state_sha256=model['input_state_sha256'],
        model_identity=MODEL_IDENTITY, actual_RX=copy.deepcopy(logical['actual_RX']),
        actual_received_kind=state['kind'], actual_received_m=state['m'], actual_received_K=state['K'],
        actual_failure_state=(state['kind']=='gray'), source_truth_supplied_to_RX=False,
        nomination_indices=copy.deepcopy(logical['nomination_indices']))
    return model, join


def require_successful_wait(directory):
    wait_path = Path(directory)/'actual_wait.json'; w = s.read(wait_path)
    require(w.get('actual_wait') is True and w.get('returncode') == 0 and not w.get('timed_out', False) and not w.get('stopped',False),
            'Actual successful child wait required: '+str(directory))
    pid = w.get('pid', w.get('process',{}).get('pid'))
    require(type(pid) is int and pid > 0, 'Wait lacks real child PID')
    live = s.process_identity(pid)
    ticks = w.get('start_ticks', w.get('process',{}).get('start_ticks'))
    if Path(directory).parent==s.TMP:
        root=Path(directory).parent;stage=Path(directory).name
        launch=s.read(root/(stage+'_launch.json'));owner=s.read(root/'owner_launch.json')
        require(s.read(root/(stage+'_actual_wait.json'))==w and launch['stage']==stage
            and same_pin(launch['registration'],owner['registration']) and launch['process']['pid']==pid
            and type(launch['process']['start_ticks']) is int and launch['process']['start_ticks']>0,
            'Actual wait does not close the registered dispatched child')
        require(ticks is None or ticks==launch['process']['start_ticks'],'Wait PID generation differs')
        ticks=launch['process']['start_ticks']
    require(live is None or live['state']=='Z' or (ticks is not None and live['start_ticks'] != ticks),
            'Completed child identity is still live')
    return s.pin(wait_path)


def packet_audit(journal, frame_keys):
    """Read immutable packet reservations/results; no decoder construction."""
    root = Path(journal); reserved = sorted((root/'packets').glob('*.reserved.json'))
    completed = list((root/'packets').glob('*.complete.json'))
    require(not list((root/'packets').glob('*.failure.json')), 'Packet failure scene cannot close')
    require({p.name[:-len('.reserved.json')] for p in reserved} == {p.name[:-len('.complete.json')] for p in completed},
            'Unresolved or extra packet completion')
    count = 0; rows = []; byframe = collections.Counter()
    for p in reserved:
        r = s.read(p); key = p.name[:-len('.reserved.json')]; f = r['frame']; request = r['request']
        require(f in frame_keys, 'Packet belongs to unregistered physical frame')
        raw = 'event' in request
        require(key == s.digest([f, request['event']]) if raw else key == s.digest([f, 'grouped', request]), 'Packet identity changed')
        decode = True if raw else request['direction'] == 'rx'
        cp = p.with_name(key+'.complete.json'); result = s.read(cp)
        if not raw:
            require(result['stage'] == 'done' and {k:v for k,v in result.items() if k not in ('stage','exception_type')} == request,
                    'Group completion differs from reserved actual call')
        count += int(decode); byframe[f] += int(decode)
        rows.append(dict(key=key, frame=f, counts_decode=decode, reservation=s.pin(p), completion=s.pin(cp)))
    require(0 < count <= 108000 and set(byframe) == set(frame_keys), 'Every physical frame needs actual paid packet decoding')
    return dict(packet_records=len(rows), packet_decodes=count, records_digest=s.digest(rows),
                per_frame_decode_counts=dict(byframe), unresolved=0)


def audit_phy(reg):
    """Deep CPU closure, including authentic complete source and canonical grids."""
    import r1_phy as phy
    prereg, sources, catalogue, groups, _ = phy.validate_registration(reg['phy_registration'])
    directory = Path(reg['runtime']['phy_directory']); require(directory == Path(prereg['runtime']['output_directory']), 'PHY location differs')
    actual_wait = require_successful_wait(directory)
    require(not (directory/'failure.json').exists(), 'A failed PHY attempt cannot become complete')
    done = s.read(directory/'completion.json')
    require(done['status']=='COMPLETE_R1_PHY_ONLY' and same_pin(done['registration'],reg['phy_registration']), 'PHY completion binding differs')
    physical = s.readpin(done['physical_frames']); logical = s.readpin(done['logical_frames'])
    require(len(physical)==prereg['expected_physical_frames']==reg['expected_physical_frames'] and
            len(logical)==prereg['expected_logical_var_images']==reg['expected_logical_var_images'], 'PHY complete grid counts differ')
    bysha={p['sha256']:p for p in physical}; require(len(bysha)==len(physical), 'Repeated physical frame')
    frames={}; keys=set(); expected_logical=[]; cursor=0; noise_hashes={}
    for source in sources:
        for group in groups:
            for true in phy.contract.true_snrs(group['snr_hat']):
                for seed in phy.contract.CAL_SEEDS:
                    p=physical[cursor]; cursor+=1; frame=s.readpin(p); identity=frame['input_identity']
                    rt=identity['runtime_identity']
                    require(same_pin(rt['registration'],reg['phy_registration']) and rt['source_code_pins']==prereg['pins'], 'Actual PHY runtime binding changed')
                    require(identity==phy.input_identity(source, group, true, seed, rt), 'Actual physical source/config/noise grid mismatch')
                    require(frame['status']=='COMPLETE', 'Physical attempt not complete')
                    nk=(source['source_id'],group['snr_hat'],seed)
                    observed_noise=frame['observation_identity']['standard_noise_sha256']
                    require(valid_sha(observed_noise) and noise_hashes.setdefault(nk,observed_noise)==observed_noise,
                            'Same source/hat/seed did not share the exact standard noise')
                    require(frame['channel_noise_variance_per_component']==10**(-float(true)/10),
                            'Actual true-SNR variance changed')
                    key=s.digest(identity); require(key not in keys, 'Physical identity duplicated');keys.add(key)
                    reservation=directory/'journal/frames'/f'{key}.reserved.json'
                    require(s.read(reservation)==dict(input_identity=identity,state='RESERVED') and
                            Path(p['path'])==directory/'journal/frames'/f'{key}.complete.json', 'Immutable physical journal does not match')
                    require(frame['receiver_views']==phy.receiver_views(frame['actual_RX'], 'single' if group['G']==1 else 'grouped',
                            sorted(v['rule'] for v in group['logical_receivers'])), 'KEEP/DROP not derived from the one actual RX event')
                    frames[p['sha256']]=frame
                    for view in group['logical_receivers']:
                        expected_logical.append(dict(source_id=source['source_id'],source_index=source['source_index'],selection_slot=source['selection_slot'],
                            snr_hat=group['snr_hat'],snr_true=true,noise_seed=seed,physical_key=group['physical_key'],profile_id=group['profile_id'],
                            public_counter=identity['public_counter'],**view,actual_RX=p,receiver_state_sha256=frame['view_sha256'][view['rule']],reused_complete_identity=False))
    require(logical==expected_logical, 'Canonical logical grid, arm aliases or order changed')
    jr=directory/'journal/frames'
    require({p.name[:-len('.reserved.json')] for p in jr.glob('*.reserved.json')}==keys and
            {p.name[:-len('.complete.json')] for p in jr.glob('*.complete.json')}==keys, 'Extra/unresolved physical reservation')
    packets=packet_audit(directory/'journal',keys); ledger=done['ledger']
    require(ledger==dict(physical_reserved=len(keys),physical_complete=len(keys),packet_decodes_reserved=packets['packet_decodes'],
        packet_decodes_complete=packets['packet_decodes'],unresolved_packets=0,physical_cap=len(keys),packet_decode_cap=108000), 'Actual packet/frame ledger differs')
    audit=dict(status='PASS_CPU_DEEP_PHY_CLOSURE',phy_completion=s.pin(directory/'completion.json'),actual_wait=actual_wait,
        physical_frames=done['physical_frames'],logical_frames=done['logical_frames'],physical_count=len(physical),logical_count=len(logical),
        packets=packets,exact_source_order=True,exact_canonical_grid=True,exact_counter_and_noise_namespace=True,
        same_standard_noise_across_true_and_candidates=True,standard_noise_keys=len(noise_hashes),
        all_failures_preserved=True,additional_scientific_calls=0)
    return audit, logical, frames


def project(registration, parent):
    reg=s.validate_registration(registration);require(reg['schema']==s.MASTER_SCHEMA,'CPU projection uses master registration')
    out=Path(reg['runtime']['projection_directory']);require(not out.exists(),'Projection already attempted');out.mkdir()
    guard=s.guard_for(reg,out,parent);guard()
    audit, logical, frames=audit_phy(reg); guard()
    model, join=[],[]
    for row in logical:
        a,b=project_one(row,frames[row['actual_RX']['sha256']],reg['phy_registration']['sha256']);model.append(a);join.append(b)
    require(len({r['view_id'] for r in model})==len(model),'Repeated projected logical identity')
    mp=s.write(out/'model_inputs.json',model);jp=s.write(out/'logical_join.json',join);ap=s.write(out/'audit.json',audit)
    # This small file is the only projection completion the model worker reads.
    vp=s.write(out/'visual_input_completion.json',dict(status='COMPLETE_RX_ONLY_PROJECTION',model_inputs=mp,
        logical_count=len(model),model_identity=MODEL_IDENTITY,projection_digest=s.digest([mp['sha256'],jp['sha256'],ap['sha256']]),
        source_truth_supplied_to_RX=False))
    guard()
    return s.write(out/'completion.json',dict(status='COMPLETE_R1_RX_PROJECTION',registration=registration,
        model_inputs=mp,logical_join=jp,audit=ap,visual_input_completion=vp,logical_count=len(model),scientific_calls=0))


def audit_projection(reg):
    """Closure/selector API: re-read authentic outputs, never rerender or score."""
    out=Path(reg['runtime']['projection_directory']);wait=require_successful_wait(out)
    done=s.read(out/'completion.json');require(done['status']=='COMPLETE_R1_RX_PROJECTION','Incomplete projection')
    audit,logical,frames=audit_phy(reg)
    saved=s.readpin(done['audit']); require(saved==audit,'CPU physical closure receipt differs')
    model=s.readpin(done['model_inputs']);join=s.readpin(done['logical_join'])
    expected=[project_one(r,frames[r['actual_RX']['sha256']],reg['phy_registration']['sha256']) for r in logical]
    require(model==[a for a,b in expected] and join==[b for a,b in expected],'Received projection or logical provenance changed')
    visual=s.readpin(done['visual_input_completion'])
    require(visual==dict(status='COMPLETE_RX_ONLY_PROJECTION',model_inputs=done['model_inputs'],logical_count=len(model),
        model_identity=MODEL_IDENTITY,projection_digest=s.digest([done['model_inputs']['sha256'],done['logical_join']['sha256'],done['audit']['sha256']]),
        source_truth_supplied_to_RX=False),'Model-only closure differs')
    return dict(status='PASS',completion=s.pin(out/'completion.json'),actual_wait=wait,audit=done['audit'],
        logical_join=done['logical_join'],model_inputs=done['model_inputs'],logical_count=len(model),additional_scientific_calls=0)


def audit_visual(reg):
    """CPU-only actual image/cache/call closure; safe to use before scoring."""
    import numpy as np
    import sqlite3
    out=Path(reg['runtime']['visual_directory']);wait=require_successful_wait(out)
    require(not (out/'failure.json').exists(),'Failed visual attempt cannot close')
    done=s.read(out/'completion.json');require(done['status']=='COMPLETE_R1_VISUAL' and same_pin(done['registration'],reg['visual_registration']),
        'Actual visual completion registration differs')
    pc=s.read(Path(reg['runtime']['projection_directory'])/'completion.json')
    require(done['model_inputs']==pc['model_inputs'] and done['projection_completion']==pc['visual_input_completion'],
        'Visual did not use the exact isolated projected input')
    inputs=s.readpin(done['model_inputs']);rows=s.readpin(done['image_rows'])
    require(len(inputs)==len(rows)==done['logical_rows']==reg['expected_logical_var_images'],'Incomplete actual visual logical grid')
    require(len({r['view_id'] for r in inputs})==len(inputs),'Duplicate projected logical identity')
    cache={};rendered=[];gray=0;reused=0
    for source,row in zip(inputs,rows):
        validate_model_row(source)
        require(set(row)=={'view_id','observation_key','actual_rx_key','receiver_state_sha256','input_state_sha256','model_identity',
            'reconstruction','reuse','status','actual_received_kind','source_truth_supplied_to_RX'},'Unreviewed visual output schema')
        require(all(row[k]==source[k] for k in ('view_id','observation_key','actual_rx_key','receiver_state_sha256','input_state_sha256','model_identity'))
            and row['status']=='COMPLETE' and row['source_truth_supplied_to_RX'] is False
            and row['actual_received_kind']==source['state']['kind'],'Actual image row or order changed')
        key=actual_input_key(source);require(row['reconstruction']['actual_input_key']==key,'Cache lacks exact actual input identity')
        if key in cache:
            old=cache[key];require(row['reconstruction']==old['reconstruction'] and row['reuse']==dict(reused=True,
                from_view_id=old['view_id'],reason='EXACT_ACTUAL_INPUT_IDENTITY'),'Invalid exact-view reuse')
            reused+=1
        else:
            image_pin=row['reconstruction']['image_archive']
            require(Path(image_pin['path'])==out/'images'/(key+'.npz'),'Image location is not the registered exact cache key')
            with np.load(s.checked(image_pin),allow_pickle=False) as z:
                require(z.files==['image'],'Actual output must contain only RGB');image=z['image']
            validate_image(image);require(array_sha(image)==row['reconstruction']['image_sha256'],'Actual RGB hash differs')
            isgray=source['state']['kind']=='gray'
            require(row['reuse']==dict(reused=False,from_view_id=None,reason='FIXED_GRAY' if isgray else 'ACTUAL_VAR_RENDER'),
                'False first-use/provenance marker')
            if isgray:require(np.all(image==np.float32(.5)),'Failed frame gray RGB changed');gray+=1
            else:rendered.append(source['view_id'])
            cache[key]=row
    counts=dict(logical_complete=len(rows),unique_actual_inputs=len(cache),exact_reuse=reused,fixed_gray=gray,
        var_render=len(rendered),model_load=int(bool(rendered)),source_image_reads=0,source_token_reads=0)
    require(done['counts']==counts,'Actual cache/logical/model accounting differs')
    ledger_path=s.checked(done['ledger_file']);require(ledger_path==out/'calls.sqlite','Unbound model call journal')
    db=sqlite3.connect(ledger_path.as_uri()+'?mode=ro',uri=True)
    try:calls=db.execute('SELECT id,kind,metadata,state FROM calls ORDER BY id').fetchall()
    finally:db.close()
    require(all(state=='COMPLETE' for _,_,_,state in calls),'Unresolved/failed visual model call')
    expected=[]
    if rendered:expected.append(('model_load',{}))
    for view in rendered:
        expected.append(('var_render',{'view_id':view}))
        for scale in range(10):expected.append(('prior_scale',{'view_id':view,'scale':scale}))
        expected.append(('decoder_forward',{'view_id':view}))
    require([(kind,json.loads(metadata)) for _,kind,metadata,_ in calls]==expected,'Actual nested model calls/order/scale grid differs')
    actual=collections.Counter(kind for _,kind,_,_ in calls);caps=dict(model_load=1,var_render=len(rows),prior_scale=10*len(rows),decoder_forward=len(rows))
    summary=dict(caps=caps,reserved={k:actual[k] for k in caps},completed={k:actual[k] for k in caps},unresolved=0)
    require(done['ledger']==summary and done['model_identity']==MODEL_IDENTITY and done['numeric_flags']==s.FLAGS
        and done['source_truth_supplied_to_RX'] is False and done['tokenizer_calls']==done['extra_qualification_calls']==0,
        'Numerical/model/call budget receipt differs')
    access=s.readpin(done['input_access_audit']);require(access['blocked']==[] and access['source_data_access_allowed'] is False
        and access['source_or_TX_paths_received'] is False,'Renderer source isolation failed')
    s.checked(done['resource_history']);s.checked(done['model_bundle'])
    return dict(status='PASS_CPU_ACTUAL_VISUAL_CLOSURE',completion=s.pin(out/'completion.json'),actual_wait=wait,
        image_rows=done['image_rows'],model_inputs=done['model_inputs'],ledger=done['ledger_file'],counts=counts,
        all_images_and_cache_keys_verified=True,actual_call_order_verified=True,scientific_calls=0)


def build_model_bundle(qualified_visual_request_pin, output_path):
    """Future CPU builder only. Source witnesses are verified here, never in GPU worker."""
    require(qualified_visual_request_pin['sha256']==QUALIFIED_REQUEST_SHA,'Original qualified visual request differs')
    request=s.readpin(qualified_visual_request_pin);spec=request['spec'];files=spec['files']
    # Authenticate all qualified input witnesses in this CPU process. Do not
    # export the source receipt, original source files, source labels or request.
    for p in files:s.checked(p)
    for p in spec['receipts'].values():s.checked(p)
    env=s.readpin(spec['environment']);s.checked(spec['host_native_addendum']);s.checked(spec['builder'])
    require(env['expected_runtime']==RUNTIME,'Qualified runtime projection differs')
    dep=spec['native_dependency_candidate'];s.checked(dep['audit']);s.checked(dep['commit_record'])
    def unique_sha(h):
        found=[p for p in files if p['sha256']==h];require(len(found)==1,'Ambiguous model-only pin')
        return {k:found[0][k] for k in ('path','sha256','bytes')}
    math={k:unique_sha(v[1]) for k,v in MATH.items()};weights={k:unique_sha(h) for k,h in WEIGHTS.items()}
    author=[{k:p[k] for k in ('path','sha256','bytes')} for p in files if Path(p['path']).parent==Path(dep['root'])/'models']
    require(len(author)==7 and {Path(p['path']).name for p in author}=={'__init__.py','basic_vae.py','basic_var.py','helpers.py','quant.py','var.py','vqvae.py'},'Original seven author modules required')
    dist=s.pin(Path(dep['root'])/'dist.py');require(dist['sha256']==DIST_SHA,'Reviewed upstream dist changed')
    runtime_files=[]
    for p in env['files']+env['projection_generated_files']+env['host_native_library_facts']:
        runtime_files.append({k:p[k] for k in ('path','sha256','bytes')})
    bundle=dict(schema='R1_MODEL_ONLY_BUNDLE_V1',models=MODELS,numeric_flags=s.FLAGS,expected_runtime=RUNTIME,
        math=math,weights=weights,author_files=author,author_root=dep['root'],dist=dist,model_identity=MODEL_IDENTITY,
        runtime_files=runtime_files,runtime_projection_sha256=spec['environment']['sha256'],
        qualification_request_sha256=qualified_visual_request_pin['sha256'],source_witnesses_checked_by_CPU_builder=True,
        conditioning='unconditional_class1000',completion='deterministic_argmax',batch_size=1,
        no_source_or_TX_paths=True,tokenizer_calls=0,extra_qualification_frames=0)
    result=s.write(output_path,bundle)
    s.write(Path(output_path).with_suffix('.cpu_provenance.json'),dict(model_bundle=result,qualified_request=qualified_visual_request_pin,
        qualified_source_witnesses_checked=len(files),science_calls=0,source_witnesses_not_exported_to_worker=True))
    return result


def validate_bundle(bundle):
    fields={'schema','models','numeric_flags','expected_runtime','math','weights','author_files','author_root','dist','model_identity',
        'qualification_request_sha256','source_witnesses_checked_by_CPU_builder','conditioning','completion','batch_size',
        'no_source_or_TX_paths','tokenizer_calls','extra_qualification_frames','runtime_files','runtime_projection_sha256'}
    require(set(bundle)==fields and bundle['schema']=='R1_MODEL_ONLY_BUNDLE_V1','Unexpected model bundle fields')
    require(bundle['models']==MODELS and bundle['numeric_flags']==s.FLAGS and bundle['expected_runtime']==RUNTIME and
        bundle['model_identity']==MODEL_IDENTITY and bundle['qualification_request_sha256']==QUALIFIED_REQUEST_SHA,
        'Model/math/numeric qualification differs')
    require(bundle['conditioning']=='unconditional_class1000' and bundle['completion']=='deterministic_argmax' and bundle['batch_size']==1
        and bundle['no_source_or_TX_paths'] is True and bundle['source_witnesses_checked_by_CPU_builder'] is True
        and bundle['tokenizer_calls']==bundle['extra_qualification_frames']==0,'Model conditioning or source isolation differs')
    require(set(bundle['math'])==set(MATH) and set(bundle['weights'])==set(WEIGHTS),'Missing model component')
    for k,p in bundle['math'].items():require(p['sha256']==MATH[k][1],'Mathematical source changed');s.checked(p)
    for k,p in bundle['weights'].items():require(p['sha256']==WEIGHTS[k],'Model weights changed');s.checked(p)
    require(bundle['dist']['sha256']==DIST_SHA,'Candidate dependency changed');s.checked(bundle['dist'])
    require(len(bundle['author_files'])==7,'Incomplete author code')
    for p in bundle['author_files']:
        require(Path(p['path']).parent==Path(bundle['author_root'])/'models','Author path not model-only');s.checked(p)
    require(bundle['runtime_projection_sha256']=='df9adc9cb2b13aa7f24137a7b1c95e1f548d0defe88ff9ef5deb4d51b79eede6',
            'Original runtime file projection changed')
    require(bundle['runtime_files'] and all(set(p)=={'path','sha256','bytes'} and valid_sha(p['sha256']) for p in bundle['runtime_files']),
            'Missing original software-runtime pins')
    return bundle


def definitions(binding,names,namespace):
    path=s.checked(binding);tree=ast.parse(path.read_text(encoding='utf8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes}==set(names),'Frozen mathematical definition missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return s.digest({n.name:ast.dump(n,include_attributes=False) for n in nodes})


def validate_image(image):
    import numpy as np
    require(isinstance(image,np.ndarray) and image.dtype==np.float32 and image.shape==(3,256,256)
        and np.isfinite(image).all() and image.min()>=0 and image.max()<=1,'Invalid actual RGB output')


def numeric_flags(torch):
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        precision=torch.get_float32_matmul_precision(),cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic=torch.are_deterministic_algorithms_enabled(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())


class IsolatedModels:
    """Only unchanged selected mathematical definitions and model checkpoints."""
    def __init__(self,bundle,reg,out,ledger,boundary,device):
        import numpy as np
        import torch
        self.t=torch;self.out=Path(out);self.boundary=boundary;self.ledger=ledger;self.context=None
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS workspace changed')
        require(all(os.environ.get(k)=='6' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Original six-thread launch required')
        device.check_cuda_environment();torch.set_num_threads(6);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One declared CUDA device required')
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction((16<<30)/props.total_memory,0)
        observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
        require(observed==RUNTIME and sorted(os.sched_getaffinity(0))==reg['runtime']['cpu_affinity'],'Qualified runtime/affinity differs')
        runtime_pins={str(Path(p['path']).resolve()):p for p in bundle['runtime_files']}
        imported=[]
        for path in (sys.executable,torch.__file__,torch._C.__file__,np.__file__):
            actual=str(Path(path).resolve());require(actual in runtime_pins,'Actual imported runtime file outside original projection')
            actual_pin=dict(runtime_pins[actual],path=actual);s.checked(actual_pin);imported.append(actual_pin)
        # Identity query only; no tensors or source files used by this helper.
        module=s.load(s.ES/'leo_whole_math_v1.py','_r1_identity_query')
        gpu=device.validate_cuda_device_identity(str(props.uuid),module.cuda_driver_pci_bus_id())
        s.write(self.out/'numerical_runtime.json',dict(observed=observed,flags=s.FLAGS,GPU=gpu,FP32=True,batch_size=1,
            model_identity=MODEL_IDENTITY,allocator_cap_bytes=16<<30,source_reads=0,tokenizer_calls=0,actual_runtime_files=imported))
        ns=dict(torch=torch,Path=Path,sys=sys,np=np,hashlib=hashlib,PATCH_NUMS=SIZES,VOCAB_SIZE=4096)
        proofs={'loader':definitions(bundle['math']['loader'],{'load_models','state_sha256'},ns)}
        self.state_sha=ns['state_sha256']
        require('dist' not in sys.modules and not any(n=='models' or n.startswith('models.') for n in sys.modules),'Fresh original author namespace required')
        dist=s.load(s.checked(bundle['dist']),'dist')
        require(dist.get_device()=='cuda' and dist.initialized() is False and not torch.distributed.is_initialized(),'Reviewed dist state differs')
        boundary(True)
        vae,var=ns['load_models'](dict(var_source=bundle['author_root'],vae_checkpoint=bundle['weights']['vae']['path'],
            var_checkpoint=bundle['weights']['var']['path']),torch.device('cuda:0'))
        allowed={str(Path(p['path']).resolve()):p for p in bundle['author_files']}
        for name,mod in list(sys.modules.items()):
            if name=='models' or name.startswith('models.'):
                require(str(Path(mod.__file__).resolve()) in allowed,'Unbound author module imported')
        dn=dict(torch=torch,nn=torch.nn,copy=copy)
        proofs['decoder']=definitions(bundle['math']['decoder'],{'ContinuousDecoder'},dn)
        checkpoint=torch.load(s.checked(bundle['weights']['decoder']),map_location='cpu',weights_only=True)
        decoder=dn['ContinuousDecoder'](vae).to('cuda:0');decoder.load_state_dict(checkpoint['decoder'],strict=True);del checkpoint
        self.models=dict(vae=vae,var=var,decoder=decoder)
        for model in self.models.values():model.eval().requires_grad_(False)
        self.verify_models()
        pn=dict(torch=torch,np=np,SIZES=SIZES)
        proofs['prior']=definitions(bundle['math']['prior'],{'_Prior'},pn)
        original=pn['_Prior'];owner=self
        class CountedPrior(original):
            def logits(inner,k):
                return ledger.call('prior_scale',lambda:super(CountedPrior,inner).logits(k),view_id=owner.context,scale=k)
        class CountedDecoder:
            def __call__(inner,latent):
                return ledger.call('decoder_forward',lambda:decoder(latent),view_id=owner.context)
        self.native=types.SimpleNamespace(receiver=types.SimpleNamespace(_Prior=CountedPrior),
            loaded=dict(vae=vae,var=var,decoder=CountedDecoder(),device=torch.device('cuda:0')))
        rn=dict(np=np,time=time,SIZES=SIZES,require=require,token_count=lambda m,K=0:OFFSETS[m]+K)
        proofs['render']=definitions(bundle['math']['render'],{'render_received'},rn);self.render_function=rn['render_received']
        s.write(self.out/'math_AST_identity.json',dict(proofs=proofs,unchanged_definition_nodes=True,source_access_allowed=False))
        boundary(True)

    def verify_models(self):
        for key,model in self.models.items():
            require(self.state_sha(model)==MODELS[key] and not model.training
                and all(not p.requires_grad and p.grad is None for p in model.parameters()),'Frozen model state differs: '+key)

    def render(self,row):
        import numpy as np
        require(numeric_flags(self.t)==s.FLAGS,'Frozen numerical flags changed before render')
        state=normalize_state(row['state']);require(state['kind']=='tokens','Gray cannot run VAR')
        received=np.asarray([v for scale in state['prefix'] for v in scale]+state['partial_values'],dtype=np.int64)
        before=array_sha(received);self.context=row['view_id']
        with self.t.no_grad():
            image=self.ledger.call('var_render',lambda:self.render_function(self.native,received,state['m'],state['K']),view_id=self.context)
        require(array_sha(received)==before,'Renderer altered actual received input');validate_image(image)
        return image

    def close(self):
        self.t.cuda.synchronize();self.verify_models();require(numeric_flags(self.t)==s.FLAGS,'Frozen numerical flags changed');self.boundary(True)


class SourceReadFirewall:
    """Python open audit gate: research data denied except explicit RX/model pins.

    OS/library runtime files are admitted outside the research PFS root. This is
    a runtime data-access assertion for reviewed Python code, not an OS sandbox.
    """
    def __init__(self,files,output,control_files,code_bindings):
        self.files={str(Path(p).resolve()) for p in files}|{str(Path(p).resolve()) for p in control_files}
        self.output=Path(output).resolve();self.codes={str(Path(p).resolve()) for p in code_bindings}
        self.root=s.BASE.resolve();self.env=(s.RT/'envs/frozen_visual_py310').resolve()
        self.metadata_dirs={str(Path(p).parent) for p in self.files if Path(p).parent.name.endswith(('.dist-info','.egg-info'))}
        interpreter=Path(sys.executable).resolve()
        base=Path(sys.base_prefix).resolve();version=str(sys.version_info.major)+'.'+str(sys.version_info.minor)
        encoding=base/'lib'/('python'+version)/'encodings/__init__.py'
        self.absent_stdlib_zip=(base/'lib'/('python'+version.replace('.','')+'.zip')
            if str(interpreter) in self.files and str(encoding) in self.files else None)
        self.reads=collections.Counter();self.blocked=[]

    def permitted(self,path,write=False):
        p=Path(path).resolve();v=str(p)
        if p==self.output or self.output in p.parents:return True
        if v in self.files and not write:return True
        if v in self.codes and not write:return True
        # CPython probes its optional standard-library zip even when absent.
        # Admit only the missing path derived from the pinned interpreter; an
        # existing unbound archive remains forbidden and no bytes are read.
        if not write and p==self.absent_stdlib_zip and not p.exists():return True
        # importlib.metadata probes optional absent records. Allow only a
        # missing metadata member of an already pinned runtime distribution;
        # opening it still raises FileNotFoundError and reads no bytes.
        if (not write and str(p.parent) in self.metadata_dirs and p.name in
            ('entry_points.txt','top_level.txt','METADATA','PKG-INFO','WHEEL','RECORD','INSTALLER','direct_url.json')
            and not p.exists()):return True
        # Imported Python modules may read the bytecode corresponding to a
        # pinned source; no cache write is allowed in the repository.
        if not write and p.suffix=='.pyc' and p.parent.name=='__pycache__':
            source=p.parent.parent/(p.name.split('.')[0]+'.py')
            if str(source) in self.codes or str(source) in self.files:return True
        if self.env==p or self.env in p.parents:return not write
        return self.root!=p and self.root not in p.parents

    def hook(self,event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        p=args[0]
        if not isinstance(p,(str,bytes,os.PathLike)):return
        if isinstance(p,bytes):p=os.fsdecode(p)
        mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        writing=(isinstance(mode,str) and any(x in mode for x in ('w','a','+','x'))) or bool((flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if not self.permitted(p,writing):
            self.blocked.append(str(p));raise RuntimeError('RX-only worker attempted unregistered data access: '+str(p))
        self.reads[str(Path(p).absolute())]+=1


def render_rows(rows,out,ledger,make_backend,boundary):
    """Injectable orchestration used by real worker and no-model CPU stubs."""
    import numpy as np
    require(0<len(rows)<=46800 and len({r['view_id'] for r in rows})==len(rows),'Unique finite logical RX grid required')
    for row in rows:validate_model_row(row)
    out=Path(out);(out/'images').mkdir();cache={};result=[];backend=None;gray=0;reuse_count=0
    try:
        for row in rows:
            boundary();key=actual_input_key(row)
            if key in cache:
                image_record,prior=cache[key];reuse=dict(reused=True,from_view_id=prior,reason='EXACT_ACTUAL_INPUT_IDENTITY');reuse_count+=1
            else:
                if row['state']['kind']=='gray':image=np.full((3,256,256),.5,dtype=np.float32);gray+=1
                else:
                    if backend is None:backend=ledger.call('model_load',make_backend)
                    image=backend.render(row)
                validate_image(image);path=out/'images'/(key+'.npz')
                with path.open('xb') as f:np.savez(f,image=image)
                image_record=dict(image_archive=s.pin(path),image_sha256=array_sha(image),actual_input_key=key)
                cache[key]=(image_record,row['view_id']);reuse=dict(reused=False,from_view_id=None,reason='FIXED_GRAY' if row['state']['kind']=='gray' else 'ACTUAL_VAR_RENDER')
            public={k:row[k] for k in ('view_id','observation_key','actual_rx_key','receiver_state_sha256','input_state_sha256','model_identity')}
            public.update(reconstruction=copy.deepcopy(image_record),reuse=reuse,status='COMPLETE',
                actual_received_kind=row['state']['kind'],source_truth_supplied_to_RX=False)
            result.append(public)
            s.status(out/'running_status.json',dict(logical_complete=len(result),logical_total=len(rows),unique_inputs=len(cache),fixed_gray=gray,reused=reuse_count))
        if backend is not None:backend.close()
        return result,dict(logical_complete=len(result),unique_actual_inputs=len(cache),exact_reuse=reuse_count,
            fixed_gray=gray,var_render=len(cache)-gray,model_load=int(backend is not None),source_image_reads=0,source_token_reads=0)
    except BaseException:
        s.write(out/'partial_logical_frames.json',result);raise


def visual(registration,parent):
    reg=s.validate_registration(registration);require(reg['schema']==s.VISUAL_SCHEMA,'Visual child requires source-free registration')
    out=Path(reg['runtime']['visual_directory']);require(not out.exists(),'Visual attempt cannot replay');out.mkdir()
    guard=s.guard_for(reg,out,parent);guard()
    projection=Path(reg['runtime']['projection_directory']);wait=require_successful_wait(projection)
    cp=projection/'visual_input_completion.json';completion=s.read(cp)
    require(set(completion)=={'status','model_inputs','logical_count','model_identity','projection_digest','source_truth_supplied_to_RX'}
        and completion['status']=='COMPLETE_RX_ONLY_PROJECTION' and completion['source_truth_supplied_to_RX'] is False
        and completion['model_identity']==MODEL_IDENTITY,'Not a pure received-state projection')
    require(Path(completion['model_inputs']['path'])==projection/'model_inputs.json','RX list outside fixed projection location')
    rows=s.readpin(completion['model_inputs']);require(len(rows)==completion['logical_count']==reg['expected_logical_var_images'],'Projected logical count differs')
    for row in rows:validate_model_row(row)
    bundle=s.readpin(reg['model_bundle'])
    controls=[Path(reg['runtime']['root'])/'owner_launch.json',cp,projection/'actual_wait.json',reg['device_receipt']['path'],
        registration['path'],reg['model_bundle']['path'],completion['model_inputs']['path']]
    files=[p['path'] for p in list(bundle['math'].values())+list(bundle['weights'].values())+bundle['author_files']+[bundle['dist']]+bundle['runtime_files']]
    firewall=SourceReadFirewall(files,out,controls,reg['code_bindings']);sys.addaudithook(firewall.hook)
    validate_bundle(bundle)
    device,g=s.resource_device(reg,out);device.check_cuda_environment();history=[];last=0.;ledger=None
    def boundary(force=False):
        nonlocal last
        guard()
        if force or time.monotonic()-last>=5:
            snap,_=device.resource_snapshot(g.helper());history.append(snap);last=time.monotonic()
    try:
        while True:
            guard();ready=s.admission(reg,out)
            if ready['status']=='READY':break
            s.status(out/'pre_model_wait.json',ready);time.sleep(30)
        Ledger=s.load_ledger(reg);limit=len(rows)
        ledger=Ledger(out/'calls.sqlite',dict(model_load=1,var_render=limit,prior_scale=10*limit,decoder_forward=limit),guard)
        s.write(out/'launch.json',dict(registration=registration,process=s.process_identity(os.getpid()),source_truth_supplied_to_RX=False))
        rendered,counts=render_rows(rows,out,ledger,lambda:IsolatedModels(bundle,reg,out,ledger,boundary,device),boundary)
        boundary(True);actual=ledger.summary()
        require(actual['unresolved']==0 and ledger.completed['var_render']==counts['var_render']
            and ledger.completed['prior_scale']==10*counts['var_render'] and ledger.completed['decoder_forward']==counts['var_render']
            and ledger.completed['model_load']==counts['model_load'],'Finite actual model ledger failed closure')
        frames=s.write(out/'logical_frames.json',rendered);resource=s.write(out/'resource_history.json',history)
        access=s.write(out/'input_access_audit.json',dict(blocked=firewall.blocked,opened_paths=dict(firewall.reads),
            source_data_access_allowed=False,source_or_TX_paths_received=False,implementation='Python open audit gate; reviewed model-only code; not an OS sandbox'))
        ledger.close()
        return s.write(out/'completion.json',dict(status='COMPLETE_R1_VISUAL',registration=registration,
            projection_completion=s.pin(cp),projection_actual_wait=wait,model_inputs=completion['model_inputs'],model_bundle=reg['model_bundle'],
            image_rows=frames,logical_rows=len(rendered),counts=counts,ledger=actual,ledger_file=s.pin(out/'calls.sqlite'),
            resource_history=resource,input_access_audit=access,model_identity=MODEL_IDENTITY,numeric_flags=s.FLAGS,
            source_truth_supplied_to_RX=False,tokenizer_calls=0,extra_qualification_calls=0))
    except BaseException as error:
        s.write(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
            ledger=None if ledger is None else ledger.summary(),input_reads=dict(firewall.reads),blocked=firewall.blocked))
        s.write(out/'resource_history_at_failure.json',history);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('project','visual'))
    p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True)
    p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--parent-ticks',type=int,required=True)
    a=p.parse_args();actual=s.process_identity(a.parent_pid)
    require(actual is not None and actual['start_ticks']==a.parent_ticks and os.getppid()==a.parent_pid,'Real owner process identity differs')
    registration=dict(path=str(Path(a.registration).absolute()),sha256=a.expected_registration)
    print(json.dumps((project if a.command=='project' else visual)(registration,a.parent_pid)))


if __name__=='__main__':main()
