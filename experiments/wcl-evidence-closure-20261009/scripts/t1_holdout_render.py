"""Finite post-hoc common500 entropy PHY and reconstruction consumer.

Independent from frozen/running calibration. No policy selection, metric calls,
bootstrap, model training, or original result writes occur here. Actual received
header/profile and payload alone drive recovery; TX truth is diagnostic only.
"""
from __future__ import annotations
import argparse, gzip, json, os, signal, time
from pathlib import Path
import numpy as np
import t1_calibrate as cal
import t2_pilot as shared
from t1_entropy_core import OFFSETS, candidates, choose_arithmetic, read, require, sha, verify
from t1_source_asset_schema import exact_received_cache, payload_sha, token_sha
from t1_codec_runtime import SourceCodec
from t1_holdout_metadata import FAMILIES, SNRS, SEEDS, MANIFEST_SHA
from t2_ledger import Ledger

SCHEMA='WCL_T1_POSTHOC_COMMON500_RENDER_V1'
pin=cal.pin;loadpin=cal.loadpin;save=cal.save;digest=cal.digest


def load_source(item,freeze):
    verify(item['checkpoint'],item['checkpoint_sha256']);r=read(item['checkpoint'])
    require(r['schema']=='T1_SOURCE_ASSET_V1' and r['source_role']=='holdout'
        and r['post_hoc_supplement'] and r['holdout_used_for_selection'] is False
        and r['source_index']==item['source_index'] and r['source_id']==item['source_id']
        and r['freeze']==freeze,'Exact post-freeze common500 source identity required')
    verify(r['archive']['path'],r['archive']['sha256'])
    with np.load(r['archive']['path'],allow_pickle=False) as z:a={k:z[k].copy()for k in z.files}
    require(token_sha(a[r['tokens_key']])==r['source_tokens_sha256'],'Frozen source token hash differs')
    return r,a


def selected_stream(sr,arrays,family,snr,candidate):
    """Selection is predetermined by frozen policy and actual encoded lengths."""
    lengths={int(m):e['arithmetic_bits']for m,e in sr['tx_prefixes'][family].items()}
    selected=choose_arithmetic(lengths,candidate,minimum_m=4)
    require(selected['status']=='SOURCE_LENGTH_FITS_LAYOUT_PENDING','Missing/nonfitting source prefix is not a channel failure')
    stored=sr['selected_by_snr'][family][str(snr)]
    require(all(stored[k]==candidate[k]for k in ('candidate_id','target_m','q','nominal_rate'))
        and all(stored[k]==v for k,v in selected.items()),'Post-freeze selected prefix changed')
    m=selected['actual_m'];entry=sr['streams'][family][str(m)];bits=arrays[entry['bits_key']]
    tx=sr['tx_prefixes'][family][str(m)]
    require(len(bits)==selected['arithmetic_bits']==entry['payload_bits']
        and payload_sha(bits)==entry['payload_sha256'] and np.array_equal(bits,arrays[tx['bits_key']]),
        'Independent roundtrip stream differs from actual selected TX stream')
    return m,bits,selected


def expected_grid(r):
    return {(f,s,n,i)for f in FAMILIES for s in SNRS for n in SEEDS for i in range(500)}


def validate_request(r):
    require(r['schema']==SCHEMA and r['families']==FAMILIES and r['snrs']==SNRS and r['seeds']==SEEDS,
        'Registered two-family, three-SNR, three-noise experiment only')
    require(r['source_count']==500 and r['frame_count']==9000 and r['packet_cap']==18000
        and len(r['records'])==500 and len(set(r['source_ids']))==500,'Finite original500 scope')
    require([(x['source_index'],x['source_id'])for x in r['records']]==list(enumerate(r['source_ids'])),
        'Original common500 ordered IDs required')
    require(type(r['workers'])is int and 1<=r['workers']<=16 and 0<r['max_seconds']<=86400,'Finite worker/window scope')
    require(r['post_hoc_supplement'] and r['holdout_used_for_selection'] is False and r['training_updates']==0,
        'Post-hoc inference only')
    byid={c['candidate_id']:c for c in candidates()}
    require(set(r['policies'])==set(FAMILIES),'Both frozen families required')
    for f in FAMILIES:
        require(set(r['policies'][f])=={str(s)for s in SNRS},'Fixed three policies per family')
        for c in r['policies'][f].values():
            require(c['candidate_id']in byid and all(c[k]==v for k,v in byid[c['candidate_id']].items()),'Noncandidate policy')


def read_output(done,name):
    entries=[(p,h)for p,h in done['outputs'].items()if Path(p).name==name]
    require(len(entries)==1,'Unique sealed old output required: '+name);p,h=entries[0];verify(p,h)
    with gzip.open(p,'rt',encoding='utf-8')as f:value=json.load(f)
    return value,dict(path=p,sha256=h)


def old_reuse_index(source_request,image_done,facts):
    """Only original received whole states; no TX reconstruction enters reuse."""
    require(image_done['status']=='MAIN_RAW64_UNIFIED500_SAME_RX_IMAGES_COMPLETE_V1'
        and image_done['source_count']==500 and image_done['source_ids']==source_request['source_ids'],
        'Complete original same-RX image closure required')
    maps=[{}for _ in range(500)];model_ids=set()
    for row in facts:
        i=row['source_index'];state=row['receiver_state']
        require(row['source_id']==source_request['source_ids'][i],'Old received source mismatch')
        if state['K']!=0:continue
        proof=row['same_RX_visual_proof'];im=row['same_RX_images']['VAR_completion']
        require(proof['same_received_information'] and not proof['TX_truth_used_for_generation']
            and not proof['target_used_for_generation'] and not proof['policy_selection']
            and proof['actual_state_sha256']==row['actual_state_sha256']
            and proof['image_sha256']['VAR_completion']==im['image_sha256'], 'Original actual-state/image evidence differs')
        model_ids.add(proof['model_identity_sha256']);p=im['image_archive']
        require(p in image_done['outputs'],'Original image is not sealed')
        value=dict(state=state,image_path=p,image_key='images',image_slot=im['image_slot'],
            image_sha256=im['image_sha256'],image_file_sha256=image_done['outputs'][p],
            reuse='EXACT_ORIGINAL_RECEIVED_STATE_IMAGE',new_VAR_render_calls=0)
        key=digest(state)
        if key in maps[i]:require(maps[i][key]['image_sha256']==value['image_sha256'],'Same received state rendered differently')
        else:maps[i][key]=value
    require(len(model_ids)==1 and all(maps),'Unique frozen original renderer and all500 reuse indexes required')
    return maps,model_ids.pop()


def prepare(args):
    source_request=read(args.source_request);policy=loadpin(source_request['freeze'])
    require(source_request['schema']=='T1_POSTHOC_COMMON500_SOURCE_REQUEST_V1'
        and source_request['source_count']==500 and source_request['noise_seeds']==SEEDS
        and source_request['snrs']==SNRS and source_request['frame_count']==9000
        and policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1'
        and not policy['holdout_used_for_selection'],'Frozen source request required')
    for p,h in source_request['input_bindings'].items():verify(p,h)
    require(source_request['original_source_manifest']['sha256']==MANIFEST_SHA,'Published500 source manifest pin differs')
    done=read(args.source_completion);mp=Path(args.source_completion).parent/'manifest.json'
    require(done['status']=='T1_POSTHOC_COMMON500_SOURCE_PREPARATION_COMPLETE'
        and done['request_sha256']==sha(args.source_request) and done['source_count']==500
        and done['freeze']==source_request['freeze'],'Actual selected source streams required')
    verify(mp,done['outputs'][str(mp.resolve())]);manifest=read(mp)
    require(manifest['status']=='T1_POSTHOC_COMMON500_SOURCE_ASSETS_READY'
        and manifest['source_ids']==source_request['source_ids'] and manifest['freeze']==source_request['freeze'],
        'Source preparation population/freeze differs')
    protocol=policy['protocol'];loadpin(protocol['static_completion']);qual=loadpin(protocol['phy_qualification'])
    require(qual['status']=='PASS' and qual['packet_decode_count']==241,'Actual PHY qualification required')
    env=read(args.environment_request);root=Path(args.root).resolve();out=Path(args.out).resolve()
    require(source_request['root']==env['root']==str(root) and out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009')
        and not(out/'request.json').exists(),'Fresh independent WCL stage required')
    for entry,original in zip(manifest['records'],source_request['records']):
        verify(entry['checkpoint'],done['outputs'][entry['checkpoint']]);sr,a=load_source(entry,source_request['freeze'])
        require(sr['source_assets_checkpoint']==dict(path=original['checkpoint'],sha256=original['checkpoint_sha256'])
            and sr['source_tokens_sha256']==original['tokens_sha256'] and sr['preprocessing_id']==original['preprocessing_id']
            and sr['static_model_completion_sha256']==protocol['static_completion']['sha256'], 'Exact original source/model binding required')
        verify(sr['archive']['path'],done['outputs'][sr['archive']['path']])
        for f in FAMILIES:
            for s in SNRS:selected_stream(sr,a,f,s,policy['policies'][f][str(s)])
    old_done=loadpin(source_request['original_statistics_completion']);facts,fact_pin=read_output(old_done,'raw_receive_facts.json.gz')
    maps,old_model_id=old_reuse_index(source_request,read(args.original_image_completion),facts)
    reuse_pins=[]
    for i,values in enumerate(maps):
        p=out/'old_reuse'/f'{i:04d}.json';save(p,dict(source_index=i,source_id=source_request['source_ids'][i],states=values));reuse_pins.append(pin(p))
    files=['t1_holdout_render.py','t1_calibrate.py','t1_entropy_core.py','t1_codec_runtime.py','t1_source_asset_schema.py',
        't1_phy.py','t1_holdout_metadata.py','t2_pilot.py','t2_ledger.py']
    r=dict(schema=SCHEMA,root=str(root),out=str(out),families=FAMILIES,snrs=SNRS,seeds=SEEDS,source_count=500,
        frame_count=9000,packet_cap=18000,source_ids=source_request['source_ids'],records=manifest['records'],
        policies=policy['policies'],freeze=source_request['freeze'],source_request=pin(args.source_request),
        source_completion=pin(args.source_completion),source_manifest=pin(mp),environment_request=pin(args.environment_request),
        static_completion=protocol['static_completion'],phy_qualification=protocol['phy_qualification'],
        source_model_id=protocol['source_model_id'],registered_profile_catalogue=protocol['registered_profile_catalogue'],
        original_statistics_completion=source_request['original_statistics_completion'],old_receive_facts=fact_pin,
        original_image_completion=pin(args.original_image_completion),old_reuse=reuse_pins,old_render_model_id=old_model_id,
        workers=args.workers,deadline_unix=args.deadline_unix,max_seconds=86400,
        noise_rule='t1_phy.standard_noise(source_id,6201/6202/6203), fixed1024 symbols, no family/profile key',
        counter_rule='((index_in_original_six_SNR_order*500)+original_source_index)*3+(noise_seed-6201)',
        source_decode_upper_bound=9000,new_VAR_render_upper_bound=9001,render_qualification_calls=1,
        GPU_budget_rule='After all CPU receipts, plan-render seals exact received identities and finite per-source bounds; no unregistered GPU work',
        failure_output='constant0.5 RGB on actual header/CRC/parser/source canonical rejection; software errors stop',
        source_bindings={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n))for n in files},
        stop_files=[str(root/'STOP'),str(out/'STOP')],post_hoc_supplement=True,holdout_used_for_selection=False,
        training_updates=0,new_metric_calls=0,old_ledgers_opened=False)
    validate_request(r);save(out/'request.json',r)
    return dict(status='T1_POSTHOC_COMMON500_RENDER_REGISTERED_NOT_RUN',request=pin(out/'request.json'),frame_count=9000,packet_cap=18000)


def registered(path):
    r=read(path);validate_request(r)
    for p,h in r['source_bindings'].items():verify(p,h)
    for key in ('freeze','source_request','source_completion','source_manifest','environment_request','static_completion','phy_qualification'):
        loadpin(r[key])
    return r,sha(path)


def guard(r,started,native=None):
    shared.guard(r,started)
    if native is not None:native.health_polling.next_check=0.;native.common.check()


def counter(i,s,n):return ([1,4,7,10,13,19].index(s)*500+i)*3+(n-6201)


def ledger_init(path):
    """One parent initializes WAL before any child opens the shared ledger."""
    r,rh=registered(path);out=Path(r['out'])
    with shared.lock(out/'ledger_init.lock'):
        meter=Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap']);snap=meter.snapshot();meter.close()
        require(snap['unresolved']==0,'Unresolved prior actual calls require inspection')
        p=out/'ledger_initialized.json'
        save(p,dict(request_sha256=rh,packet_cap=r['packet_cap'],status='WAL_INITIALIZED_NO_SCIENTIFIC_CALLBACK'))
    return dict(status='WAL_READY',actual_calls=snap['total'],packet_cap=r['packet_cap'])


def open_worker_ledger(r,rh):
    import fcntl
    out=Path(r['out']);init=read(out/'ledger_initialized.json')
    require(init['request_sha256']==rh and init['packet_cap']==r['packet_cap']
        and(out/'packet_ledger.sqlite').is_file(),'Parent ledger-init must precede children')
    # Serialize constructor-only DDL/PRAGMA; actual callbacks remain parallel.
    with(out/'ledger_open.lock').open('a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:return Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap'])
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def cpu_worker(path,index):
    import t1_phy as phy
    r,rh=registered(path);out=Path(r['out']);require(0<=index<r['workers'],'Registered worker index')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only actual LDPC worker')
    started=time.monotonic()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,shared.stop)
    with shared.lock(out/f'cpu_worker_{index}.lock'):
        rt=phy.create_runtime(r['root'],r['phy_qualification']['path']);ledger=open_worker_ledger(r,rh)
        require(rt.catalogue==r['registered_profile_catalogue'],'Registered public paid profile catalogue changed')
        profiles={(p['family'],p['m'],p['q'],p['nominal_rate']):p for p in rt.catalogue['profiles']};completed=[]
        try:
            for entry in r['records'][index::r['workers']]:
                guard(r,started);i=entry['source_index'];cp=out/'cpu_sources'/f'{i:04d}.json'
                if cp.exists():
                    require(read(cp)['request_sha256']==rh,'CPU checkpoint request differs');completed.append(pin(cp));continue
                sr,arrays=load_source(entry,r['freeze']);frames=[]
                for snr in SNRS:
                    for seed in SEEDS:
                        ctr=counter(i,snr,seed);noise=phy.standard_noise(entry['source_id'],seed)*10**(-snr/20)
                        for family in FAMILIES:
                            guard(r,started);c=r['policies'][family][str(snr)];m,bits,selection=selected_stream(sr,arrays,family,snr,c)
                            p=profiles[(family,m,c['q'],c['nominal_rate'])];pid=p['profile_id']
                            actual=out/'physical_frames'/f'{i:04d}'/f'{snr}_{seed}_{pid:03d}.json'
                            identity=dict(request_sha256=rh,source_index=i,source_id=entry['source_id'],snr_db=snr,noise_seed=seed,
                                TX_profile_id=pid,payload_sha256=phy.array_sha(bits),noise_sha256=phy.array_sha(noise),public_frame_counter=ctr)
                            if actual.exists():
                                packet=read(actual);require(all(packet[k]==v for k,v in identity.items()),'Actual packet identity differs')
                            else:
                                wave,tx=rt.transmit(pid,bits,ctr);observed=wave+noise
                                event=f'{SCHEMA}/source{i}/snr{snr}/noise{seed}/profile{pid}'
                                rx=rt.receive(observed,snr,ctr,rt.profiles,ledger,event,phase='posthoc_common500')
                                packet=dict(identity,observation_sha256=phy.array_sha(observed),transmission=tx,actual_RX=rx)
                                save(actual,packet)
                            frames.append(dict(source_index=i,source_id=entry['source_id'],snr_db=snr,noise_seed=seed,
                                family=family,point_id=family+'_SNR_'+str(snr),candidate_id=c['candidate_id'],target_m=c['target_m'],
                                actual_m=m,q=c['q'],nominal_rate=c['nominal_rate'],fallback_attempts=selection['attempts'],physical_frame=pin(actual)))
                save(cp,dict(status='T1_POSTHOC_CPU_SOURCE_COMPLETE',request_sha256=rh,source_index=i,source_id=entry['source_id'],frames=frames))
                completed.append(pin(cp));print(json.dumps(dict(stage='T1_POSTHOC_CPU',worker=index,source=i,frames=len(frames))),flush=True)
            save(out/'cpu_workers'/f'{index:02d}.json',dict(status='COMPLETE',request_sha256=rh,sources=completed))
        finally:ledger.close()


def cached_recovery(sr,arrays,packet,models):
    """Read-only preflight: None means a real source decoder is still needed."""
    rx=packet['actual_RX'];body=rx['body']
    if not rx['header']['header_ok'] or body is None or not body['crc_accepted'] or not body['parser_accepted']:return cal.gray_state()
    p=rx['rx_profile'];entry=sr['streams'].get(p['family'],{}).get(str(p['m']))
    restricted={}if entry is None else{k:arrays[k]for k in(entry['bits_key'],entry['received_tokens_key'])}
    result=exact_received_cache(sr,restricted,family=p['family'],m=p['m'],received_bits=body['payload'],expected_source_model_id=models[p['family']])
    return None if result is None else cal.receiver_state(result['received_tokens'],p['m'])


def cpu_source(r,rh,i):
    p=Path(r['out'])/'cpu_sources'/f'{i:04d}.json';v=read(p)
    require(v['status']=='T1_POSTHOC_CPU_SOURCE_COMPLETE' and v['request_sha256']==rh
        and v['source_index']==i and v['source_id']==r['source_ids'][i] and len(v['frames'])==18,'Complete fixed CPU source required')
    require({(x['family'],x['snr_db'],x['noise_seed'],x['source_index'])for x in v['frames']}
        =={(f,s,n,i)for f in FAMILIES for s in SNRS for n in SEEDS},'Complete fixed18 source grid required')
    return v,pin(p)


def plan_render(path):
    r,rh=registered(path);out=Path(r['out']);plans=[];packet_calls=0;physical_paths=set()
    for entry in r['records']:
        i=entry['source_index'];cp,cp_pin=cpu_source(r,rh,i);sr,a=load_source(entry,r['freeze']);old=loadpin(r['old_reuse'][i])
        pending={};known={};decode_ids={};packets={}
        for frame in cp['frames']:
            packet=loadpin(frame['physical_frame']);require(packet['request_sha256']==rh,'Physical request differs')
            packets[frame['physical_frame']['path']]=frame['physical_frame']['sha256']
            if frame['physical_frame']['path']not in physical_paths:
                packet_calls+=1+int(packet['actual_RX']['header']['header_ok']);physical_paths.add(frame['physical_frame']['path'])
            identity=cal.recovery_input(packet,r['source_model_id']);key=digest(identity);decode_ids[key]=identity
            state=cached_recovery(sr,a,packet,r['source_model_id'])
            if state is None:pending[key]=identity['actual_received_profile']['family']
            else:known[digest(state)]=state
        new_known=[k for k,v in known.items()if v['kind']!='gray'and k not in old['states']]
        plans.append(dict(source_index=i,source_id=entry['source_id'],cpu_source=cp_pin,physical_frames=packets,
            recovery_identities=decode_ids,pending_actual_source_decodes=pending,
            new_VAR_source_decode_upper_bound=sum(f=='EC_VAR_WHOLE'for f in pending.values()),
            new_static_source_decode_upper_bound=sum(f=='EC_STATIC_WHOLE'for f in pending.values()),
            known_new_VAR_render_states=new_known,new_VAR_render_upper_bound=len(new_known)+len(pending),
            existing_received_state_image_hits=len([k for k in known if k in old['states']])))
    ledger=cal.ledger_snapshot(out/'packet_ledger.sqlite',rh,r['packet_cap']);require(ledger['total']==packet_calls,'Actual packet receipts and ledger differ')
    plan=dict(status='T1_POSTHOC_RENDER_BUDGET_REGISTERED_NOT_RUN',request_sha256=rh,plans=plans,source_count=500,frame_count=9000,
        new_VAR_source_decode_upper_bound=sum(p['new_VAR_source_decode_upper_bound']for p in plans),
        new_static_source_decode_upper_bound=sum(p['new_static_source_decode_upper_bound']for p in plans),
        new_VAR_render_upper_bound=sum(p['new_VAR_render_upper_bound']for p in plans),render_qualification_calls=1,
        new_metric_calls=0,new_packet_decodes=0,completed_PHY_ledger=ledger)
    require(plan['new_VAR_render_upper_bound']+1<=9001 and plan['new_VAR_source_decode_upper_bound']<=9000,'Finite GPU cap exceeded')
    save(out/'render_plan.json',plan);return {k:v for k,v in plan.items()if k!='plans'}


def checked_image(value):
    verify(value['image_path'],value['image_file_sha256'])
    with np.load(value['image_path'],allow_pickle=False)as z:
        image=z[value['image_key']].copy()
        if value['image_key']=='images':image=image[value['image_slot']]
        else:require(value['image_key']=='image'and value['image_slot']==0,'Single image uses slot0')
    require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all()
        and shared.image_sha(image)==value['image_sha256'],'Sealed reconstruction identity differs')
    return image


def build_native(env,out):
    shared.add_paths(env);v=env['visual_config'];b=env['source_bindings']
    quality=shared.load(v['quality_driver_module'],'_t1_posthoc_quality',b)
    source=shared.load(v['source_driver_module'],'_t1_posthoc_source',b)
    native=quality.build_native(Path(env['root']),v['native_runtime'],shared.stop)
    require(native.loaded['identity']==env['old_visual_identity'] and native.flags==env['old_numerical_runtime']
        and native.driver_bindings==env['native_source_bindings'],'Frozen model/runtime/source graph differs')
    return native,source


def qualify_render(r,rh,out,native,source):
    p=out/'render_qualification.json'
    if p.exists():require(read(p)['request_sha256']==rh and read(p)['status']=='PASS','Qualification differs');return
    old=loadpin(r['old_reuse'][0]);key=next(k for k,v in old['states'].items()if v['state']['kind']!='gray')
    value=old['states'][key];truth=checked_image(value);reserved=p.with_suffix('.reserved.json')
    require(not reserved.exists(),'Unresolved one-call renderer qualification; inspect before retry')
    save(reserved,dict(request_sha256=rh,state=value['state'],new_VAR_render_calls=1))
    image=shared.render_state(native,source,value['state'])
    require(np.array_equal(image,truth) and shared.image_sha(image)==value['image_sha256'],'Original whole-prefix renderer exact replay failed')
    save(p,dict(status='PASS',request_sha256=rh,old_reuse=r['old_reuse'][0],state_key=key,
        image_sha256=value['image_sha256'],exact_float32_equal=True,new_VAR_render_calls=1,new_metric_calls=0))


def diagnostic_token_errors(state,tx_tokens):
    """Called only after output selection; cannot change reconstruction or status."""
    if state['kind']=='gray':return dict(token_error_count=None,token_error_denominator=0,received_prefix_matches_source=None)
    actual=np.asarray([v for scale in state['prefix']for v in scale],dtype=np.int64)
    require(len(actual)==OFFSETS[state['m']],'Diagnostic prefix length differs')
    errors=int(np.count_nonzero(actual!=tx_tokens[:len(actual)]))
    return dict(token_error_count=errors,token_error_denominator=len(actual),received_prefix_matches_source=errors==0)


def gpu_worker(path):
    r,rh=registered(path);out=Path(r['out']);env=loadpin(r['environment_request'])
    # Recompute the read-only manifest; immutable save detects any changed budget.
    plan_render(path);plan=read(out/'render_plan.json')
    require(plan['status']=='T1_POSTHOC_RENDER_BUDGET_REGISTERED_NOT_RUN'and plan['request_sha256']==rh,'Seal actual-RX GPU plan first')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Registered GPU0 required');started=time.monotonic()
    for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,shared.stop)
    with shared.lock(env['visual_config']['visual_lock']),shared.lock(out/'gpu.lock'):
        native,source=build_native(env,out);guard(r,started,native);qualify_render(r,rh,out,native,source)
        codec=SourceCodec(r['root'],r['static_completion']['path'],native)
        for entry,pplan in zip(r['records'],plan['plans']):
            guard(r,started,native);i=entry['source_index'];cp=out/'gpu_sources'/f'{i:04d}.json'
            if cp.exists():require(read(cp)['request_sha256']==rh,'GPU source request differs');continue
            physical=loadpin(pplan['cpu_source']);sr,arrays=load_source(entry,r['freeze']);old=loadpin(r['old_reuse'][i]);cache={}
            rows=[];decode_pins={};new_states={};checked_old=set();attempted_new_non_gray=set()
            for frame in physical['frames']:
                guard(r,started,native);packet=loadpin(frame['physical_frame'])
                require(pplan['physical_frames'].get(frame['physical_frame']['path'])==frame['physical_frame']['sha256'],'Unplanned physical receipt')
                identity=cal.recovery_input(packet,r['source_model_id']);require(pplan['recovery_identities'].get(digest(identity))==identity,'Unplanned source decode')
                state,evidence,dp=cal.recover_once(out/'source_decodes'/f'{i:04d}',rh,sr,arrays,packet,codec,r['source_model_id'])
                decode_pins[dp['path']]=dp;key=digest(state)
                if key not in cache:
                    if key in old['states']:
                        value=old['states'][key];require(value['state']==state,'Original received state differs')
                        if key not in checked_old:checked_image(value);checked_old.add(key)
                    else:
                        sp=out/'gpu_states'/f'{i:04d}'/(key+'.json');image_path=sp.with_suffix('.npz')
                        if state['kind']!='gray':
                            require(len(attempted_new_non_gray)<pplan['new_VAR_render_upper_bound'],'Source render cap exhausted before model call')
                            attempted_new_non_gray.add(key)
                        if sp.exists():
                            value=read(sp);require(value['request_sha256']==rh and value['state']==state,'Saved render state differs');checked_image(value)
                        else:
                            reserved=sp.with_suffix('.reserved.json');require(not reserved.exists(),'Unresolved render; no automatic repeat')
                            save(reserved,dict(request_sha256=rh,state=state,source_id=entry['source_id']))
                            image=shared.render_state(native,source,state)
                            with image_path.open('xb')as f:np.savez(f,image=image)
                            value=dict(request_sha256=rh,state=state,image_path=str(image_path),image_key='image',image_slot=0,
                                image_sha256=shared.image_sha(image),image_file_sha256=sha(image_path),reuse='NEW_ACTUAL_RECEIVED_STATE',
                                new_VAR_render_calls=int(state['kind']!='gray'))
                            save(sp,value)
                        new_states[str(sp)]=pin(sp)
                    cache[key]=value
                value=cache[key];rx=packet['actual_RX'];body=rx['body'];tx=packet['transmission'];rp=rx['rx_profile']
                # Ground truth enters only the following offline diagnostic.
                diag=diagnostic_token_errors(state,arrays[sr['tokens_key']])
                rows.append(dict(source_index=i,source_id=entry['source_id'],family=frame['family'],point_id=frame['point_id'],
                    snr_db=frame['snr_db'],noise_seed=frame['noise_seed'],candidate_id=frame['candidate_id'],target_m=frame['target_m'],
                    actual_m=frame['actual_m'],K=0,N=1024,transmitted_token_count=int(OFFSETS[frame['actual_m']]),q=frame['q'],nominal_rate=frame['nominal_rate'],
                    actual_k=tx['k'],actual_n=tx['n'],header_symbols=tx['header_symbols'],body_symbols=tx['body_symbols'],
                    actual_code_rate=tx['k']/tx['n'],header_energy=tx['header_energy'],body_energy=tx['body_energy'],
                    frame_padding_symbols=tx['frame_padding_symbols'],arithmetic_bits=tx['arithmetic_bits'],
                    known_information_padding_bits=tx['known_information_padding_bits'],E_frame=tx['E_frame'],rho=tx['rho'],
                    fallback_attempts=json.dumps(frame['fallback_attempts'],sort_keys=True,separators=(',',':')),
                    header_ok=rx['header']['header_ok'],body_crc_accept=None if body is None else body['crc_accepted'],
                    body_parser_accept=None if body is None else body['parser_accepted'],physical_status=rx['status'],
                    received_family=None if rp is None else rp['family'],received_m=None if rp is None else rp['m'],
                    received_profile_id=None if rp is None else rp['profile_id'],source_status=evidence['source_status'],
                    source_canonical_accept=evidence['source_status']=='ARITHMETIC_SOURCE_DECODED',
                    source_parse_reason=evidence.get('parse_reason',''),source_decode_cache_hit=evidence['source_decode_cache_hit'],
                    gray=state['kind']=='gray',received_state_sha256=key,visual_reuse=value['reuse'],
                    image_path=value['image_path'],image_key=value['image_key'],image_slot=value['image_slot'],
                    image_sha256=value['image_sha256'],image_file_sha256=value['image_file_sha256'],
                    diagnostic_truth_used_for_output=False,**diag))
            actual_VAR_decodes=sum(loadpin(d)['evidence']['new_VAR_source_decode']for d in decode_pins.values())
            actual_static_decodes=0
            for d in decode_pins.values():
                decoded=loadpin(d);identity=decoded['recovery_input']
                actual_static_decodes+=int(identity['received_payload']is not None
                    and identity['actual_received_profile']['family']=='EC_STATIC_WHOLE'
                    and not decoded['evidence']['source_decode_cache_hit'])
            actual_renders=sum(loadpin(d)['new_VAR_render_calls']for d in new_states.values())
            require(actual_VAR_decodes<=pplan['new_VAR_source_decode_upper_bound'] and actual_renders<=pplan['new_VAR_render_upper_bound']
                and actual_static_decodes<=pplan['new_static_source_decode_upper_bound'],
                'Actual GPU work exceeds registered source bounds')
            save(cp,dict(status='T1_POSTHOC_GPU_SOURCE_COMPLETE',request_sha256=rh,source_index=i,source_id=entry['source_id'],rows=rows,
                cpu_source=pplan['cpu_source'],source_decode_checkpoints=list(decode_pins.values()),render_state_checkpoints=list(new_states.values()),
                new_VAR_source_decode_calls=actual_VAR_decodes,new_static_source_decode_calls=actual_static_decodes,
                new_VAR_render_calls=actual_renders,new_metric_calls=0))
            print(json.dumps(dict(stage='T1_POSTHOC_GPU',source=i,rows=len(rows),new_VAR_renders=actual_renders)),flush=True)
        native.frozen()


def close(path):
    r,rh=registered(path);out=Path(r['out']);rows=[];outputs={};packets=set();packet_calls=0;decode_count=0;static_decode_count=0;render_count=0
    plan=read(out/'render_plan.json');qual=read(out/'render_qualification.json')
    require(plan['request_sha256']==qual['request_sha256']==rh and qual['status']=='PASS','Actual render plan/qualification required')
    outputs[str(out/'render_plan.json')]=sha(out/'render_plan.json');outputs[str(out/'render_qualification.json')]=sha(out/'render_qualification.json')
    for entry in r['records']:
        i=entry['source_index'];p=out/'gpu_sources'/f'{i:04d}.json';cp=read(p)
        require(cp['status']=='T1_POSTHOC_GPU_SOURCE_COMPLETE'and cp['request_sha256']==rh
            and cp['source_index']==i and cp['source_id']==entry['source_id'],'Complete actual GPU source required')
        outputs[str(p)]=sha(p);rows+=cp['rows'];cpu=loadpin(cp['cpu_source']);outputs[cp['cpu_source']['path']]=cp['cpu_source']['sha256']
        for frame in cpu['frames']:
            d=frame['physical_frame'];packet=loadpin(d);require(packet['request_sha256']==rh,'Packet request differs')
            if d['path']not in packets:packet_calls+=1+int(packet['actual_RX']['header']['header_ok']);packets.add(d['path'])
            outputs[d['path']]=d['sha256']
        for d in cp['source_decode_checkpoints']+cp['render_state_checkpoints']:
            v=loadpin(d);require(v['request_sha256']==rh,'Recovery/render receipt differs');outputs[d['path']]=d['sha256']
        decode_count+=cp['new_VAR_source_decode_calls'];static_decode_count+=cp['new_static_source_decode_calls'];render_count+=cp['new_VAR_render_calls']
        for row in cp['rows']:
            if row['image_path']not in outputs:verify(row['image_path'],row['image_file_sha256'])
            require(outputs.get(row['image_path'],row['image_file_sha256'])==row['image_file_sha256'],'Image closure hash conflict')
            outputs[row['image_path']]=row['image_file_sha256']
    require(len(rows)==9000 and {(x['family'],x['snr_db'],x['noise_seed'],x['source_index'])for x in rows}==expected_grid(r),
        'All9000 logical conditions including failures must be retained')
    for row in rows:
        c=r['policies'][row['family']][str(row['snr_db'])]
        require(row['source_id']==r['source_ids'][row['source_index']] and row['candidate_id']==c['candidate_id']
            and row['point_id']==row['family']+'_SNR_'+str(row['snr_db']),'Frozen policy/source mapping differs')
    ledger=cal.ledger_snapshot(out/'packet_ledger.sqlite',rh,18000)
    require(ledger['total']==packet_calls and decode_count<=plan['new_VAR_source_decode_upper_bound']
        and static_decode_count<=plan['new_static_source_decode_upper_bound']
        and render_count<=plan['new_VAR_render_upper_bound'],'Actual independent budgets differ')
    rows.sort(key=lambda x:(x['source_index'],x['snr_db'],x['noise_seed'],x['family']))
    cal.csv_save(out/'per_frame.csv',rows);outputs[str(out/'per_frame.csv')]=sha(out/'per_frame.csv')
    result=dict(status='T1_POSTHOC_COMMON500_RENDER_COMPLETE',request_path=str(Path(path).resolve()),request_sha256=rh,
        source_count=500,source_ids=r['source_ids'],noise_count=3,frame_count=9000,families=FAMILIES,snrs=SNRS,noise_seeds=SEEDS,
        freeze=r['freeze'],source_request=r['source_request'],original_statistics_completion=r['original_statistics_completion'],
        per_frame=str(out/'per_frame.csv'),outputs=outputs,ledger=ledger,new_VAR_source_decode_calls=decode_count,
        new_static_source_decode_calls=static_decode_count,
        new_VAR_render_calls=render_count,new_VAR_qualification_render_calls=1,new_metric_calls=0,training_updates=0,
        post_hoc_supplement=True,holdout_used_for_selection=False,policy_selection=False,scientific_statistics_completed=False)
    save(out/'completion.json',result);return {k:v for k,v in result.items()if k not in('outputs','source_ids')}


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare')
    for n in('root','source-request','source-completion','environment-request','original-image-completion','out'):q.add_argument('--'+n,required=True)
    q.add_argument('--workers',type=int,default=8);q.add_argument('--deadline-unix',type=float,required=True)
    for n in('ledger-init','cpu-worker','plan-render','gpu-worker','close'):
        q=s.add_parser(n);q.add_argument('--request',required=True)
        if n=='cpu-worker':q.add_argument('--index',type=int,required=True)
    a=p.parse_args()
    if a.command=='prepare':v=prepare(a)
    elif a.command=='ledger-init':v=ledger_init(a.request)
    elif a.command=='cpu-worker':v=cpu_worker(a.request,a.index)
    elif a.command=='plan-render':v=plan_render(a.request)
    elif a.command=='gpu-worker':v=gpu_worker(a.request)
    else:v=close(a.request)
    if v is not None:print(json.dumps(v),flush=True)


if __name__=='__main__':main()
