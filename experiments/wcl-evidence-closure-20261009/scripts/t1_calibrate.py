"""Finite entropy whole-prefix calibration, with independent actual PHY ledger.

Pilot: first100 calibration sources, 36 candidates/family, 4/10/19, seed4101.
Full: calibration1000 x three noises, the pilot top3 per family and SNR.
The former common500 holdout is never read by this program.
"""
from __future__ import annotations
import argparse, contextlib, csv, io, json, math, os
from pathlib import Path
import signal, sqlite3, sys, time
import numpy as np
from t1_entropy_core import OFFSETS, candidates, choose_arithmetic, read, require, sha, verify
from t1_source_asset_schema import exact_received_cache, model_id, payload_sha, token_sha
from t1_codec_runtime import SourceCodec, InvalidSourceStream
from t2_ledger import Ledger
import t2_pilot as shared

SCHEMA='WCL_T1_ENTROPY_CALIBRATION_V1'
SNRS=[4,10,19]
FAMILIES=['EC_STATIC_WHOLE','EC_VAR_WHOLE']
METRICS=shared.METRICS
STOP=False

def stop(*_):
    global STOP;STOP=True

def pin(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def loadpin(d):verify(d['path'],d['sha256']);return read(d['path'])
def save(path,value):shared.save(path,value)
def digest(value):return shared.digest(value)

def load_source(item):
    verify(item['checkpoint'],item['checkpoint_sha256']);r=read(item['checkpoint'])
    require(r['schema']=='T1_SOURCE_ASSET_V1' and r['source_id']==item['source_id'] and r['source_index']==item['source_index'],
            'Exact source checkpoint identity required')
    verify(r['archive']['path'],r['archive']['sha256'])
    with np.load(r['archive']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    require(r['source_role']=='calibration','No holdout images enter calibration')
    require(token_sha(a[r['tokens_key']])==r['source_tokens_sha256'],'Source token asset changed')
    return r,a

def selected_stream(record,arrays,family,candidate):
    entries=record['streams'][family]
    selection=choose_arithmetic({int(m):e['payload_bits'] for m,e in entries.items()},candidate,minimum_m=4)
    require(selection['status']=='SOURCE_LENGTH_FITS_LAYOUT_PENDING',
            'Missing or non-fitting registered fallback is not a channel failure: '+str(selection))
    m=selection['actual_m'];entry=entries[str(m)];bits=arrays[entry['bits_key']]
    require(len(bits)==selection['arithmetic_bits'],'Selected real source length differs')
    return m,bits,selection

def common_old_record(entry,v,old_complete):
    original=loadpin(entry['source_assets_checkpoint']);archive=original['archive']
    require(original['source_id']==entry['source_id'] and original['source_index']==entry['source_index']
        and original['tokens_sha256']==entry['source_tokens_sha256'],'Original source asset identity differs')
    i=entry['source_index'];oldcp=Path(v['out'])/'source_checkpoints'/f'{i:04d}.json';oc=read(oldcp)
    verify(oldcp,old_complete['outputs'][str(oldcp)])
    rows=Path(v['out'])/'sources'/f'{i:04d}.json'
    require(oc['source_id']==entry['source_id'] and oc['frame_count']==90 and oc['images_scored'],'Old complete calibration source required')
    require(oc['outputs'][str(rows)]==old_complete['outputs'][str(rows)],'Old source rows not sealed')
    return dict(source_index=i,source_id=entry['source_id'],preprocessing_id=entry['preprocessing_id'],
        tokens_sha256=original['tokens_sha256'],archive=archive,archive_sha256=original['outputs'][archive],
        class_index=original['evaluation_class_index'],old_visual_checkpoint=pin(oldcp),
        old_visual_rows=dict(path=str(rows),sha256=oc['outputs'][str(rows)]),
        old_image_archives={p:h for p,h in oc['outputs'].items() if p.endswith('.npz')})

def check_source_gate(gate):
    require(gate['status']=='T1_SOURCE32_REAL_RECOVERY_GATE_COMPLETE' and gate['source_count']==32
        and gate['whole_prefixes']==[7,8,9] and gate['static_raw_image_pairs']==gate['exact_equal_image_pairs']==96
        and gate['historical_VAR_prefix_roundtrips_reused']==96 and gate['holdout_used'] is False,
        'Actual source32 exact recovery gate required')

def pilot_source(r,i,kind):
    if r['phase']!='full' or i>=100:return None
    done=loadpin(r['pilot_completion']);request=read(done['request_path']);verify(done['request_path'],done['request_sha256'])
    for key in ('families','snrs','candidates','source_model_id','registered_profile_catalogue','noise_rule','counter_rule'):
        require(request[key]==r[key],'Pilot protocol differs: '+key)
    require(request['records'][i]==r['records'][i] and request['environment_request']==r['environment_request'],'Pilot source/environment differs')
    p=str(Path(request['out'])/kind/f'{i:04d}.json');verify(p,done['outputs'][p]);cp=read(p)
    require(cp['request_sha256']==done['request_sha256'],'Pilot source request differs');return cp

def prepare(args):
    require(args.family in FAMILIES+['both'],'Fixed source families only')
    families=FAMILIES if args.family=='both' else [args.family]
    source_done=read(args.source_completion);manifest_path=Path(args.source_completion).parent/'manifest.json'
    verify(manifest_path,source_done['outputs'][str(manifest_path.resolve())]);manifest=read(manifest_path)
    require(source_done['status'] in ('T1_CALIBRATION_COMPLETE_SOURCE_ASSETS_SEALED','T1_CALIBRATION_SOURCE_CPU_PREPARATION_COMPLETE'), 'Completed source preparation required')
    require(manifest['source_count']==1000 and len(manifest['records'])==1000,'Prepare complete originalcal1000 source closure first')
    require([x['source_index']for x in manifest['records']]==list(range(1000)) and [x['source_id']for x in manifest['records']]==manifest['source_ids']
        and len(set(manifest['source_ids']))==1000,'Original1000 source order and identities')
    source_gate=read(args.source_visual_gate);check_source_gate(source_gate)
    require(source_gate['input_bindings'].get(str(Path(args.static_completion).resolve()))==sha(args.static_completion),'Source gate uses another static model')
    for p,h in source_gate['outputs'].items():verify(p,h)
    environment=read(args.environment_request);v=read(environment['original_visual_request']['path'])
    verify(environment['original_visual_request']['path'],environment['original_visual_request']['sha256'])
    old_done=loadpin(environment['original_visual_completion'])
    q=read(args.phy_qualification);require(q['status']=='PASS' and q['packet_decode_count']==241,'Actual241-call qualification first')
    verify(q['request_path'],q['request_sha256']);constructor=read(q['request_path'])['catalogue']
    base=candidates();byid={r['candidate_id']:r for r in base};pilot=None
    if args.phase=='pilot':
        source_count=100;seeds=[4101];grid={f:{str(s):list(byid) for s in SNRS} for f in families}
    else:
        require(args.pilot_completion is not None,'Full stage requires completed pilot')
        pilot=read(args.pilot_completion);require(pilot['status']=='T1_PILOT_COMPLETE' and pilot['families']==families,'Matching complete family pilot first')
        verify(pilot['rankings_path'],pilot['outputs'][pilot['rankings_path']])
        ranking=list(csv.DictReader(Path(pilot['rankings_path']).open(newline='')))
        grid={f:{str(s):[x['candidate_id'] for x in ranking if x['family']==f and int(x['snr_db'])==s and int(x['rank'])<=3] for s in SNRS} for f in families}
        require(all(len(g)==3 for d in grid.values() for g in d.values()),'Three legal candidates/family/SNR')
        source_count=1000;seeds=[4101,4102,4103]
    records=manifest['records'][:source_count];old_records=[]
    for entry in records:
        sr,arrays=load_source(entry)
        require(sr['static_model_completion_sha256']==sha(args.static_completion),'Source streams use another static model')
        for family in families:
            for candidate in base:selected_stream(sr,arrays,family,candidate)
        old_records.append(common_old_record(sr,v,old_done))
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'Independent new output path')
    frame_count=source_count*len(seeds)*sum(len(g) for d in grid.values() for g in d.values())
    files=['t1_calibrate.py','t1_entropy_core.py','t1_source_asset_schema.py','t1_codec_runtime.py','t1_phy.py','t2_ledger.py','t2_pilot.py']
    bindings={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in files}
    r=dict(schema=SCHEMA,phase=args.phase,root=str(root),out=str(out),source_count=source_count,
        families=families,snrs=SNRS,seeds=seeds,candidates=base,grid=grid,frame_count=frame_count,packet_cap=frame_count*2,
        records=records,old_records=old_records,source_completion=pin(args.source_completion),source_manifest=pin(manifest_path),
        source_bindings=bindings,environment_request=pin(args.environment_request),phy_qualification=pin(args.phy_qualification),
        source_visual_gate=pin(args.source_visual_gate),static_completion=pin(args.static_completion),
        source_model_id={f:model_id(f,sha(args.static_completion)) for f in FAMILIES},
        registered_profile_catalogue=constructor,workers=args.workers,deadline_unix=args.deadline_unix,max_seconds=86400,
        noise_rule='t1_phy.standard_noise(source_id,seed), all1024 paid symbols, no family/m/q in noise key',
        counter_rule='((index_in_original_six_SNR_order*1000)+original_source_index)*3+(noise_seed-4101)',
        fallback='target7/8/9 then descending full prefixes to4; actual encoded length only; no raw substitution',
        metric_selection='mean per source DINOv2-L cosine; descending, candidate_id lexical tie break',
        failure_output='exact constant0.5 RGB for header, CRC, framing or entropy-canonical rejection; no truth check',
        old_ledgers_opened=False,holdout_used=False,training_updates=0,
        pilot_completion=None if pilot is None else pin(args.pilot_completion),
        stop_files=[str(root/'STOP'),str(out/'STOP')])
    validate_request(r);save(out/'request.json',r)
    return dict(status='T1_CALIBRATION_REGISTERED_NOT_RUN',phase=args.phase,source_count=source_count,
        families=families,logical_frames=frame_count,packet_cap=frame_count*2,request=pin(out/'request.json'))

def validate_request(r):
    require(r['schema']==SCHEMA and r['snrs']==SNRS and r['phase'] in ('pilot','full'),'Registered finite stage')
    require(r['source_count']==(100 if r['phase']=='pilot' else 1000) and r['seeds']==([4101] if r['phase']=='pilot' else [4101,4102,4103]),'Registered originalcal/noise scope')
    require(r['families'] in ([FAMILIES[0]],[FAMILIES[1]],FAMILIES) and r['candidates']==candidates(),'Frozen family/candidate set')
    require(type(r['workers'])is int and 1<=r['workers']<=16,'Finite CPU workers')
    require([x['source_index']for x in r['records']]==list(range(r['source_count'])) and len({x['source_id']for x in r['records']})==r['source_count'],'Unique original source order')
    cids={x['candidate_id']for x in r['candidates']};count=36 if r['phase']=='pilot' else 3
    require(set(r['grid'])==set(r['families']) and all(set(d)=={str(s)for s in SNRS}for d in r['grid'].values()),'Exact family/SNR grid')
    require(all(len(g)==len(set(g))==count and set(g)<=cids for d in r['grid'].values()for g in d.values()),'Finite unique legal candidate grid')
    require(r['frame_count']==r['source_count']*len(r['seeds'])*len(r['families'])*len(SNRS)*count and r['packet_cap']==2*r['frame_count'],'Finite frame/packet budget')

def registered(path):
    r=read(path);validate_request(r)
    for p,h in r['source_bindings'].items():verify(p,h)
    check_source_gate(loadpin(r['source_visual_gate']));loadpin(r['static_completion']);loadpin(r['phy_qualification'])
    return r,sha(path)

def guard(r,started):
    require(not STOP and not shared.STOP and time.time()<r['deadline_unix'] and time.monotonic()-started<r['max_seconds'],'Stopped/deadline; preserve checkpoints')
    require(not any(Path(p).exists() for p in r['stop_files']),'Explicit safe stop')

def cpu_worker(path,index):
    import t1_phy as phy
    r,rh=registered(path);out=Path(r['out']);require(0<=index<r['workers'],'Registered worker index')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only actual LDPC worker')
    started=time.monotonic()
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,stop)
    with shared.lock(out/f'cpu_worker_{index}.lock'):
        rt=phy.create_runtime(r['root'],r['phy_qualification']['path']);ledger=Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap'])
        require(rt.catalogue==r['registered_profile_catalogue'],'Registered paid profile catalogue changed')
        profiles={(p['family'],p['m'],p['q'],p['nominal_rate']):p for p in rt.catalogue['profiles']}
        cands={c['candidate_id']:c for c in r['candidates']};completed=[]
        for entry in r['records'][index::r['workers']]:
            guard(r,started);i=entry['source_index'];cp=out/'cpu_sources'/f'{i:04d}.json'
            if cp.exists():
                previous=read(cp);require(previous['request_sha256']==rh,'Different completed CPU source');completed.append(pin(cp));continue
            sr,arrays=load_source(entry);frames=[]
            previous_pilot=pilot_source(r,i,'cpu_sources');pilot_packets={}
            if previous_pilot is not None:
                for oldframe in previous_pilot['frames']:
                    oldpacket=loadpin(oldframe['physical_frame'])
                    pilot_packets[(oldpacket['snr_db'],oldpacket['noise_seed'],oldpacket['TX_profile_id'])]=oldpacket
            for snr in SNRS:
                for seed in r['seeds']:
                    counter=([1,4,7,10,13,19].index(snr)*1000+i)*3+(seed-4101)
                    noise=phy.standard_noise(entry['source_id'],seed)*10**(-snr/20)
                    for family in r['families']:
                        for cid in r['grid'][family][str(snr)]:
                            guard(r,started);c=cands[cid];m,bits,selection=selected_stream(sr,arrays,family,c)
                            p=profiles[(family,m,c['q'],c['nominal_rate'])];pid=p['profile_id']
                            actual=out/'physical_frames'/f'{i:04d}'/f'{snr}_{seed}_{pid:03d}.json'
                            if actual.exists():
                                packet=read(actual);require(packet['request_sha256']==rh and packet['payload_sha256']==phy.array_sha(bits),'Changed actual frame cache')
                            else:
                                wave,tx=rt.transmit(pid,bits,counter);observed=wave+noise
                                prior=pilot_packets.get((snr,seed,pid))
                                if prior is None:
                                    event=f'{SCHEMA}/{r["phase"]}/source{i}/snr{snr}/noise{seed}/profile{pid}'
                                    rx=rt.receive(observed,snr,counter,rt.profiles,ledger,event,phase='calibration');reused=False
                                else:
                                    require(prior['payload_sha256']==phy.array_sha(bits) and prior['noise_sha256']==phy.array_sha(noise)
                                        and prior['observation_sha256']==phy.array_sha(observed) and prior['public_frame_counter']==counter
                                        and prior['transmission']==tx,'Pilot actual physical reception mismatch')
                                    rx=prior['actual_RX'];reused=True
                                packet=dict(request_sha256=rh,source_index=i,source_id=entry['source_id'],snr_db=snr,noise_seed=seed,
                                    TX_profile_id=pid,payload_sha256=phy.array_sha(bits),noise_sha256=phy.array_sha(noise),
                                    observation_sha256=phy.array_sha(observed),public_frame_counter=counter,transmission=tx,actual_RX=rx,
                                    pilot_actual_RX_reused=reused)
                                save(actual,packet)
                            frames.append(dict(source_index=i,source_id=entry['source_id'],snr_db=snr,noise_seed=seed,
                                family=family,candidate_id=cid,target_m=c['target_m'],actual_m=m,q=c['q'],nominal_rate=c['nominal_rate'],
                                fallback_attempts=selection['attempts'],physical_frame=pin(actual),actual_packet_reuse='same_source_noise_actual_profile'))
            save(cp,dict(status='T1_CPU_SOURCE_COMPLETE',request_sha256=rh,source_index=i,source_id=entry['source_id'],frames=frames))
            completed.append(pin(cp));print(json.dumps(dict(stage='T1_CPU',worker=index,source=i,logical_frames=len(frames))),flush=True)
        save(out/'cpu_workers'/f'{index:02d}.json',dict(status='COMPLETE',request_sha256=rh,sources=completed));ledger.close()

def receiver_state(flat,m):
    flat=np.asarray(flat);require(flat.shape==(OFFSETS[m],) and np.issubdtype(flat.dtype,np.integer)
        and ((flat>=0)&(flat<4096)).all(),'Actual entropy recovered tokens must match the received prefix')
    return dict(K=0,kind='tokens',m=m,order='raster',partial_values=[],prefix=[flat[OFFSETS[j]:OFFSETS[j+1]].tolist() for j in range(m)])

def gray_state():return dict(K=0,kind='gray',m=0,order='raster',partial_values=[],prefix=[])

def recover(sr,arrays,packet,codec,models):
    rx=packet['actual_RX'];body=rx['body']
    if not rx['header']['header_ok'] or body is None or not body['crc_accepted'] or not body['parser_accepted']:
        return gray_state(),dict(source_status=rx['status'],source_decode_cache_hit=False,new_VAR_source_decode=0)
    p=rx['rx_profile'];family=p['family'];m=p['m'];bits=body['payload']
    entry=sr['streams'].get(family,{}).get(str(m));restricted={} if entry is None else{k:arrays[k]for k in(entry['bits_key'],entry['received_tokens_key'])}
    decoded=exact_received_cache(sr,restricted,family=family,m=m,received_bits=bits,expected_source_model_id=models[family])
    if decoded is not None:return receiver_state(decoded['received_tokens'],m),dict(source_status='ARITHMETIC_SOURCE_DECODED',source_decode_cache_hit=True,new_VAR_source_decode=0)
    try:decoded=codec.decode(family,bits,m)
    except InvalidSourceStream as error:
        return gray_state(),dict(source_status='SOURCE_PARSE_REJECT',source_decode_cache_hit=False,new_VAR_source_decode=int(family=='EC_VAR_WHOLE'),parse_reason=str(error))
    return receiver_state(decoded['received_tokens'],m),dict(source_status='ARITHMETIC_SOURCE_DECODED',source_decode_cache_hit=False,new_VAR_source_decode=int(family=='EC_VAR_WHOLE'))

def recovery_input(packet,models):
    rx=packet['actual_RX'];b=rx['body'];p=rx['rx_profile']
    accepted=bool(rx['header']['header_ok'] and b is not None and b['crc_accepted'] and b['parser_accepted'])
    return dict(header_ok=rx['header']['header_ok'],status=rx['status'],actual_received_profile=None if p is None else dict(p),
        crc_accepted=None if b is None else b['crc_accepted'],parser_accepted=None if b is None else b['parser_accepted'],
        received_payload=list(b['payload'])if accepted else None,source_model_id=models[p['family']]if accepted else None)

def recover_once(directory,rh,sr,arrays,packet,codec,models,prior=None):
    """Durable actual-received source decode; no TX metadata enters the key."""
    identity=recovery_input(packet,models);path=Path(directory)/(digest(identity)+'.json');reservation=path.with_suffix('.reserved.json')
    if path.exists():
        value=read(path);require(value['request_sha256']==rh and value['recovery_input']==identity,'Changed source decoding checkpoint')
        return value['state'],value['evidence'],pin(path)
    require(not reservation.exists(),'Unresolved actual source decode; no automatic repeat')
    old=None if prior is None else prior.get(digest(identity))
    if old is None:
        save(reservation,dict(request_sha256=rh,recovery_input=identity));state,evidence=recover(sr,arrays,packet,codec,models)
        parent=None
    else:
        previous=loadpin(old);require(previous['recovery_input']==identity,'Parent actual source decode differs')
        state=previous['state'];evidence=dict(previous['evidence'],new_VAR_source_decode=0,source_decode_checkpoint_reused=True);parent=old
    save(path,dict(request_sha256=rh,recovery_input=identity,state=state,evidence=evidence,parent=parent))
    return state,evidence,pin(path)

def gpu_worker(path):
    r,rh=registered(path);out=Path(r['out']);env=loadpin(r['environment_request']);started=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Registered GPU0')
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,stop)
    with shared.lock(env['visual_config']['visual_lock']),shared.lock(out/'gpu.lock'):
        native,source,scorer=shared.gpu_build(env,out)
        codec=SourceCodec(r['root'],r['static_completion']['path'],native)
        for entry,oldrec in zip(r['records'],r['old_records']):
            guard(r,started);i=entry['source_index'];cp=out/'gpu_sources'/f'{i:04d}.json'
            if cp.exists():require(read(cp)['request_sha256']==rh,'Different completed GPU source');continue
            physical=read(out/'cpu_sources'/f'{i:04d}.json');require(physical['request_sha256']==rh,'CPU source request changed')
            sr,arrays=load_source(entry);prior=shared.old_rows(oldrec);cache={};old_archives={}
            for old in prior:
                key=shared.state_key(old)
                if key in cache:continue
                archive=old['image_archive']
                if archive not in old_archives:
                    verify(archive,oldrec['old_image_archives'][archive])
                    with np.load(archive,allow_pickle=False) as z:old_archives[archive]=z['images'].copy()
                image=old_archives[archive][old['image_slot']];require(shared.image_sha(image)==old['image_sha256'],'Original scored image changed')
                cache[key]=dict(metrics={m:old[m] for m in METRICS},image_sha256=old['image_sha256'],image_path=archive,image_slot=old['image_slot'],reuse='EXACT_OLD_RECEIVED_STATE_AND_SCORE')
            previous_pilot=pilot_source(r,i,'gpu_sources');prior_decodes={}
            if previous_pilot is not None:
                for descriptor in previous_pilot['source_decode_checkpoints']:
                    previous=loadpin(descriptor);prior_decodes[digest(previous['recovery_input'])]=descriptor
                for row in previous_pilot['rows']:
                    key=row['received_state_sha256']
                    if key in cache:continue
                    archive=row['image_path']
                    if archive not in old_archives:
                        with np.load(archive,allow_pickle=False)as z:old_archives[archive]=z['images'].copy()if'images'in z.files else z['image'][None].copy()
                    image=old_archives[archive][row['image_slot']];require(shared.image_sha(image)==row['image_sha256'],'Pilot scored image changed')
                    cache[key]=dict(metrics={m:row[m]for m in METRICS},image_sha256=row['image_sha256'],image_path=archive,image_slot=row['image_slot'],reuse='EXACT_PILOT_RECEIVED_STATE_AND_SCORE')
            _,pixels=shared.source_assets(oldrec);record=shared.score_record(oldrec)
            scorer.prepare_source(record,pixels.astype(np.float32)/np.float32(255),i)
            rows=[];recovered={};new_render=0;decode_pins={}
            for frame in physical['frames']:
                guard(r,started);packet_path=frame['physical_frame']['path']
                if packet_path not in recovered:
                    packet=loadpin(frame['physical_frame']);state,evidence,decode_pin=recover_once(out/'source_decodes'/f'{i:04d}',rh,sr,arrays,packet,codec,r['source_model_id'],prior_decodes)
                    decode_pins[decode_pin['path']]=decode_pin
                    recovered[packet_path]=(packet,state,evidence)
                packet,state,evidence=recovered[packet_path];key=digest(state)
                if key not in cache:
                    state_path=out/'gpu_states'/f'{i:04d}'/(key+'.json');image_path=state_path.with_suffix('.npz')
                    if state_path.exists():
                        value=read(state_path);require(value['request_sha256']==rh and value['state']==state,'Changed state checkpoint');verify(image_path,value['image_file_sha256'])
                    else:
                        reserved=state_path.with_suffix('.reserved.json');require(not reserved.exists(),'Unresolved render attempt requires inspection')
                        save(reserved,dict(request_sha256=rh,state=state,source_id=entry['source_id']))
                        image=shared.render_state(native,source,state);metrics=scorer(record,[image])[0]
                        with image_path.open('xb') as f:np.savez(f,image=image)
                        value=dict(request_sha256=rh,state=state,metrics={m:metrics[m] for m in METRICS},image_path=str(image_path),image_slot=0,
                            image_sha256=shared.image_sha(image),image_file_sha256=sha(image_path),reuse='NEW_ACTUAL_RECEIVED_STATE',new_VAR_render_calls=int(state['kind']!='gray'))
                        save(state_path,value);new_render+=int(state['kind']!='gray')
                    cache[key]=value
                value=cache[key];body=packet['actual_RX']['body'];tx=packet['transmission']
                rows.append(dict(source_index=i,source_id=entry['source_id'],snr_db=frame['snr_db'],noise_seed=frame['noise_seed'],
                    family=frame['family'],candidate_id=frame['candidate_id'],target_m=frame['target_m'],actual_m=frame['actual_m'],q=frame['q'],nominal_rate=frame['nominal_rate'],
                    header_ok=packet['actual_RX']['header']['header_ok'],body_crc_accept=None if body is None else body['crc_accepted'],
                    received_family=None if packet['actual_RX']['rx_profile']is None else packet['actual_RX']['rx_profile']['family'],
                    received_m=None if packet['actual_RX']['rx_profile']is None else packet['actual_RX']['rx_profile']['m'],
                    received_profile_id=None if packet['actual_RX']['rx_profile']is None else packet['actual_RX']['rx_profile']['profile_id'],
                    physical_status=packet['actual_RX']['status'],pilot_actual_RX_reused=packet['pilot_actual_RX_reused'],
                    gray=state['kind']=='gray',source_status=evidence['source_status'],source_decode_cache_hit=evidence['source_decode_cache_hit'],
                    E_frame=tx['E_frame'],rho=tx['rho'],arithmetic_bits=tx['arithmetic_bits'],known_information_padding_bits=tx['known_information_padding_bits'],
                    received_state_sha256=key,image_sha256=value['image_sha256'],image_path=value['image_path'],image_slot=value['image_slot'],
                    visual_reuse=value['reuse'],**value['metrics']))
            save(cp,dict(status='T1_GPU_SOURCE_COMPLETE',request_sha256=rh,source_index=i,source_id=entry['source_id'],rows=rows,new_VAR_render_calls_this_attempt=new_render,
                cpu_source=pin(out/'cpu_sources'/f'{i:04d}.json'),source_decode_checkpoints=list(decode_pins.values()),
                new_VAR_source_decode_calls=sum(loadpin(d)['evidence']['new_VAR_source_decode']for d in decode_pins.values()),
                new_VAR_render_calls_total=sum(v.get('new_VAR_render_calls',0)for v in cache.values()if v.get('request_sha256')==rh)))
            print(json.dumps(dict(stage='T1_GPU',source=i,rows=len(rows),new_VAR_renders=new_render)),flush=True)
        native.frozen()

def csv_save(path,rows):
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    value=stream.getvalue().encode();path=Path(path)
    if path.exists():require(path.read_bytes()==value,'Changed completed CSV')
    else:path.write_bytes(value)

def ledger_snapshot(path,rh,cap):
    path=Path(path);require(path.is_file(),'Missing actual packet ledger; never recreate during close')
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        require(db.execute('SELECT hash,cap FROM config WHERE id=1').fetchone()==(rh,cap),'Actual packet ledger request/cap changed')
        total=db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        unresolved=db.execute("SELECT COUNT(*) FROM events WHERE status!='COMPLETE'").fetchone()[0]
        require(total<=cap and unresolved==0,'Unresolved or excessive actual packet callbacks')
        return dict(total=total,unresolved=unresolved,cap=cap)
    finally:db.close()

def close(path):
    r,rh=registered(path);out=Path(r['out']);rows=[];outputs={};packet_calls=0;source_decodes=0;VAR_renders=0
    for entry in r['records']:
        p=out/'gpu_sources'/f'{entry["source_index"]:04d}.json';v=read(p);require(v['request_sha256']==rh,'GPU source request differs');rows+=v['rows'];outputs[str(p)]=sha(p)
        source_decodes+=v['new_VAR_source_decode_calls'];VAR_renders+=v['new_VAR_render_calls_total']
        cpu=loadpin(v['cpu_source']);require(cpu['request_sha256']==rh,'CPU source request differs');outputs[v['cpu_source']['path']]=v['cpu_source']['sha256']
        for frame in cpu['frames']:
            packet=loadpin(frame['physical_frame'])
            if frame['physical_frame']['path']not in outputs and not packet['pilot_actual_RX_reused']:packet_calls+=1+int(packet['actual_RX']['header']['header_ok'])
            outputs[frame['physical_frame']['path']]=frame['physical_frame']['sha256']
        for descriptor in v['source_decode_checkpoints']:
            decoded=loadpin(descriptor);require(decoded['request_sha256']==rh,'Source decode request differs');outputs[descriptor['path']]=descriptor['sha256']
    expected={(f,s,n,e['source_index'],c) for f in r['families'] for s in SNRS for n in r['seeds'] for e in r['records'] for c in r['grid'][f][str(s)]}
    require(len(rows)==r['frame_count'] and {(x['family'],x['snr_db'],x['noise_seed'],x['source_index'],x['candidate_id']) for x in rows}==expected,'All conditions including failures must be retained')
    ranking=[]
    for family in r['families']:
        for snr in SNRS:
            group=[]
            for cid in r['grid'][family][str(snr)]:
                values=[x for x in rows if x['family']==family and x['snr_db']==snr and x['candidate_id']==cid]
                require(len(values)==r['source_count']*len(r['seeds']),'Balanced complete calibration grid')
                c=next(c for c in r['candidates'] if c['candidate_id']==cid)
                bysource={e['source_index']:[]for e in r['records']}
                for x in values:bysource[x['source_index']].append(x)
                require(all(len(v)==len(r['seeds'])for v in bysource.values()),'Three noises per original source')
                group.append(dict(family=family,snr_db=snr,candidate_id=cid,target_m=c['target_m'],q=c['q'],nominal_rate=c['nominal_rate'],
                    source_count=r['source_count'],noise_count=len(r['seeds']),**{m:math.fsum(math.fsum(float(x[m])for x in v)/len(v)for v in bysource.values())/len(bysource) for m in METRICS},
                    gray_fraction=sum(x['gray'] for x in values)/len(values),body_crc_rejected_fraction=sum(x['body_crc_accept'] is False for x in values)/len(values),
                    fallback_fraction=sum(x['actual_m']<x['target_m'] for x in values)/len(values)))
            group.sort(key=lambda v:(-v['dinov2_vitl14_cosine'],v['candidate_id']));ranking.extend(dict(v,rank=i+1) for i,v in enumerate(group))
    budget=ledger_snapshot(out/'packet_ledger.sqlite',rh,r['packet_cap']);require(budget['total']==packet_calls,'Physical receive receipts and actual packet ledger count differ')
    for n,values in [('per_frame.csv',rows),('calibration_rankings.csv',ranking)]:csv_save(out/n,values);outputs[str(out/n)]=sha(out/n)
    status='T1_PILOT_COMPLETE' if r['phase']=='pilot' else 'T1_FULL_CALIBRATION_COMPLETE'
    done=dict(status=status,request_sha256=rh,request_path=str(Path(path).resolve()),families=r['families'],source_count=r['source_count'],noise_count=len(r['seeds']),
        frame_count=len(rows),rankings_path=str(out/'calibration_rankings.csv'),ledger=budget,outputs=outputs,holdout_used=False,training_updates=0,
        new_VAR_source_decode_calls=source_decodes,new_VAR_render_calls=VAR_renders,
        full_test_complete=False,policy_selection_metric='dinov2_vitl14_cosine_only')
    save(out/'completion.json',done);return done

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for n in ('root','source-completion','phy-qualification','environment-request','source-visual-gate','static-completion','out'):p.add_argument('--'+n,required=True)
    p.add_argument('--phase',choices=('pilot','full'),required=True);p.add_argument('--family',choices=FAMILIES+['both'],default='both')
    p.add_argument('--pilot-completion');p.add_argument('--workers',type=int,default=8);p.add_argument('--deadline-unix',type=float,required=True)
    for name in ('cpu-worker','gpu-worker','close'):
        p=sub.add_parser(name);p.add_argument('--request',required=True)
        if name=='cpu-worker':p.add_argument('--index',type=int,required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args)
    elif args.command=='cpu-worker':result=cpu_worker(args.request,args.index)
    elif args.command=='gpu-worker':result=gpu_worker(args.request)
    else:result=close(args.request)
    if result is not None:print(json.dumps(result),flush=True)

if __name__=='__main__':main()
