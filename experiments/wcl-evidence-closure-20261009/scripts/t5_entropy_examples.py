"""Fixed development16 entropy examples: actual source/PHY/recovery, no scoring.

New, separate 48-frame / 96-packet experiment. All policies come from calibration;
neither holdout quality nor display quality enters family/source/noise selection.
"""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,read,require,sha,verify,write,choose_arithmetic,source_tokens
from t1_holdout_metadata import SNRS,FAMILIES,pin,pinned
from t1_codec_runtime import SourceCodec
from t1_source_check import MODELS
from t1_source_asset_schema import build_record,token_sha,model_id
from t1_holdout_render import build_native
from t2_ledger import Ledger
import t1_calibrate as cal
import t2_pilot as shared
write=cal.save

INDICES=[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
SEED=6201
H_SHA='5483c4acee7acd3a6082dc992a1b68e4e677db004b42c9ae64a43a09c6dcc3de'

def old_source(done,index):
    e=done['records'][index];require(e['source_index']==index,'Historical source index differs')
    verify(e['checkpoint'],e['sha256']);require(done['outputs'][e['checkpoint']]==e['sha256'],'Historical source checkpoint not sealed')
    cp=read(e['checkpoint'])
    require(cp['status']=='H_DEVELOPMENT100_SOURCE_CODEC_SOURCE_COMPLETE' and cp['source_index']==index
        and cp['source_id']==e['source_id'] and cp['independent_roundtrip'] and cp['encoder_tokens_verified']
        and cp['new_canonical_roundtrips']==4 and cp['registration_sha256']==done['registration_sha256'],
        'Actual independent development source roundtrips required')
    asset=pinned(cp['source_assets_checkpoint']);path=asset['archive'];verify(path,asset['outputs'][path])
    with np.load(path,allow_pickle=False) as z:tokens=source_tokens(z['tokens']).copy();pixels=z['pixels'].copy()
    require(token_sha(tokens)==cp['tokens_sha256']==asset['tokens_sha256'] and asset['source_id']==cp['source_id']
        and asset['source_index']==index and pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
        and hashlib.sha256(pixels.tobytes()).hexdigest()==cp['preprocessing_id']==asset['preprocessing_id'],
        'Original development tokens/pixels changed')
    require(len(cp['outputs'])==1,'Historical stream archive mapping differs');path,digest=next(iter(cp['outputs'].items()))
    verify(path,digest);require(done['outputs'][path]==digest,'Historical stream archive unsealed')
    lengths={x['m']:x for x in cp['lengths']};require(set(lengths)=={6,7,8,9},'Actual complete m6..9 required')
    with np.load(path,allow_pickle=False) as z:streams={m:z[f'm{m}_bits'].copy() for m in lengths}
    for m,bits in streams.items():
        row=lengths[m];require(bits.dtype==np.uint8 and bits.ndim==1 and np.isin(bits,(0,1)).all()
            and len(bits)==row['arithmetic_bits'] and row['raw_bits']==12*OFFSETS[m]
            and row['zero_extension_reads']==30 and 2<=row['flush_bits']<=len(bits),'Historical canonical bit evidence differs')
    return tokens,pixels,streams,cp,e

def record_write(r,index,tokens,pixels,streams,upstream,stage):
    family=r['family'];prefix='static' if family==FAMILIES[0] else 'var';out=Path(r['out'])/stage
    arrays=dict(tokens=tokens,pixels=pixels)
    for m,bits in streams.items():
        arrays[f'{prefix}_m{m}_bits']=bits;arrays[f'{prefix}_m{m}_received_tokens']=tokens[:OFFSETS[m]].copy()
    archive=out/'arrays'/f'{index:04d}.npz';archive.parent.mkdir(parents=True,exist_ok=True)
    require(not archive.exists(),'Preserve existing source array, no silent overwrite')
    with archive.open('xb') as f:np.savez(f,**arrays)
    cp=build_record(source_index=index,source_id=upstream['source_id'],tokens=tokens,preprocessing_id=upstream['preprocessing_id'],
        source_assets_checkpoint=upstream['source_assets_checkpoint'],archive=archive,arrays=arrays,
        static_completion_sha=r['static_completion']['sha256'],origins={f:'T5_ACTUAL_STATIC_OR_SEALED_H_WITH_EXPLICIT_SHORT_PREFIXES' for f in FAMILIES},
        upstream_evidence={f:upstream for f in FAMILIES})
    cp.update(source_role='development_fixed16',family=r['family'],family_freeze=r['family_freeze'],holdout_used=False)
    path=out/'checkpoints'/f'{index:04d}.json';write(path,cp);return dict(source_index=index,source_id=cp['source_id'],checkpoint=str(path),checkpoint_sha256=sha(path))

def load(entry):
    verify(entry['checkpoint'],entry['checkpoint_sha256']);cp=read(entry['checkpoint'])
    verify(cp['archive']['path'],cp['archive']['sha256'])
    with np.load(cp['archive']['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    require(token_sha(arrays['tokens'])==cp['source_tokens_sha256'],'Prepared source tokens differ');return cp,arrays

def selections(r,cp):
    lengths={int(m):e['payload_bits'] for m,e in cp['streams'][r['family']].items()}
    return {str(s):choose_arithmetic(lengths,r['policies'][str(s)],minimum_m=4) for s in SNRS}

def prepare(a):
    family=pinned(pin(a.family_freeze));policy=pinned(family['original_policy_freeze'])
    require(family['status']=='T1_SINGLE_ENTROPY_FAMILY_FROZEN_CALIBRATION_ONLY' and not family['holdout_used_for_selection']
        and family['holdout_files_read'] is False and family['family'] in FAMILIES,'Single calibration-only family freeze required')
    verify(a.h_completion,H_SHA);old=read(a.h_completion)
    require(old['status']=='H_DEVELOPMENT100_SOURCE_CODEC_COMPLETE' and old['source_count']==100 and old['independent_roundtrip'],
        'Original development100 source completion required')
    require(all(old['frozen_visual_identity']['models'][k]==v for k,v in MODELS.items()),'Original source models differ')
    env=read(a.environment_request);out=Path(a.out).resolve();root=Path(a.root).resolve()
    require(env['root']==str(root) and out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009') and not out.exists(),'Fresh WCL output required')
    gate=pinned(pin(a.source_gate));require(gate['status']=='T1_SOURCE32_REAL_RECOVERY_GATE_COMPLETE' and gate['exact_equal_image_pairs']==96,'Completed exact renderer qualification required')
    protocol=policy['protocol'];qual=pinned(protocol['phy_qualification']);require(qual['status']=='PASS' and qual['packet_decode_count']==241,'Actual N1024 PHY qualification required')
    static=pinned(protocol['static_completion']);require(static['source_count']==20000 and not static['calibration_used_for_fitting'],'Original training static model required')
    original_path=root/'outputs/MAIN-RAW64-20261007/development_same_rx_images_r2/completion.json';original=read(original_path)
    require(original['status']=='MAIN_RAW64_DEVELOPMENT_SAME_RX_IMAGES_COMPLETE_V1' and original['source_count']==100
        and original['population_role']=='development' and original['frozen_visual_identity']==env['old_visual_identity']
        and original['numerical_runtime']==env['old_numerical_runtime'],'Completed original development renderer/model identity required')
    out.mkdir(parents=True)
    r=dict(schema='T5_ENTROPY_FIXED16_ACTUAL_V1',root=str(root),out=str(out),family=family['family'],family_freeze=pin(a.family_freeze),
        policies=family['policies'],original_policy_freeze=family['original_policy_freeze'],h_completion=pin(a.h_completion),
        environment_request=pin(a.environment_request),source_gate=pin(a.source_gate),static_completion=protocol['static_completion'],
        phy_qualification=protocol['phy_qualification'],source_model_id=protocol['source_model_id'],registered_profile_catalogue=protocol['registered_profile_catalogue'],
        source_indices=INDICES,snrs=SNRS,noise_seed=SEED,N=1024,frame_count=48,packet_cap=96,
        population='development_fixed16',holdout_used=False,policy_selection=False,new_metric_calls=0,new_bootstrap_calls=0,
        deadline_unix=a.deadline_unix,max_seconds=86400,stop_files=[str(root/'STOP'),str(out/'STOP')],records=[],pending_short_sources=[],
        noise_rule='fixed source_id and seed6201, t1_phy.standard_noise, no source/method-specific quality selection',
        counter_rule='(six_SNR_index*100+historical_source_index)*3 + noise_seed-6201',original_images=pin(original_path),old_reuse={})
    codec=SourceCodec(root,static_completion=protocol['static_completion']['path']);static_calls=0
    for i in INDICES:
        tokens,pixels,streams,cp,e=old_source(old,i)
        if r['family']==FAMILIES[0]:
            encoded=codec.encode(r['family'],tokens,modes=tuple(range(4,10)));streams={m:v['bits'] for m,v in encoded.items()}
            for m,bits in streams.items():
                rx=codec.decode(r['family'],bits,m);require(np.array_equal(rx['received_tokens'],tokens[:OFFSETS[m]]),'Actual static roundtrip differs');static_calls+=1
        upstream=dict(source_id=cp['source_id'],preprocessing_id=cp['preprocessing_id'],source_assets_checkpoint=cp['source_assets_checkpoint'],
            original_H_source=e,original_H_completion=r['h_completion'],independent_roundtrip_reused=r['family']==FAMILIES[1])
        entry=record_write(r,i,tokens,pixels,streams,upstream,'cpu_source_assets');r['records'].append(entry)
        old_rows_path=original_path.parent/'sources'/f'{i:04d}.json';verify(old_rows_path,original['outputs'][str(old_rows_path)])
        reusable={}
        for row in read(old_rows_path):
            state=row['receiver_state'];proof=row['same_RX_visual_proof']
            require(row['source_index']==i and row['source_id']==cp['source_id'],'Original received image source differs')
            if state['K']!=0:continue
            key=cal.digest(state);im=row['same_RX_images']['VAR_completion']
            require(key==row['actual_state_sha256']==proof['actual_state_sha256'] and proof['same_received_information']
                and not proof['TX_truth_used_for_generation'] and not proof['target_used_for_generation']
                and proof['model_identity_sha256']==cal.digest(original['model_identity'])
                and proof['image_sha256']['VAR_completion']==im['image_sha256'],'Original received-state/renderer evidence differs')
            value=dict(state=state,archive=dict(path=im['image_archive'],sha256=original['outputs'][im['image_archive']]),
                image_slot=im['image_slot'],image_sha256=im['image_sha256'],actual_rows=pin(old_rows_path))
            if key in reusable:require(reusable[key]['image_sha256']==value['image_sha256'],'Same state has inconsistent original image')
            else:reusable[key]=value
        rp=out/'old_reuse'/f'{i:04d}.json';write(rp,reusable);r['old_reuse'][str(i)]=pin(rp)
        source,_=load(entry);selected=selections(r,source)
        require(all(v['status'] in ('SOURCE_LENGTH_FITS_LAYOUT_PENDING','BLOCKED_MISSING_PREFIX') for v in selected.values()),'Unexpected source selection status')
        if any(v['status']=='BLOCKED_MISSING_PREFIX' for v in selected.values()):r['pending_short_sources'].append(i)
    r['new_static_roundtrips']=static_calls;r['source_ids']=[e['source_id'] for e in r['records']]
    files=['t5_entropy_examples.py','t1_select_entropy_family.py','t1_entropy_core.py','t1_codec_runtime.py','t1_source_asset_schema.py',
        't1_holdout_metadata.py','t1_holdout_render.py','t1_calibrate.py','t1_phy.py','t2_pilot.py','t2_ledger.py']
    r['source_bindings']={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in files}
    write(out/'request.json',r);return dict(status='T5_FIXED16_REGISTERED',pending_short_sources=r['pending_short_sources'],new_static_roundtrips=static_calls)

def registered(path):
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,shared.stop)
    r=read(path);require(r['schema']=='T5_ENTROPY_FIXED16_ACTUAL_V1' and r['source_indices']==INDICES and r['snrs']==SNRS
        and r['noise_seed']==6201 and r['frame_count']==48 and r['packet_cap']==96 and not r['holdout_used'],'Fixed16 scope differs')
    for p,h in r['source_bindings'].items():verify(p,h)
    pinned(r['family_freeze']);return r,sha(path)

def source_stage(path):
    r,rh=registered(path);out=Path(r['out']);require(not(out/'sources_complete.json').exists(),'Source stage already complete')
    env=pinned(r['environment_request']);started=time.monotonic();native=None;records=[];calls=0
    def guard():
        shared.guard(r,started)
        if native is not None:native.health_polling.next_check=0.;native.common.check()
    def process():
        nonlocal calls
        codec=SourceCodec(r['root'],static_completion=r['static_completion']['path'],native=native)
        for entry in r['records']:
            guard();i=entry['source_index'];cp,arrays=load(entry)
            prefix='static' if r['family']==FAMILIES[0] else 'var'
            streams={int(m):arrays[v['bits_key']] for m,v in cp['streams'][r['family']].items()}
            if i in r['pending_short_sources']:
                reservation=out/'source_calls'/f'{i:04d}.reserved.json';require(not reservation.exists(),'Unresolved source traversal, do not automatically repeat')
                write(reservation,dict(request_sha256=rh,source_index=i,VAR_TX_upper=1,VAR_RX_upper=2))
                encoded=codec.encode(r['family'],arrays['tokens'],modes=(4,5));calls+=1
                for m,e in encoded.items():
                    guard();rx=codec.decode(r['family'],e['bits'],m);calls+=1
                    require(np.array_equal(rx['received_tokens'],arrays['tokens'][:OFFSETS[m]]),'Independent short-prefix roundtrip differs');streams[m]=e['bits']
                write(reservation.with_name(f'{i:04d}.complete.json'),dict(request_sha256=rh,VAR_TX=1,VAR_RX=2))
            upstream=dict(source_id=cp['source_id'],preprocessing_id=cp['preprocessing_id'],source_assets_checkpoint=cp['source_assets_checkpoint'],
                prepared_source=entry,independent_short_prefixes_added=i in r['pending_short_sources'])
            e=record_write(r,i,arrays['tokens'],arrays['pixels'],streams,upstream,'source_assets');ready,_=load(e)
            require(all(v['status']=='SOURCE_LENGTH_FITS_LAYOUT_PENDING' for v in selections(r,ready).values()),'Actual frozen fallback chain incomplete');records.append(e)
    if r['pending_short_sources']:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit GPU0 source owner required')
        with shared.lock(env['visual_config']['visual_lock']),shared.lock(out/'source_gpu.lock'):
            native,_=build_native(env,out);process();native.frozen()
    else:process()
    done=dict(status='T5_FIXED16_SOURCE_COMPLETE',request_sha256=rh,records=records,source_count=16,new_VAR_source_calls=calls,new_metric_calls=0)
    write(out/'sources_complete.json',done);return done

def cpu(path):
    import t1_phy as phy
    r,rh=registered(path);out=Path(r['out']);require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only LDPC owner required')
    source=read(out/'sources_complete.json');require(source['request_sha256']==rh,'Completed source request differs')
    started=time.monotonic()
    with shared.lock(out/'cpu.lock'):
        rt=phy.create_runtime(r['root'],r['phy_qualification']['path']);require(rt.catalogue==r['registered_profile_catalogue'],'Paid profile catalogue differs')
        ledger=Ledger(out/'packet_ledger.sqlite',rh,96);profiles={(p['family'],p['m'],p['q'],p['nominal_rate']):p for p in rt.catalogue['profiles']};frames=[]
        for entry in source['records']:
            cp,arrays=load(entry);choices=selections(r,cp);i=entry['source_index']
            for snr in SNRS:
                shared.guard(r,started);c=r['policies'][str(snr)];selected=choices[str(snr)];m=selected['actual_m'];bits=arrays[cp['streams'][r['family']][str(m)]['bits_key']]
                p=profiles[r['family'],m,c['q'],c['nominal_rate']];counter=([1,4,7,10,13,19].index(snr)*100+i)*3
                target=out/'physical_frames'/f'{i:04d}_{snr}.json'
                if target.exists():packet=read(target);require(packet['request_sha256']==rh,'Existing packet request differs')
                else:
                    wave,tx=rt.transmit(p['profile_id'],bits,counter);noise=phy.standard_noise(entry['source_id'],6201)*10**(-snr/20);obs=wave+noise
                    rx=rt.receive(obs,snr,counter,rt.profiles,ledger,f'T5_FIXED16/source{i}/snr{snr}/noise6201',phase='development')
                    packet=dict(request_sha256=rh,source_index=i,source_id=entry['source_id'],snr_db=snr,noise_seed=6201,family=r['family'],
                        actual_RX=rx,transmission=tx,public_frame_counter=counter,payload_sha256=phy.array_sha(bits),noise_sha256=phy.array_sha(noise),
                        observation_sha256=phy.array_sha(obs),target_m=c['target_m'],actual_m=m,candidate_id=c['candidate_id'])
                    write(target,packet)
                frames.append(dict(source_index=i,snr_db=snr,physical_frame=pin(target)))
        snap=ledger.snapshot();ledger.close();require(snap['unresolved']==0 and len(frames)==48,'Actual full48 and resolved ledger required')
        write(out/'cpu_complete.json',dict(status='T5_FIXED16_PHY_COMPLETE',request_sha256=rh,frames=frames,ledger=snap));return snap

def render(path):
    r,rh=registered(path);out=Path(r['out']);require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit GPU0 renderer required')
    env=pinned(r['environment_request']);source=read(out/'sources_complete.json');cpu_done=read(out/'cpu_complete.json')
    require(source['request_sha256']==cpu_done['request_sha256']==rh and cpu_done['ledger']['unresolved']==0,'Complete source/PHY stages required')
    started=time.monotonic();rows=[];new_renders=source_decodes=0
    with shared.lock(env['visual_config']['visual_lock']),shared.lock(out/'render_gpu.lock'):
        native,module=build_native(env,out);codec=SourceCodec(r['root'],static_completion=r['static_completion']['path'],native=native)
        for entry in source['records']:
            i=entry['source_index'];cp,arrays=load(entry);frames=[x for x in cpu_done['frames'] if x['source_index']==i]
            completed=out/'rendered_sources'/f'{i:04d}.json'
            if completed.exists():
                done=read(completed);require(done['request_sha256']==rh,'Completed display source request differs');rows+=done['rows'];new_renders+=done['new_VAR_renders'];source_decodes+=done['new_VAR_source_decodes'];continue
            images=[];cache={};origins={};source_rows=[];count=decodes=0;decode_records={};old=pinned(r['old_reuse'][str(i)])
            for frame in frames:
                shared.guard(r,started);native.health_polling.next_check=0.;native.common.check();packet=pinned(frame['physical_frame'])
                state,evidence,recovery=cal.recover_once(out/'received_source_decodes'/f'{i:04d}',rh,cp,arrays,packet,codec,r['source_model_id'])
                decode_records[recovery['path']]=evidence['new_VAR_source_decode']
                key=cal.digest(state)
                if key not in cache:
                    rec=out/'render_states'/f'{i:04d}'/(key+'.json');archive=rec.with_suffix('.npz');reservation=rec.with_suffix('.reserved.json')
                    if key in old:
                        original=old[key];require(original['state']==state,'Original exact received-state mismatch')
                        verify(original['archive']['path'],original['archive']['sha256'])
                        with np.load(original['archive']['path'],allow_pickle=False) as z:image=z['images'][original['image_slot']].copy()
                        require(image.dtype==np.float32 and image.shape==(3,256,256) and shared.image_sha(image)==original['image_sha256'],
                            'Sealed original received-state float32 image differs')
                        origins[key]='EXACT_ORIGINAL_RECEIVED_STATE_IMAGE'
                    elif rec.exists():
                        v=read(rec);require(v['request_sha256']==rh and v['state']==state,'Completed render identity differs');verify(archive,v['archive_sha256'])
                        with np.load(archive,allow_pickle=False) as z:image=z['image'].copy()
                        origins[key]='NEW_ACTUAL_RECEIVED_STATE'
                    else:
                        require(not reservation.exists(),'Unresolved renderer call, no automatic repeat')
                        write(reservation,dict(request_sha256=rh,state=state));image=shared.render_state(native,module,state)
                        with archive.open('xb') as f:np.savez(f,image=image)
                        write(rec,dict(request_sha256=rh,state=state,archive_sha256=sha(archive),image_sha256=shared.image_sha(image)))
                        origins[key]='NEW_ACTUAL_RECEIVED_STATE'
                    count+=int(state['kind']!='gray' and origins[key]!='EXACT_ORIGINAL_RECEIVED_STATE_IMAGE');cache[key]=len(images);images.append(image)
                slot=cache[key]
                source_rows.append(dict(source_index=i,source_id=entry['source_id'],snr_db=frame['snr_db'],noise_seed=6201,family=r['family'],
                    array_key='images',array_slot=slot,reference_key='source_rgb',image_sha256=shared.image_sha(images[slot]),
                    source_status=evidence['source_status'],gray=state['kind']=='gray',physical_frame=frame['physical_frame'],source_decode=recovery,
                    received_state_sha256=key,visual_reuse=origins[key]))
            ap=out/'images'/f'{i:04d}.npz';ap.parent.mkdir(exist_ok=True)
            with ap.open('xb') as f:np.savez_compressed(f,images=np.stack(images),source_rgb=arrays['pixels'].astype(np.float32)/np.float32(255))
            for row in source_rows:row['archive']=pin(ap)
            decodes=sum(decode_records.values())
            write(completed,dict(request_sha256=rh,rows=source_rows,new_VAR_renders=count,new_VAR_source_decodes=decodes))
            rows+=source_rows;new_renders+=count;source_decodes+=decodes
        native.frozen()
    require(len(rows)==48 and new_renders<=48 and source_decodes<=48,'Fixed48 output and GPU bounds required')
    done=dict(status='T5_FIXED16_ACTUAL_RENDER_COMPLETE',request_sha256=rh,rows=rows,new_VAR_renders=new_renders,new_VAR_source_decodes=source_decodes,new_metric_calls=0)
    write(out/'render_complete.json',done);return dict(status=done['status'],new_VAR_renders=new_renders,new_VAR_source_decodes=source_decodes)

def close(a):
    r,rh=registered(a.request);out=Path(r['out']);obs=read(a.execution_observation)
    require(obs['request_sha256']==rh and obs['actual_children_waited'] and len(obs['worker_exit_codes'])>=2
        and all(x==0 for x in obs['worker_exit_codes']),'Actual parent wait/exit observation required, never inferred from files')
    source=read(out/'sources_complete.json');cpu_done=read(out/'cpu_complete.json');gpu=read(out/'render_complete.json')
    require(all(x['request_sha256']==rh for x in (source,cpu_done,gpu)) and len(gpu['rows'])==48,'Actual completed stages required')
    snap=cal.ledger_snapshot(out/'packet_ledger.sqlite',rh,96);require(snap['unresolved']==0,'Unresolved actual PHY call')
    outputs={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.npz') and p.name not in ('completion.json','display_manifest.json')}
    done=dict(status='T5_FIXED16_SCIENTIFIC_DISPLAY_COMPLETE',request_sha256=rh,source_count=16,frame_count=48,
        actual_children_waited=True,worker_exit_codes=obs['worker_exit_codes'],execution_observation=pin(a.execution_observation),
        family=r['family'],family_freeze=r['family_freeze'],population='development_fixed16',holdout_used=False,
        new_metric_calls=0,new_bootstrap_calls=0,ledger=snap,new_VAR_renders=gpu['new_VAR_renders'],new_VAR_source_decodes=gpu['new_VAR_source_decodes'],outputs=outputs)
    write(out/'completion.json',done)
    manifest=dict(status='T5_FROZEN_ENTROPY_FIXED16_DISPLAY_COMPLETE',N=1024,source_indices=INDICES,snrs=SNRS,noise_seed=6201,
        family=r['family'],population='development_fixed16',holdout_used=False,rows=gpu['rows'],artifact_root=str(out),
        completion=pin(out/'completion.json'),freeze=r['family_freeze'],image_hash_domain='T1_shared_image_sha')
    write(out/'display_manifest.json',manifest);return dict(status=manifest['status'],manifest=str(out/'display_manifest.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare')
    for n in ('root','family-freeze','h-completion','environment-request','source-gate','out'):q.add_argument('--'+n,required=True)
    q.add_argument('--deadline-unix',type=float,required=True)
    for n in ('sources','cpu','render'):
        q=s.add_parser(n);q.add_argument('--request',required=True)
    q=s.add_parser('close');q.add_argument('--request',required=True);q.add_argument('--execution-observation',required=True)
    a=p.parse_args();value=prepare(a) if a.command=='prepare' else close(a) if a.command=='close' else {'sources':source_stage,'cpu':cpu,'render':render}[a.command](a.request)
    print(json.dumps(value,sort_keys=True))
