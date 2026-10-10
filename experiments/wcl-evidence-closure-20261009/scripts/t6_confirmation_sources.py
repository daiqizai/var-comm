"""Owned first access to pre-registered confirmation100, after N2048 policy freeze.

Exactly original256 preprocessing and one Encoder/VQ call per image. Content
duplicates are checked before any Encoder call. Only the frozen entropy family
is prepared; no channel, reconstruction, scoring, bootstrap or source reselection.
"""
import argparse,hashlib,importlib.util,json,os,signal,sys,time,traceback
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,read,require,sha,verify,choose_arithmetic
from t1_holdout_metadata import pin,pinned
from t1_source_asset_schema import token_sha
from t1_holdout_render import build_native
from t6_source_codec import SourceCodec10,build_record
from t6_register_confirmation_metadata import canonical_id
import t2_pilot as shared
save=shared.save
ORIGINAL_WORKER_SHA='c8469584a08ca0f1053ff506a74bae54f5427b04fa90025eaa36656d200db458'
PREPROCESS_SHA='568217086c94ee7607abc256415d705a11702ffcaaed9c79ca04d5e9a62da562'
REGISTRATION_SHA='fd846f0d3ab1c093ec009739fc6ced4b53b4650a9d496aa4f76908e3ea8bb0ac'

def module(desc,name):
    verify(desc['path'],desc['sha256']);s=importlib.util.spec_from_file_location(name,desc['path']);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def normalize_candidate(c):
    result=dict(c);result['target_m']=int(c.get('target_m',c.get('m',0)))
    result['source_capacity']=int(c.get('source_capacity',c.get('source_capacity_bits',min(int(c.get('k',0))-29,8191))))
    require(4<=result['target_m']<=10 and 2<=result['source_capacity']<=8191,'Frozen complete-prefix/uint13 capacity required')
    return result

def prepare(a):
    verify(a.confirmation_registration,REGISTRATION_SHA);reg=read(a.confirmation_registration);freeze=read(a.calibration_freeze);selected=read(a.family_freeze)
    require(reg['schema']=='WCL_N2048_CONFIRMATION100_METADATA_V1' and reg['source_count']==100 and reg['noise_seeds']==[9201,9202,9203],
        'Exactly pre-registered confirmation100 required')
    require(freeze['status']=='T6_N2048_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and freeze['N']==2048 and freeze['source_count']==1000
        and not freeze['selection_used_confirmation'] and not freeze['holdout_used_for_selection']
        and freeze['entropy_family_selection']==pin(a.family_freeze),'Actual N2048 fullcal policies must precede source content')
    require(selected['family']=='EC_VAR_WHOLE' and selected['holdout_files_read'] is False and not selected['holdout_used_for_selection'],
        'Original calibration-only single VAR family required')
    reference=read(a.reference_hash_inventory)
    require(reference['status']=='T6_AUDITED_OLD_CONTENT_HASH_INVENTORY_COMPLETE' and reference['confirmation_registration']==pin(a.confirmation_registration)
        and reference['new_confirmation_files_opened']==0,'All five audited-role hash references required before new content')
    template=read(a.original_source_request);verify(template['worker'],ORIGINAL_WORKER_SHA);verify(template['preprocess_module'],PREPROCESS_SHA)
    require(template['source_bindings'][template['worker']]==ORIGINAL_WORKER_SHA and template['source_bindings'][template['preprocess_module']]==PREPROCESS_SHA,
        'Original Encoder/preprocess source bindings differ')
    original_split=read(a.class_mapping_manifest);classes={}
    for row in original_split['entries']:
        synset=row['synset'];label=int(row['class_index']);require(synset not in classes or classes[synset]==label,'Class map inconsistency');classes[synset]=label
    require(len(classes)==1000 and set(classes.values())==set(range(1000)),'Original class-index map must be complete')
    root=Path(a.root).resolve();out=Path(a.out).resolve();env=read(a.environment_request)
    require(env['root']==str(root) and out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009') and not out.exists(),'Fresh root-owned source output required')
    excluded=pinned(reg['excluded_manifest']);oldids=set(excluded['canonical_source_ids']);records=[]
    for i,row in enumerate(reg['records']):
        require(row['confirmation_index']==i and row['canonical_source_id']==canonical_id(row['source_id']) and row['canonical_source_id'] not in oldids,'Original fixed source order/exclusion differs')
        records.append(dict(row,source_index=i,evaluation_class_index=classes[row['source_id'].split('/')[0]]))
    out.mkdir(parents=True);T1=pinned(selected['original_policy_freeze'])
    r=dict(schema='T6_CONFIRMATION100_SOURCE_REQUEST_V1',root=str(root),out=str(out),records=records,source_count=100,
        confirmation_registration=pin(a.confirmation_registration),calibration_freeze=pin(a.calibration_freeze),entropy_family_selection=pin(a.family_freeze),
        reference_hash_inventory=pin(a.reference_hash_inventory),class_mapping_manifest=pin(a.class_mapping_manifest),original_source_request=pin(a.original_source_request),
        original_encoder_module=dict(path=template['worker'],sha256=ORIGINAL_WORKER_SHA),preprocess_module=dict(path=template['preprocess_module'],sha256=PREPROCESS_SHA),
        environment_request=pin(a.environment_request),static_completion=T1['protocol']['static_completion'],family='EC_VAR_WHOLE',
        entropy_policies={str(s):normalize_candidate(freeze['policies']['ENTROPY_WHOLE'][str(s)]) for s in [4,10,19]},
        N=2048,SNRs=[4,10,19],noise_seeds=[9201,9202,9203],source_ids=[x['source_id'] for x in records],
        max_Encoder_VQ_calls=100,max_VAR_TX_calls=100,max_VAR_RX_calls=300,new_pixel_access=False,new_model_calls=0,
        deadline_unix=a.deadline_unix,max_seconds=86400,stop_files=[str(root/'STOP'),str(out/'STOP')],
        source_reselection_allowed=False,training_updates=0,new_packet_decodes=0,new_metric_calls=0)
    names=['t6_confirmation_sources.py','t6_source_codec.py','t6_register_confirmation_metadata.py','t1_codec_runtime.py','t1_source_asset_schema.py',
        't1_entropy_core.py','t1_holdout_metadata.py','t1_holdout_render.py','t1_calibrate.py','t2_pilot.py']
    r['source_bindings']={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in names}
    save(out/'request.json',r);return dict(status='T6_CONFIRMATION100_SOURCE_METADATA_PREPARED',pixel_reads=0,new_model_calls=0)

def run(path):
    r=read(path);rh=sha(path);out=Path(r['out']);require(r['schema']=='T6_CONFIRMATION100_SOURCE_REQUEST_V1','Frozen new100 source request required')
    for p,h in r['source_bindings'].items():verify(p,h)
    reg=pinned(r['confirmation_registration']);freeze=pinned(r['calibration_freeze']);selected=pinned(r['entropy_family_selection']);reference=pinned(r['reference_hash_inventory'])
    require(not freeze['selection_used_confirmation'] and selected['holdout_files_read'] is False,'No confirmation-conditioned policy selection')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Original GPU0 numerical environment required')
    require(not(out/'owner_started.json').exists(),'New source owner already attempted, no automatic repeats')
    env=pinned(r['environment_request']);template=pinned(r['original_source_request']);pinned(r['static_completion'])
    require(os.path.abspath(sys.executable)==template['python'] and sorted(os.sched_getaffinity(0))==[4,5,6,7,8,9]
        and os.getpriority(os.PRIO_PROCESS,0)==15,'Original admitted source interpreter/CPU resources required')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,shared.stop)
    native=None;started=time.monotonic();counts={'Encoder_VQ':0,'VAR_TX':0,'VAR_RX':0};outputs={};records=[];inputs={};duplicate=None
    def guard():
        shared.guard(r,started)
        if native is not None:native.health_polling.next_check=0.;native.common.check()
    def call(kind,i,fn,m=None):
        guard();require(counts[kind]<r['max_'+kind+'_calls'],'Frozen source call cap exhausted')
        p=out/'calls'/(f'{i:04d}_{kind}'+('' if m is None else '_m'+str(m))+'.reserved.json');require(not p.exists(),'Unresolved new source call, no automatic repeat')
        save(p,dict(request_sha256=rh,source_index=i,kind=kind,m=m));v=fn();counts[kind]+=1
        save(p.with_name(p.name.replace('.reserved.json','.complete.json')),dict(request_sha256=rh,source_index=i,kind=kind,m=m));return v
    try:
        with shared.lock(env['visual_config']['visual_lock']),shared.lock(out/'gpu.lock'):
            save(out/'owner_started.json',dict(pid=os.getpid(),request_sha256=rh,started_unix=time.time(),content_opened_after_N2048_freeze=True))
            shared.add_paths(env);preprocess=module(r['preprocess_module'],'_t6_exact_original_preprocess');encoder=module(r['original_encoder_module'],'_t6_exact_original_encoder')
            raw_hashes={h for x in reference['records'] for h in x['raw_JPEG_sha256']};pixel_hashes={h for x in reference['records'] for h in x['preprocessing_sha256']}
            seen_raw=set();seen_pixels=set();pixels_by_source=[];duplicates=[];actual_content=[];image_root=Path(reg['image_root']).resolve()
            for row in r['records']:
                guard();i=row['source_index'];image=Path(row['path']);require(image.resolve().is_relative_to(image_root) and not image.is_symlink()
                    and image.stat().st_size==row['original_bytes'],'Original registered regular source bytes/location differ')
                raw=sha(image);tensor,pre=preprocess.preprocess(image)
                pixels=np.ascontiguousarray(np.rint((tensor.cpu().numpy()+1)*127.5),dtype=np.uint8)
                require(pixels.shape==(3,256,256) and hashlib.sha256(pixels.tobytes()).hexdigest()==pre and sha(image)==raw,'Original256 preprocessing or JPEG changed')
                flip=hashlib.sha256(np.ascontiguousarray(pixels[:,:,::-1]).tobytes()).hexdigest()
                reasons=[]
                if raw in raw_hashes:reasons.append('old_original_JPEG')
                if pre in pixel_hashes or flip in pixel_hashes:reasons.append('old_preprocessed_or_horizontal_flip_pixels')
                if raw in seen_raw:reasons.append('within_confirmation_original_JPEG')
                if pre in seen_pixels or flip in seen_pixels:reasons.append('within_confirmation_preprocessed_or_horizontal_flip_pixels')
                if reasons:duplicates.append(dict(source_index=i,source_id=row['source_id'],reasons=reasons))
                seen_raw.add(raw);seen_pixels.update((pre,flip));pixels_by_source.append((pixels,pre,raw));inputs[str(image)]=raw
                actual_content.append(dict(source_index=i,source_id=row['source_id'],original_JPEG_sha256=raw,preprocessing_sha256=pre,horizontal_flip_preprocessing_sha256=flip))
            duplicate=dict(status='T6_CONFIRMATION100_CONTENT_DUPLICATES_CHECKED_PASS' if not duplicates else 'T6_CONFIRMATION100_CONTENT_DUPLICATES_DETECTED_STOP',
                source_count=100,duplicate_source_id_count=0,duplicate_preprocessing_count=sum(any('pixels' in k for k in x['reasons']) for x in duplicates),
                duplicate_original_JPEG_count=sum(any('JPEG' in k for k in x['reasons']) for x in duplicates),duplicates=duplicates,
                confirmation_registration=r['confirmation_registration'],reference_hash_inventory=r['reference_hash_inventory'],scope='all_available_audited_source_content_hashes',
                available_reference_hash_count=reference['available_reference_hash_count'],unavailable_reference_hash_count=reference['unavailable_reference_hash_count'],
                reference_sources_without_raw_JPEG_hash=reference['source_without_raw_JPEG_hash'],reference_sources_without_preprocessing_hash=reference['source_without_preprocessing_hash'],
                historical_exclusions_complete_beyond_supplied_manifests=False,actual_content=actual_content,Encoder_calls_before_check=0,source_reselection=False)
            dp=out/'content_duplicate_check.json';save(dp,duplicate);outputs[str(dp)]=sha(dp)
            require(not duplicates,'Content duplicate found; no source replacement or Encoder is permitted')
            native,_=build_native(env,out);codec=SourceCodec10(r['root'],static_completion=r['static_completion']['path'],native=native)
            forbidden={f.__code__ for f in (native.data,native.score,native.qualify,native.phy.receive,native.common.quality,native.receiver.complete_partial)}
            require(sys.getprofile() is None,'Unexpected existing scientific profile guard')
            def profile(frame,event,arg):
                if event=='call' and frame.f_code in forbidden:raise RuntimeError('Forbidden score/PHY/reconstruction/population call in source-only owner')
            sys.setprofile(profile)
            try:
                with native.torch.no_grad():
                    for row,(pixels,pre,raw) in zip(r['records'],pixels_by_source):
                        guard();i=row['source_index'];tokens,F=call('Encoder_VQ',i,lambda:encoder.encode_flat(native,pixels))
                        ap=out/'assets'/f'{i:04d}.npz';ap.parent.mkdir(exist_ok=True)
                        with ap.open('xb') as f:np.savez(f,tokens=tokens,pixels=pixels)
                        with np.load(ap,allow_pickle=False) as z:require(np.array_equal(z['tokens'],tokens) and np.array_equal(z['pixels'],pixels),'Exact original Encoder/pixel readback failed')
                        fp=out/'F_latents'/f'{i:04d}.npz';fp.parent.mkdir(exist_ok=True)
                        with fp.open('xb') as f:np.savez(f,F=F)
                        cp=out/'source_checkpoints'/f'{i:04d}.json'
                        record=dict(source_index=i,source_id=row['source_id'],evaluation_class_index=row['evaluation_class_index'],preprocessing_id=pre,tokens_sha256=token_sha(tokens),
                            archive=str(ap),archive_sha256=sha(ap),checkpoint=str(cp),original_JPEG_sha256=raw,F_archive=pin(fp))
                        save(cp,dict(status='T6_CONFIRMATION100_FROZEN_SOURCE_ASSET_READY',**record,population_role='confirmation',
                            confirmation_registration=r['confirmation_registration'],calibration_freeze=r['calibration_freeze'],image_path=row['path'],image_file_sha256=raw,
                            encoder_tokens_verified=True,asset_readback_exact=True,Encoder_VQ_calls=1,preprocess='original_bound_preprocess',outputs={str(ap):sha(ap),str(fp):sha(fp)},
                            new_packet_decodes=0,new_metric_calls=0,policy_selection=False))
                        record['checkpoint_sha256']=sha(cp)
                        maximum=max(c['target_m'] for c in r['entropy_policies'].values())
                        encoded=call('VAR_TX',i,lambda:codec.encode(r['family'],tokens,tuple(range(4,maximum+1))),m=maximum)
                        selected={str(s):choose_arithmetic({m:x['arithmetic_bits'] for m,x in encoded.items()},r['entropy_policies'][str(s)],minimum_m=4) for s in [4,10,19]}
                        require(all(v['status']=='SOURCE_LENGTH_FITS_LAYOUT_PENDING' for v in selected.values()),'Frozen entropy fallback incomplete')
                        arrays={'tokens':tokens};tx={str(m):dict(bits_key=f'tx_var_m{m}_bits',raw_bits=x['raw_bits'],arithmetic_bits=x['arithmetic_bits'],flush_bits=x['flush_bits']) for m,x in encoded.items()}
                        arrays.update({f'tx_var_m{m}_bits':x['bits'] for m,x in encoded.items()})
                        for m in sorted({v['actual_m'] for v in selected.values()}):
                            rx=call('VAR_RX',i,lambda:codec.decode(r['family'],encoded[m]['bits'],m),m=m)
                            require(np.array_equal(rx['received_tokens'],tokens[:OFFSETS[m]]),'Actual independent entropy roundtrip differs')
                            arrays[f'var_m{m}_bits']=encoded[m]['bits'];arrays[f'var_m{m}_received_tokens']=rx['received_tokens']
                        ea=out/'entropy_assets'/f'{i:04d}.npz';ea.parent.mkdir(exist_ok=True)
                        with ea.open('xb') as f:np.savez(f,**arrays)
                        er=build_record(source_index=i,source_id=row['source_id'],tokens=tokens,preprocessing_id=pre,source_assets_checkpoint=pin(cp),archive=ea,arrays=arrays,
                            static_completion_sha=r['static_completion']['sha256'],origins={f:'T6_NEW_CONFIRMATION_SELECTED_PREFIX_ACTUAL_ROUNDTRIP' for f in ('EC_STATIC_WHOLE','EC_VAR_WHOLE')},
                            upstream_evidence={f:dict(request=pin(path),calibration_freeze=r['calibration_freeze']) for f in ('EC_STATIC_WHOLE','EC_VAR_WHOLE')})
                        er.update(source_role='confirmation',N=2048,entropy_family_selection=r['entropy_family_selection'],tx_prefixes={r['family']:tx},
                            selected_by_snr={r['family']:{str(s):dict(candidate_id=r['entropy_policies'][str(s)]['candidate_id'],target_m=r['entropy_policies'][str(s)]['target_m'],
                                q=r['entropy_policies'][str(s)]['q'],nominal_rate=r['entropy_policies'][str(s)]['nominal_rate'],**selected[str(s)]) for s in [4,10,19]}})
                        ep=out/'entropy_checkpoints'/f'{i:04d}.json';save(ep,er);record['entropy_checkpoint']=pin(ep);records.append(record)
                        for p in (ap,fp,cp,ea,ep):outputs[str(p)]=sha(p)
                        print(f'T6 frozen confirmation source {i+1}/100; pixels, Encoder and selected entropy prefixes actually complete',flush=True)
            finally:sys.setprofile(None)
            native.frozen();guard()
        manifest=dict(schema='T6_CONFIRMATION100_SOURCE_MANIFEST_V1',source_count=100,source_ids=r['source_ids'],records=records,
            confirmation_registration=r['confirmation_registration'],calibration_freeze=r['calibration_freeze'],entropy_family_selection=r['entropy_family_selection'],
            content_duplicate_check_completion=pin(out/'content_duplicate_check.json'),population='confirmation',N=2048,source_reselection=False)
        mp=out/'manifest.json';save(mp,manifest);outputs[str(mp)]=sha(mp)
        for p in (out/'calls').glob('*.json'):outputs[str(p)]=sha(p)
        require(counts['Encoder_VQ']==counts['VAR_TX']==100 and len(records)==100,'Exact registered100 source calls required')
        done=dict(status='T6_CONFIRMATION100_SOURCE_ASSETS_COMPLETE',request_sha256=rh,source_count=100,source_manifest=pin(mp),source_ids=r['source_ids'],counts=counts,
            outputs=outputs,actual_original_JPEG_inputs=inputs,content_duplicate_check_completion=manifest['content_duplicate_check_completion'],
            new_packet_decodes=0,new_metric_calls=0,new_image_renders=0,new_bootstrap_calls=0,training_updates=0,source_reselection=False,seconds=time.monotonic()-started)
        save(out/'completion.json',done);return dict(status=done['status'],source_count=100,counts=counts)
    except BaseException:
        save(out/'owner_failed.json',dict(request_sha256=rh,counts=counts,traceback=traceback.format_exc(),no_auto_retry=True));raise
    finally:save(out/'owner_exited.json',dict(pid=os.getpid(),counts=counts,completed=(out/'completion.json').exists()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);q=s.add_parser('prepare')
    for n in ('root','confirmation-registration','calibration-freeze','family-freeze','reference-hash-inventory','class-mapping-manifest','original-source-request','environment-request','out'):q.add_argument('--'+n,required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q=s.add_parser('run');q.add_argument('--request',required=True);a=p.parse_args()
    print(json.dumps(prepare(a) if a.command=='prepare' else run(a.request)))
