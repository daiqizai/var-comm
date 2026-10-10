"""Frozen N2048 confirmation100: actual PHY and reconstruction, zero metrics.

Nine pre-frozen method/SNR policies, three noises and all100 fixed sources are
retained. CPU raw and entropy runtimes live in separate processes because their
original module bindings differ. They share paired noise and one durable cap.
"""
import argparse,contextlib,json,math,os,signal,time
from pathlib import Path
import numpy as np
import t2_pilot as h
import t1_calibrate as cal
import t1_holdout_render as oldrender
import t6_raw_calibrate as rawcal
import t6_raw_phy as rawphy
import t6_entropy_phy as ecphy
from t6_source_codec import SourceCodec10
from t1_source_asset_schema import token_sha,model_id
from t6_plan import write_csv

SCHEMA='WCL_T6_CONFIRMATION100_RENDER_EXECUTION_V1'
COUNT=100;SNRS=[4,10,19];SEEDS=[9201,9202,9203]
RAW_WHOLE='N2048_RAW_WHOLE_VAR_COMPLETION';RAW_PARTIAL='N2048_RAW_PARTIAL_VAR_COMPLETION'


def methods(family):return[RAW_WHOLE,RAW_PARTIAL,'N2048_'+family+'_VAR_COMPLETION']
def counter(i,s,n):return(SNRS.index(s)*100+i)*3+SEEDS.index(n)


def load_source(record,r):
    cp=h.checked(dict(path=record['checkpoint'],sha256=record['checkpoint_sha256']))
    h.require(cp['source_id']==record['source_id']and cp['source_index']==record['source_index']and cp['preprocessing_id']==record['preprocessing_id']
        and cp['calibration_freeze']==r['calibration_freeze']and cp['outputs'][record['archive']]==record['archive_sha256'],'Exact post-freeze new source asset')
    tokens,pixels=h.source_assets(record);sr=h.checked(record['entropy_checkpoint'])
    h.require(sr['schema']=='T1_SOURCE_ASSET_V1'and sr['source_role']=='confirmation'and sr['source_index']==record['source_index']
        and sr['source_id']==record['source_id']and sr['source_tokens_sha256']==record['tokens_sha256']
        and sr['entropy_family_selection']==r['entropy_family_selection']and sr['source_assets_checkpoint']==dict(path=record['checkpoint'],sha256=record['checkpoint_sha256']),
        'Independent entropy proof must bind the same new source')
    h.require(h.sha(sr['archive']['path'])==sr['archive']['sha256'],'Source entropy archive changed')
    with np.load(sr['archive']['path'],allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
    h.require(token_sha(arrays[sr['tokens_key']])==record['tokens_sha256'],'Encoder/source streams disagree')
    return tokens,pixels,sr,arrays


def prepare(a):
    done=h.read(a.source_completion);manifest=h.checked(done['source_manifest']);freeze=h.read(a.calibration_freeze)
    h.require(done['status']=='T6_CONFIRMATION100_SOURCE_ASSETS_COMPLETE'and manifest['schema']=='T6_CONFIRMATION100_SOURCE_MANIFEST_V1'
        and done['source_count']==manifest['source_count']==100 and len(manifest['records'])==100,'Completed fixed new100 source assets required')
    h.require(freeze['status']=='T6_N2048_POLICIES_FROZEN_CALIBRATION_ONLY_V1'and freeze['N']==2048 and freeze['source_count']==1000 and freeze['noise_count']==3
        and not freeze['selection_used_confirmation']and not freeze['holdout_used_for_selection']and manifest['calibration_freeze']==h.desc(a.calibration_freeze),'Nine policies frozen before confirmation content')
    family=h.checked(freeze['entropy_family_selection']);h.require(not family['holdout_files_read']and not family['holdout_used_for_selection'],'Calibration-only entropy family')
    registration=h.checked(manifest['confirmation_registration']);duplicate=h.checked(manifest['content_duplicate_check_completion'])
    ids=[x['source_id']for x in manifest['records']]
    h.require(ids==manifest['source_ids']==done['source_ids']==[x['source_id']for x in registration['records']]
        and registration['noise_seeds']==SEEDS and len(set(ids))==100,'Pre-pixel frozen order and three noise seeds required')
    h.require(duplicate['status']=='T6_CONFIRMATION100_CONTENT_DUPLICATES_CHECKED_PASS'and duplicate['source_count']==100
        and duplicate['duplicate_source_id_count']==duplicate['duplicate_preprocessing_count']==duplicate['duplicate_original_JPEG_count']==0
        and duplicate['confirmation_registration']==manifest['confirmation_registration'],'Actual pre-Encoder content duplicate gate required')
    rawq=h.checked(freeze['raw_phy_qualification']);ecq=h.checked(freeze['entropy_phy_qualification'])
    h.require(rawq['status']==ecq['status']=='PASS'and rawq['schema']==rawphy.QUAL_SCHEMA and ecq['schema']==ecphy.QUAL_SCHEMA,'Both actual N2048 PHY qualifications required')
    rawqr=h.checked(rawq['request']);rawcat=h.checked(rawqr['candidate_catalogue']);ecqr=h.checked(ecq['request'])
    h.require(ecqr['family']==family['family']and ecqr['entropy_family_selection']==freeze['entropy_family_selection'],'Qualified frozen entropy family differs')
    env=h.read(a.environment_request);root=Path(env['root']);out=Path(a.out).resolve();h.require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'Independent confirmation namespace')
    selected=freeze['policies'];schedule=[]
    h.require(set(selected)=={'RAW_WHOLE','RAW_PARTIAL','ENTROPY_WHOLE'}and all(set(v)=={'4','10','19'}for v in selected.values()),'Exactly nine frozen policies')
    for s in SNRS:
        for f,m in [('RAW_WHOLE',RAW_WHOLE),('RAW_PARTIAL',RAW_PARTIAL)]:
            p=selected[f][str(s)];h.require(p==rawcat['profiles'][p['profile_id']]and(f!='RAW_WHOLE'or p['K']==0),'Frozen actual raw profile/permissions differ')
            schedule.append(dict(method_id=m,snr_db=s,branch='raw',profile=p))
        c=selected['ENTROPY_WHOLE'][str(s)];expected=next(x for x in ecqr['admitted_candidates']if x['candidate_id']==c['candidate_id'])
        h.require(all(c[k]==v for k,v in expected.items()),'Frozen entropy candidate differs from actual constructor admission')
        schedule.append(dict(method_id=methods(family['family'])[2],snr_db=s,branch='entropy',candidate=c))
    static=h.checked(family['original_policy_freeze'])['protocol']['static_completion'];h.checked(static)
    r=dict(schema=SCHEMA,N=2048,root=str(root),out=str(out),source_count=100,records=manifest['records'],source_ids=ids,snrs=SNRS,noise_seeds=SEEDS,
        methods=methods(family['family']),family=family['family'],schedule=schedule,frame_count=2700,packet_cap=5400,workers=a.workers,gpu_workers=a.gpu_workers,
        source_completion=h.desc(a.source_completion),source_manifest=done['source_manifest'],calibration_freeze=h.desc(a.calibration_freeze),
        entropy_family_selection=freeze['entropy_family_selection'],confirmation_registration=manifest['confirmation_registration'],
        content_duplicate_check_completion=manifest['content_duplicate_check_completion'],raw_phy_qualification=freeze['raw_phy_qualification'],entropy_phy_qualification=freeze['entropy_phy_qualification'],
        static_completion=static,source_model_id={family['family']:model_id(family['family'],static['sha256'])},environment_request=h.desc(a.environment_request),
        source_bindings=dict(env['source_bindings']),native_source_bindings=dict(env['native_source_bindings']),old_visual_identity=env['old_visual_identity'],old_numerical_runtime=env['old_numerical_runtime'],
        old_score_identity=env['old_score_identity'],visual_config=env['visual_config'],input_bindings=env['input_bindings'],
        qualification_records=env['records'][:a.gpu_workers],gpu_cohort_runtime=env['gpu_cohort_runtime'],python_gpu=env['python_gpu'],python_cpu=env['python_cpu'],
        noise_rule='t6_raw_phy.standard_noise(source_id,seed), exact paired full2048 for all methods; waveform-specific paid protocol',
        counter_rule='(index_in_[4,10,19]*100+confirmation_index)*3+index_in_[9201,9202,9203]',
        source_only_image_reuse=False,old_calibration_quality_reuse=False,source_reselection=False,policy_reselection=False,new_metric_calls=0,new_bootstrap_calls=0,training_updates=0,
        deadline_unix=a.deadline_unix,max_seconds=86400,stop_files=[str(root/'STOP'),str(out/'STOP')],automatic_successor=False)
    v=r['visual_config'];closure=h.load(v['static_closure_module'],'_t6_confirmation_closure',r['source_bindings'])
    bindings=closure.collect_bindings(root,v['native_runtime'],v['var_source'],v['dino_source'],v['uep_runtime'])
    for p,value in r['native_source_bindings'].items():h.require(bindings.get(p)==value,'Frozen native source changed')
    r['native_source_bindings']=bindings;r['source_bindings'].update(bindings)
    for name in bound_names():p=str(Path(__file__).with_name(name).resolve());r['source_bindings'][p]=h.sha(p)
    for rec in r['records']:
        _,_,sr,arrays=load_source(rec,r)
        for p in[x for x in schedule if x['branch']=='entropy']:oldrender.selected_stream(sr,arrays,r['family'],p['snr_db'],p['candidate'])
    validate(r);h.save(out/'request.json',r)
    return dict(status='T6_CONFIRMATION_RECEIVE_RENDER_REGISTERED_NOT_RUN',request=h.desc(out/'request.json'),frame_count=2700,packet_cap=5400,new_scientific_calls=0)


def bound_names():return('t6_confirmation_render.py','t6_confirmation_gpu_parallel.py','t6_raw_phy.py','t6_entropy_phy.py','t6_source_codec.py',
    't6_raw_calibrate.py','t1_calibrate.py','t1_holdout_render.py','t1_source_asset_schema.py','t2_pilot.py','t2_ledger.py')


def validate(r):
    h.require(r['schema']==SCHEMA and r['N']==2048 and r['source_count']==100 and r['noise_seeds']==SEEDS and r['snrs']==SNRS
        and r['frame_count']==2700 and r['packet_cap']==5400,'Exactly fixed2700 new confirmation conditions')
    h.require(r['methods']==methods(r['family'])and len(r['schedule'])==9 and {(x['method_id'],x['snr_db'])for x in r['schedule']}=={(m,s)for m in r['methods']for s in SNRS},'Fixed nine method/SNR policies')
    h.require([(x['source_index'],x['source_id'])for x in r['records']]==list(enumerate(r['source_ids']))and len(set(r['source_ids']))==100,'Original fixed100 identity order')
    h.require(1<=r['workers']<=16 and r['gpu_workers']in(1,2)and not r['source_reselection']and not r['policy_reselection'],'Finite workers; no reselection')


def registered(path):
    r=h.read(path);validate(r)
    for name in bound_names():
        p=str(Path(__file__).with_name(name).resolve());h.require(r['source_bindings'][p]==h.sha(p),'Registered confirmation source changed')
    return r,h.sha(path)


def ledger_init(path):
    r,rh=registered(path);out=Path(r['out'])
    with h.lock(out/'ledger_initialize.lock'):
        meter=rawcal.Ledger(out/'packet_ledger.sqlite',rh,5400);snap=meter.snapshot();meter.close();h.require(snap['unresolved']==0,'Unresolved actual event requires audit')
        h.save(out/'ledger_initialized.json',dict(request_sha256=rh,packet_cap=5400))
    return snap


def cpu_worker(path,branch,index):
    r,rh=registered(path);out=Path(r['out']);began=time.monotonic()
    h.require(branch in('raw','entropy')and 0<=index<r['workers']and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Valid separate CPU branch')
    for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
    with h.lock(out/f'cpu_{branch}_{index}.lock'):
        rt=rawphy.create_runtime(r['raw_phy_qualification']['path'])if branch=='raw'else ecphy.create_runtime(r['root'],r['entropy_phy_qualification']['path'])
        profiles=None if branch=='raw'else{(p['m'],p['q'],p['nominal_rate']):p for p in rt.catalogue['profiles']}
        meter=rawcal.open_ledger(r,rh);completed=[]
        try:
            for rec in r['records'][index::r['workers']]:
                h.guard(r,began);i=rec['source_index'];cp=out/f'cpu_{branch}_sources'/f'{i:04d}.json'
                if cp.exists():h.require(h.read(cp)['request_sha256']==rh,'Saved CPU source differs');completed.append(h.desc(cp));continue
                tokens,_,sr,arrays=load_source(rec,r);scales=[tokens[rawcal.OFFSETS[j]:rawcal.OFFSETS[j+1]]for j in range(10)];frames=[]
                for point in[x for x in r['schedule']if x['branch']==branch]:
                    snr=point['snr_db'];method=point['method_id'];selection=None
                    if branch=='raw':p=point['profile'];pid=p['profile_id'];h.require(p==rt.catalogue.entry(pid),'Frozen raw policy differs')
                    else:
                        c=point['candidate'];m,bits,selection=oldrender.selected_stream(sr,arrays,r['family'],snr,c);p=profiles[(m,c['q'],c['nominal_rate'])];pid=p['profile_id']
                    for seed in SEEDS:
                        h.guard(r,began);ctr=counter(i,snr,seed);pp=out/'physical_frames'/branch/f'{i:04d}'/f'{snr}_{seed}_{pid:03d}.json'
                        if pp.exists():
                            packet=h.read(pp);h.require(packet['request_sha256']==rh and packet['source_id']==rec['source_id']and packet['public_frame_counter']==ctr,'Actual frame cache identity differs')
                        else:
                            wave,tx=rt.transmit(pid,scales if branch=='raw'else bits,ctr)
                            z=rawphy.standard_noise(rec['source_id'],seed);noise=z/math.sqrt(10**(snr/10));observed=wave+noise
                            event=f'{SCHEMA}/{rh[:16]}/{branch}/source{i}/snr{snr}/noise{seed}/profile{pid}'
                            rx=rt.receive(observed,snr,ctr,meter,event,'confirmation')if branch=='raw'else rt.receive(observed,snr,ctr,rt.profiles,meter,event,'confirmation')
                            packet=dict(request_sha256=rh,N=2048,branch=branch,source_id=rec['source_id'],source_index=i,snr_db=snr,noise_seed=seed,
                                TX_profile_id=pid,public_frame_counter=ctr,standard_noise_sha256=h.image_sha(z),observation_sha256=h.image_sha(observed),transmission=tx,actual_RX=rx)
                            h.save(pp,packet)
                        frames.append(dict(source_index=i,source_id=rec['source_id'],method_id=method,point_id=method+'_SNR_'+str(snr),snr_db=snr,noise_seed=seed,branch=branch,
                            candidate_id=point['profile']['candidate_id']if branch=='raw'else point['candidate']['candidate_id'],physical_frame=h.desc(pp),
                            target_m=p['m']if branch=='raw'else c['target_m'],actual_m=p['m'],K=p['K'],q=rawphy.MODULATIONS[p['groups'][0]['modulation']]if branch=='raw'else c['q'],
                            nominal_rate=None if branch=='raw'else c['nominal_rate'],fallback_attempts=[]if selection is None else selection['attempts']))
                h.save(cp,dict(status='T6_CONFIRMATION_CPU_BRANCH_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],branch=branch,frames=frames));completed.append(h.desc(cp))
                print(h.canonical(dict(stage='T6_CONFIRMATION_CPU',branch=branch,worker=index,source=i,logical_frames=len(frames))),flush=True)
            h.save(out/f'cpu_{branch}_workers'/f'{index:02d}.json',dict(status='COMPLETE',request_sha256=rh,sources=completed))
        finally:meter.close()


class NeedsIndependentDecode(Exception):pass
class NoModelCodec:
    def decode(self,*_):raise NeedsIndependentDecode()


def plan_render(path):
    r,rh=registered(path);out=Path(r['out']);outputs={};rendercap=decodecap=knowncount=pendingcount=0
    for rec in r['records']:
        i=rec['source_index'];frames=[];branches=[]
        for branch,count in [('raw',18),('entropy',9)]:
            p=out/f'cpu_{branch}_sources'/f'{i:04d}.json';cp=h.read(p)
            h.require(cp['request_sha256']==rh and cp['source_id']==rec['source_id']and len(cp['frames'])==count,'Both complete CPU branches required')
            frames.extend(cp['frames']);branches.append(h.desc(p))
        cp=out/'cpu_sources'/f'{i:04d}.json';h.save(cp,dict(status='T6_CONFIRMATION_CPU_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],frames=frames,branch_checkpoints=branches))
        _,_,sr,arrays=load_source(rec,r);states={};pending={};recovery={}
        for f in frames:
            packet=h.checked(f['physical_frame'])
            if f['branch']=='raw':state=packet['actual_RX']['receiver_state'];states[h.digest(state)]=state
            else:
                identity=cal.recovery_input(packet,r['source_model_id']);key=h.digest(identity)
                if key in recovery:continue
                recovery[key]=f['physical_frame']
                try:state,_=cal.recover(sr,arrays,packet,NoModelCodec(),r['source_model_id']);states[h.digest(state)]=state
                except NeedsIndependentDecode:pending[key]=identity
        rc=sum(s['kind']!='gray'for s in states.values())+len(pending);dc=sum(p['actual_received_profile']['family']=='EC_VAR_WHOLE'for p in pending.values())
        p=out/'render_plan_sources'/f'{i:04d}.json';h.save(p,dict(request_sha256=rh,source_id=rec['source_id'],cpu_source=h.desc(cp),known_states=states,
            pending_independent_decodes=pending,recovery_inputs=recovery,new_VAR_render_cap=rc,new_VAR_source_decode_cap=dc))
        outputs[str(p)]=h.sha(p);rendercap+=rc;decodecap+=dc;knowncount+=len(states);pendingcount+=len(pending)
    snap=cal.ledger_snapshot(out/'packet_ledger.sqlite',rh,5400)
    value=dict(status='T6_CONFIRMATION_ACTUAL_RX_RENDER_BUDGET_REGISTERED',request=h.desc(path),request_sha256=rh,source_count=100,
        known_unique_states=knowncount,pending_actual_source_decode_states=pendingcount,new_VAR_render_cap=rendercap+r['gpu_workers'],new_VAR_source_decode_cap=decodecap,
        qualification_VAR_calls=r['gpu_workers'],new_metric_calls=0,new_bootstrap_calls=0,packet_ledger=snap,outputs=outputs,
        deduplication='Same source and exact received prefix/partial state; no source-only or TX-token substitution')
    h.save(out/'render_plan.json',value);return{k:v for k,v in value.items()if k!='outputs'}


def token_diagnostics(state,tokens):
    if state['kind']=='gray':return dict(token_error_count=None,token_error_denominator=0,received_tokens_equal_source=None)
    actual=np.asarray([v for scale in state['prefix']for v in scale]+list(state['partial_values']),np.int64)
    h.require(len(actual)==rawcal.OFFSETS[state['m']]+state['K'],'Actual diagnostic prefix+partial length differs')
    count=int(np.count_nonzero(actual!=tokens[:len(actual)]))
    return dict(token_error_count=count,token_error_denominator=len(actual),received_tokens_equal_source=count==0)


def qualify_single(r,rh,out,native,source):
    p=out/'render_qualification.json'
    if p.exists():h.require(h.read(p)['request_sha256']==rh and h.read(p)['status']=='PASS','Prior qualification differs');return
    rec=r['qualification_records'][0];row=next(x for x in h.old_rows(rec)if not x['gray'])
    v=dict(image_path=row['image_archive'],image_key='images',image_slot=row['image_slot'],image_file_sha256=rec['old_image_archives'][row['image_archive']],image_sha256=row['image_sha256'])
    truth=rawcal.checked_image(v,{});reserved=p.with_suffix('.reserved.json');h.require(not reserved.exists(),'Unresolved qualification call')
    h.save(reserved,dict(request_sha256=rh,new_VAR_calls=1,new_metric_calls=0));image=h.render_state(native,source,row['receiver_state'])
    h.require(np.array_equal(image,truth)and h.image_sha(image)==row['image_sha256'],'Frozen original exact float32 renderer replay failed')
    h.save(p,dict(status='PASS',request_sha256=rh,image_sha256=row['image_sha256'],exact_float32_equal=True,new_VAR_calls=1,new_metric_calls=0))


def gpu_worker(path,cohort_config=None):
    r,rh=registered(path);out=Path(r['out']);began=time.monotonic();plan=h.read(out/'render_plan.json')
    h.require(plan['request_sha256']==rh and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Exact CPU-state plan and GPU0 required')
    parallel=None
    if cohort_config is None:
        h.require(r['gpu_workers']==1,'Two-worker request needs owner');shared=h.lock(r['visual_config']['visual_lock']);own=h.lock(out/'gpu.lock');indices=range(r['source_count'])
    else:
        import t6_confirmation_gpu_parallel as parallel
        cohort,probe=parallel.admit(r,rh,cohort_config);shared=contextlib.nullcontext();own=h.lock(out/f'gpu_worker_{cohort_config["worker_id"]}.lock');indices=cohort_config['source_indices']
    for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
    with shared,own:
        native,source=oldrender.build_native(r,out);codec=SourceCodec10(r['root'],r['static_completion']['path'],native)
        if parallel:
            h.require(native.assets.old.b is probe,'Owned GPU runtime differs');parallel.qualify(r,rh,native,source,cohort_config,cohort)
        else:qualify_single(r,rh,out,native,source)
        for i in indices:
            h.guard(r,began)
            if parallel:cohort.require_available()
            rec=r['records'][i];cp=out/'gpu_sources'/f'{i:04d}.json'
            if cp.exists():h.require(h.read(cp)['request_sha256']==rh,'Saved GPU source differs');continue
            pp=out/'render_plan_sources'/f'{i:04d}.json';h.require(plan['outputs'][str(pp)]==h.sha(pp),'Actual source render budget changed');budget=h.read(pp);cpu=h.checked(budget['cpu_source'])
            tokens,_,sr,arrays=load_source(rec,r);states={};decoded={};pins={};rows=[]
            for f in cpu['frames']:
                h.guard(r,began);packet=h.checked(f['physical_frame']);rx=packet['actual_RX'];tx=packet['transmission']
                if f['branch']=='raw':state=rx['receiver_state'];evidence=dict(source_status=rx['source_status'],source_decode_cache_hit=False,new_VAR_source_decode=0)
                else:
                    key=h.digest(cal.recovery_input(packet,r['source_model_id']))
                    if key not in decoded:
                        state,evidence,dp=cal.recover_once(out/'source_decodes'/f'{i:04d}',rh,sr,arrays,packet,codec,r['source_model_id']);pins[dp['path']]=dp;decoded[key]=(state,evidence)
                    state,evidence=decoded[key]
                key=h.digest(state)
                if key not in states:
                    jp=out/'render_states'/f'{i:04d}'/(key+'.json');ip=jp.with_suffix('.npz');res=jp.with_suffix('.reserved.json')
                    if jp.exists():
                        v=h.read(jp);h.require(v['request_sha256']==rh and v['state']==state,'Saved exact received-state render differs');rawcal.checked_image(v,{})
                    else:
                        h.require(not res.exists(),'Unresolved received-state render; never automatically repeat');h.save(res,dict(request_sha256=rh,state=state,new_VAR_calls=int(state['kind']!='gray')))
                        image=h.render_state(native,source,state)
                        h.require(image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()and image.min()>=0 and image.max()<=1,'Finite exact float32 RGB')
                        with ip.open('xb')as handle:np.savez(handle,image=image)
                        v=dict(request_sha256=rh,state=state,image_path=str(ip),image_key='image',image_slot=0,image_file_sha256=h.sha(ip),image_sha256=h.image_sha(image),new_VAR_calls=int(state['kind']!='gray'))
                        h.save(jp,v)
                    states[key]=v
                v=states[key];body=rx['body'];header=rx['header'];ec=f['branch']=='entropy'
                bodyok=(None if body is None else body['crc_accepted'])if ec else rx['body_crc_accept']
                diagnostics=token_diagnostics(state,tokens) # Offline only, after image selection.
                rows.append(dict(source_index=i,source_id=rec['source_id'],preprocessing_id=rec['preprocessing_id'],method_id=f['method_id'],point_id=f['point_id'],snr_db=f['snr_db'],noise_seed=f['noise_seed'],N=2048,
                    status=evidence['source_status'],branch=f['branch'],candidate_id=f['candidate_id'],target_m=f['target_m'],actual_m=f['actual_m'],K=f['K'],q=f['q'],nominal_rate=f['nominal_rate'],
                    tokens=rawcal.OFFSETS[f['actual_m']]+f['K'],header_symbols=68,body_symbols=1980 if ec else tx['body_symbols'],padding_symbols=0 if ec else tx['padding_symbols'],
                    arithmetic_bits=tx['arithmetic_bits']if ec else None,source_bits=tx['arithmetic_bits']if ec else tx['source_bits'],k=tx['k'],n=tx['n'],actual_code_rate=tx['k']/tx['n'],
                    known_information_padding_bits=tx['known_information_padding_bits']if ec else 0,E_frame=tx['E_frame'],rho=tx['rho'],
                    header_ok=header['header_ok'],header_crc_ok=header['header_crc_ok'],header_fields_legal=header['header_fields_legal'],received_profile_id=header['profile_id'],
                    body_crc_accept=bodyok,parser_accepted=None if body is None else(body['parser_accepted']if ec else True),
                    canonical_source_accepted=(evidence['source_status']=='ARITHMETIC_SOURCE_DECODED')if ec else None,
                    source_decode_cache_hit=evidence['source_decode_cache_hit'],gray=state['kind']=='gray',received_m=state['m'],received_K=state['K'],
                    fallback_attempts=h.canonical(f['fallback_attempts']),received_state_sha256=key,physical_frame_path=f['physical_frame']['path'],physical_frame_sha256=f['physical_frame']['sha256'],
                    image_path=v['image_path'],image_key=v['image_key'],image_slot=v['image_slot'],image_sha256=v['image_sha256'],image_file_sha256=v['image_file_sha256'],
                    **diagnostics,undetected_token_error_diagnostic=bool(bodyok and diagnostics['token_error_count']is not None and diagnostics['token_error_count']>0)))
            rendercalls=sum(v['new_VAR_calls']for v in states.values());decodecalls=sum(h.checked(d)['evidence']['new_VAR_source_decode']for d in pins.values())
            h.require(rendercalls<=budget['new_VAR_render_cap']and decodecalls<=budget['new_VAR_source_decode_cap'],'Finite actual-RX neural budget exceeded')
            h.save(cp,dict(status='T6_CONFIRMATION_RENDER_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],rows=rows,cpu_source=budget['cpu_source'],source_decode_checkpoints=list(pins.values()),
                new_VAR_render_calls=rendercalls,new_VAR_source_decode_calls=decodecalls,new_metric_calls=0))
            print(h.canonical(dict(stage='T6_CONFIRMATION_RENDER',source=i,unique_states=len(states),new_VAR_calls=rendercalls)),flush=True)
        native.frozen()
        if parallel:cohort.require_available()


def close(path,receipt):
    r,rh=registered(path);out=Path(r['out']);owner=h.read(receipt)
    h.require(owner['request_sha256']==rh and owner['actual_children_waited']is True and owner['worker_exit_codes']==[0]*r['gpu_workers'],'Real owner wait and exit receipt required')
    rows=[];outputs={str(Path(receipt).resolve()):h.sha(receipt)};renders=decodes=0
    for rec in r['records']:
        p=out/'gpu_sources'/f'{rec["source_index"]:04d}.json';cp=h.read(p);h.require(cp['request_sha256']==rh and cp['source_id']==rec['source_id'],'All100 exact sources required')
        outputs[str(p)]=h.sha(p);cpu=h.checked(cp['cpu_source']);outputs[cp['cpu_source']['path']]=cp['cpu_source']['sha256'];rows.extend(cp['rows'])
        for f in cpu['frames']:h.checked(f['physical_frame']);outputs[f['physical_frame']['path']]=f['physical_frame']['sha256']
        for dp in cp['source_decode_checkpoints']:h.checked(dp);outputs[dp['path']]=dp['sha256']
        renders+=cp['new_VAR_render_calls'];decodes+=cp['new_VAR_source_decode_calls']
    expected={(m,s,i,n)for m in r['methods']for s in SNRS for i in range(100)for n in SEEDS}
    h.require(len(rows)==2700 and{(x['method_id'],x['snr_db'],x['source_index'],x['noise_seed'])for x in rows}==expected,'All2700 conditions including failures, no reselection')
    for row in rows:
        p=row['image_path'];v=row['image_file_sha256']
        if p not in outputs:h.require(h.sha(p)==v,'Actual referenced image changed');outputs[p]=v
        else:h.require(outputs[p]==v,'Conflicting image seals')
    write_csv(out/'per_frame.csv',rows);outputs[str(out/'per_frame.csv')]=h.sha(out/'per_frame.csv')
    plan=out/'render_plan.json';outputs[str(plan)]=h.sha(plan);snap=cal.ledger_snapshot(out/'packet_ledger.sqlite',rh,5400)
    value=dict(status='T6_CONFIRMATION100_RENDER_COMPLETE',request=h.desc(path),request_sha256=rh,N=2048,source_count=100,source_ids=r['source_ids'],noise_count=3,noise_seeds=SEEDS,frame_count=2700,
        source_manifest=r['source_manifest'],calibration_freeze=r['calibration_freeze'],entropy_family_selection=r['entropy_family_selection'],confirmation_registration=r['confirmation_registration'],
        content_duplicate_check_completion=r['content_duplicate_check_completion'],actual_children_waited=True,worker_exit_codes=owner['worker_exit_codes'],
        actual_packet_ledger=snap,new_VAR_render_calls=renders,new_VAR_source_decode_calls=decodes,qualification_VAR_calls=r['gpu_workers'],new_metric_calls=0,new_bootstrap_calls=0,outputs=outputs)
    h.save(out/'completion.json',value);return{k:v for k,v in value.items()if k not in('outputs','source_ids')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);q=s.add_parser('prepare')
    for x in('source-completion','calibration-freeze','environment-request','out'):q.add_argument('--'+x,required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--workers',type=int,default=8);q.add_argument('--gpu-workers',type=int,choices=(1,2),default=2)
    for cmd in('ledger-init','cpu-worker','plan-render','gpu-worker','close'):
        q=s.add_parser(cmd);q.add_argument('--request',required=True)
        if cmd=='cpu-worker':q.add_argument('--branch',choices=('raw','entropy'),required=True);q.add_argument('--index',type=int,required=True)
        if cmd=='close':q.add_argument('--owner-receipt',required=True)
    a=p.parse_args();f={'ledger-init':ledger_init,'plan-render':plan_render,'gpu-worker':gpu_worker}
    value=prepare(a)if a.command=='prepare'else cpu_worker(a.request,a.branch,a.index)if a.command=='cpu-worker'else close(a.request,a.owner_receipt)if a.command=='close'else f[a.command](a.request)
    if value is not None:print(h.canonical(value),flush=True)
