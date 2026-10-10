"""Finite single-family N2048 entropy calibration with actual paid PHY.

No training or confirmation input. Frozen N1024 calibration chooses the entropy
family. Constructor admission, first100 pilot and finite full1000 recheck each
precede the next stage; no automatic successor is launched.
"""
import argparse,contextlib,csv,json,math,os,signal,subprocess,time
from pathlib import Path
import numpy as np
import t2_pilot as h
import t1_calibrate as cal
import t6_raw_calibrate as raw
import t6_entropy_phy as phy
from t6_source_codec import SourceCodec10
from t1_source_asset_schema import model_id
from t6_plan import write_csv

SCHEMA='WCL_T6_N2048_ENTROPY_CALIBRATION_EXECUTION_V1'
SNRS=[4,10,19]


def prepare(a):
    done=h.read(a.source_completion);mf=h.checked(done['source_manifest']);family=h.read(a.entropy_family_selection)
    h.require(done['status']=='T6_CALIBRATION_ENTROPY_SOURCE_ASSETS_COMPLETE'and done['family']==family['family']
        and done['entropy_family_selection']==h.desc(a.entropy_family_selection),'Sealed selected-family m10 source assets required')
    h.require(family['status']=='T1_SINGLE_ENTROPY_FAMILY_FROZEN_CALIBRATION_ONLY'
        and family['family']=='EC_VAR_WHOLE'and not family['holdout_used_for_selection']and not family['holdout_files_read'],'N1024 calibration-only family choice required')
    oldfreeze=h.checked(family['original_policy_freeze']);static=oldfreeze['protocol']['static_completion'];h.checked(static)
    qual=h.read(a.phy_qualification);qr=h.checked(qual['request']);h.require(qual['status']=='PASS'and qual['schema']==phy.QUAL_SCHEMA
        and qual['entropy_family_selection']==h.desc(a.entropy_family_selection)and qual['packet_decode_count']==qr['packet_cap']
        and qual['ledger']==dict(total=qr['packet_cap'],cap=qr['packet_cap'],unresolved=0),'Actual N2048 single-family PHY qualification first')
    phy.old.verify_bindings(qual['outputs']);phy.old.verify_bindings(qr['source_bindings'])
    env=h.read(a.environment_request);t2=h.read(a.t2_completion)
    h.require(env['schema']=='WCL_T2_FULL1000_RECHECK_V1'and t2['status']=='T2_FULL1000_CALIBRATION_COMPLETE_NOT_HOLDOUT'
        and t2['request_sha256']==h.sha(a.environment_request),'Exact completed original calibration environment')
    count=100 if a.phase=='pilot' else 1000;seeds=[4101]if count==100 else[4101,4102,4103]
    h.require(done['source_count']>=count and done['original_calibration_source_ids']==[e['source_id']for e in env['records']]
        and mf['source_count']==done['source_count']and len(mf['records'])==done['source_count'], 'Original1000 calibration order required')
    records=mf['records'][:count];oldrecords=env['records'][:count]
    h.require([e['source_index']for e in records]==list(range(count))and[e['source_id']for e in records]==[e['source_id']for e in oldrecords],'Exact first100/all1000 source membership')
    candidates=qr['admitted_candidates'];byid={c['candidate_id']:c for c in candidates};pilot=None;pr=None
    if a.phase=='pilot':schedule=[dict(snr_db=s,candidate_id=cid,candidate=byid[cid])for s in SNRS for cid in sorted(byid)]
    else:
        h.require(a.pilot_completion,'Completed entropy pilot first');pilot=h.read(a.pilot_completion);pr=h.checked(pilot['request'])
        h.require(pilot['status']=='T6_ENTROPY_PILOT_CALIBRATION_COMPLETE'and pr['family']==family['family']
            and pr['phy_qualification']==h.desc(a.phy_qualification)and pr['candidates']==candidates
            and pr['environment_request']==h.desc(a.environment_request)and pr['static_completion']==static
            and pr['entropy_family_selection']==h.desc(a.entropy_family_selection),'Same complete N2048 pilot required')
        rp=Path(pilot['rankings_path']);h.require(pilot['outputs'][str(rp)]==h.sha(rp),'Pilot rankings seal')
        with rp.open(newline='')as f:rows=list(csv.DictReader(f))
        schedule=[]
        for s in SNRS:
            sr=[x for x in rows if int(x['snr_db'])==s]
            h.require({x['candidate_id']for x in sr}==set(byid)and len(sr)==len(byid),'All constructor-legal pilot candidates required')
            h.require(all(int(x['source_count'])==100 and int(x['noise_count'])==1 and math.isfinite(float(x['dinov2_vitl14_cosine']))for x in sr),'Complete finite pilot source means')
            selected=sorted(sr,key=lambda x:(-float(x['dinov2_vitl14_cosine']),x['candidate_id']))[:3]
            schedule.extend(dict(snr_db=s,candidate_id=x['candidate_id'],candidate=byid[x['candidate_id']])for x in selected)
    for entry,old in zip(records,oldrecords):
        sr,arrays=cal.load_source(entry)
        h.require(done['outputs'][entry['checkpoint']]==entry['checkpoint_sha256']and done['outputs'][sr['archive']['path']]==sr['archive']['sha256']
            and sr['source_assets_checkpoint']==dict(path=old['checkpoint'],sha256=old['checkpoint_sha256'])
            and sr['static_model_completion_sha256']==static['sha256'],'Exact original tokens and fixed source models')
        h.require(sr['m10_independent_roundtrip']and sr['N']==2048,'Actual independent m10 qualification required')
        for c in candidates:cal.selected_stream(sr,arrays,family['family'],c)
    root=Path(env['root']);out=Path(a.out).resolve();h.require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'Independent WCL output')
    frames=len(schedule)*count*len(seeds);h.require(a.packet_cap==frames*2,'Explicit finite maximum packet cap')
    r=dict(schema=SCHEMA,N=2048,root=str(root),out=str(out),phase=a.phase,family=family['family'],records=records,old_records=oldrecords,
        source_count=count,source_ids=[e['source_id']for e in records],snrs=SNRS,noise_seeds=seeds,candidates=candidates,schedule=schedule,
        frame_count=frames,packet_cap=2*frames,workers=a.workers,gpu_workers=a.gpu_workers,
        source_completion=h.desc(a.source_completion),source_manifest=done['source_manifest'],entropy_family_selection=h.desc(a.entropy_family_selection),
        static_completion=static,source_model_id={family['family']:model_id(family['family'],static['sha256'])},
        phy_qualification=h.desc(a.phy_qualification),registered_profile_catalogue=qr['catalogue'],environment_request=h.desc(a.environment_request),t2_completion=h.desc(a.t2_completion),
        pilot_completion=h.desc(a.pilot_completion)if pilot else None,pilot_execution_request=pilot['request']if pilot else None,
        source_bindings=dict(env['source_bindings']),native_source_bindings=dict(env['native_source_bindings']),
        old_visual_identity=env['old_visual_identity'],old_score_identity=env['old_score_identity'],old_numerical_runtime=env['old_numerical_runtime'],
        visual_config=env['visual_config'],input_bindings=env['input_bindings'],python_gpu=env['python_gpu'],python_cpu=env['python_cpu'],
        qualification=env['qualification'],gpu_cohort_runtime=env['gpu_cohort_runtime'],
        noise_rule='t6_raw_phy.standard_noise: same new N2048 full2048 standard variates for source/seed, all methods',
        counter_rule='(index_in_[4,10,19]*1000+source_index)*3+noise_seed-4101',
        policy_selection='mean within source over registered noises then mean source DINO-L; lexical candidate ties',
        deadline_unix=a.deadline_unix,max_seconds=172800,stop_files=[str(root/'STOP'),str(out/'STOP')],
        holdout_used_for_selection=False,confirmation_images_opened=False,training_updates=0,automatic_successor=False)
    r['t1_calibration_cache']=None
    if a.t1_completion:
        t1=h.read(a.t1_completion);t1r=h.checked(dict(path=t1['request_path'],sha256=t1['request_sha256']));t1env=h.checked(t1r['environment_request'])
        h.require(t1['status']=='T1_FULL_CALIBRATION_COMPLETE'and t1['source_count']==1000 and t1['noise_count']==3
            and not t1['holdout_used']and t1['policy_selection_metric']=='dinov2_vitl14_cosine_only','Completed original1000 T1 cache required')
        for k in('old_visual_identity','old_score_identity','old_numerical_runtime'):
            h.require(t1env[k]==env[k],'T1 entropy cache has different visual/scorer/numerical identity')
        h.require(len(t1r['old_records'])==1000,'Complete original1000 T1 cache required')
        for old,current in zip(t1r['old_records'],env['records']):
            h.require(all(old[k]==current[k]for k in('source_id','source_index','preprocessing_id','tokens_sha256','archive','archive_sha256')),'T1 cache source assets differ')
        r['t1_calibration_cache']=dict(completion=h.desc(a.t1_completion),request=dict(path=t1['request_path'],sha256=t1['request_sha256']))
    v=r['visual_config'];closure=h.load(v['static_closure_module'],'_t6_ec_closure',r['source_bindings'])
    bindings=closure.collect_bindings(root,v['native_runtime'],v['var_source'],v['dino_source'],v['uep_runtime'])
    for p,value in r['native_source_bindings'].items():h.require(bindings.get(p)==value,'Previously admitted native source changed')
    r['native_source_bindings']=bindings;r['source_bindings'].update(bindings)
    for name in bound_names():
        p=str(Path(__file__).with_name(name).resolve());r['source_bindings'][p]=h.sha(p)
    validate(r);p=out/'execution_request.json';h.save(p,r)
    return dict(status='T6_ENTROPY_FINITE_CALIBRATION_REGISTERED_NOT_RUN',request=h.desc(p),source_count=count,candidate_count=len(candidates),frame_count=frames,packet_cap=frames*2)


def bound_names():return('t6_entropy_calibrate.py','t6_entropy_gpu_parallel.py','t6_entropy_phy.py','t6_qualify_entropy_phy.py',
    't6_source_codec.py','t6_calibration_plan.py','t6_raw_phy.py','t6_raw_calibrate.py','t6_plan.py',
    't1_calibrate.py','t1_codec_runtime.py','t1_source_asset_schema.py','t1_entropy_core.py','t2_pilot.py','t2_ledger.py')


def validate(r):
    count=100 if r['phase']=='pilot' else 1000
    h.require(r['schema']==SCHEMA and r['N']==2048 and r['source_count']==count and r['snrs']==SNRS
        and r['noise_seeds']==([4101]if count==100 else[4101,4102,4103]),'Finite original calibration scope')
    h.require(r['frame_count']==count*len(r['noise_seeds'])*len(r['schedule'])and r['packet_cap']==2*r['frame_count'],'Finite registered packet cap')
    h.require(r['gpu_workers']in(1,2)and 1<=r['workers']<=16 and len(r['records'])==count and len(r['old_records'])==count,'Finite workers/source scope')
    h.require([e['source_id']for e in r['records']]==r['source_ids']==[e['source_id']for e in r['old_records']],'Source identities agree')
    h.require([e['source_index']for e in r['records']]==list(range(count))and len(set(r['source_ids']))==count,'Unique ordered sources')
    h.require(r['phase']in('pilot','full')and r['family']=='EC_VAR_WHOLE','Single preregistered entropy family')
    candidates={c['candidate_id']:c for c in r['candidates']}
    h.require(len(candidates)==len(r['candidates'])and len(candidates)>=3,'Unique admitted candidates')
    legal={(p['q'],p['nominal_rate'])for p in r['registered_profile_catalogue']['profiles']}
    from t6_calibration_plan import entropy_candidates
    expected=[dict(c,actual_ldpc_layout_status='ACTUAL_CONSTRUCTOR_ADMITTED')for c in entropy_candidates()if(c['q'],c['nominal_rate'])in legal]
    h.require(r['candidates']==expected,'Candidate set is exactly all actual constructor-admitted resource queries')
    h.require(len({(p['snr_db'],p['candidate_id'])for p in r['schedule']})==len(r['schedule'])and
        all(p['snr_db']in SNRS and p['candidate']==candidates[p['candidate_id']]for p in r['schedule']),'Unique unchanged candidate schedule')
    h.require(all(sum(p['snr_db']==s for p in r['schedule'])==(len(candidates)if r['phase']=='pilot'else 3)for s in SNRS),'All pilot candidates or exactly three full candidates per SNR')
    h.require(not r['holdout_used_for_selection']and not r['confirmation_images_opened'],'No confirmation selection')


def registered(path):
    r=h.read(path);validate(r)
    for name in bound_names():
        p=str(Path(__file__).with_name(name).resolve());h.require(r['source_bindings'][p]==h.sha(p),'Bound EC runner changed')
    for key in('source_completion','source_manifest','entropy_family_selection','static_completion','phy_qualification','environment_request','t2_completion'):
        h.checked(r[key])
    return r,h.sha(path)


def ledger_init(path):
    r,rh=registered(path);out=Path(r['out'])
    with h.lock(out/'ledger_initialize.lock'):
        meter=raw.Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap']);snap=meter.snapshot();meter.close()
        h.require(snap['unresolved']==0,'Unresolved actual packet event requires audit');h.save(out/'ledger_initialized.json',dict(request_sha256=rh,packet_cap=r['packet_cap']))
    return snap


def cpu_worker(path,index):
    r,rh=registered(path);out=Path(r['out']);began=time.monotonic()
    h.require(os.environ.get('CUDA_VISIBLE_DEVICES')==''and 0<=index<r['workers'],'Explicit valid CPU worker')
    for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
    with h.lock(out/f'cpu_worker_{index}.lock'):
        rt=phy.create_runtime(r['root'],r['phy_qualification']['path']);meter=raw.open_ledger(r,rh);completed=[]
        h.require(rt.catalogue==r['registered_profile_catalogue'],'Exact qualified public headers required')
        profiles={(p['m'],p['q'],p['nominal_rate']):p for p in rt.catalogue['profiles']}
        pilot=h.checked(r['pilot_execution_request'])if r['pilot_execution_request']else None;pd=h.checked(r['pilot_completion'])if pilot else None
        try:
            for entry in r['records'][index::r['workers']]:
                h.guard(r,began);i=entry['source_index'];cp=out/'cpu_sources'/f'{i:04d}.json'
                if cp.exists():h.require(h.read(cp)['request_sha256']==rh,'Saved CPU source differs');completed.append(h.desc(cp));continue
                sr,arrays=cal.load_source(entry);prior={}
                if pilot and i<100:
                    pc=raw.prior_checkpoint(pilot,pd,'cpu_sources',i)
                    for f in pc['frames']:
                        packet=h.checked(f['physical_frame']);prior[(f['snr_db'],f['noise_seed'],packet['TX_profile_id'])]=packet
                frames=[]
                for point in r['schedule']:
                    snr=point['snr_db'];c=point['candidate'];m,bits,selection=cal.selected_stream(sr,arrays,r['family'],c);p=profiles[(m,c['q'],c['nominal_rate'])]
                    for seed in r['noise_seeds']:
                        h.guard(r,began);counter=raw.counter(i,snr,seed);pid=p['profile_id'];pp=out/'physical_frames'/f'{i:04d}'/f'{snr}_{seed}_{pid:03d}.json'
                        if pp.exists():
                            packet=h.read(pp);h.require(packet['request_sha256']==rh and packet['payload_sha256']==phy.array_sha(bits),'Saved actual physical frame differs')
                        else:
                            wave,tx=rt.transmit(pid,bits,counter);noise=phy.standard_noise(entry['source_id'],seed)/math.sqrt(10**(snr/10));observed=wave+noise
                            expected=dict(source_id=entry['source_id'],source_index=i,snr_db=snr,noise_seed=seed,TX_profile_id=pid,
                                payload_sha256=phy.array_sha(bits),noise_sha256=phy.array_sha(noise),observation_sha256=phy.array_sha(observed),public_frame_counter=counter,transmission=tx)
                            previous=prior.get((snr,seed,pid))
                            if previous is None:
                                event=f'{SCHEMA}/{rh[:16]}/source{i}/snr{snr}/noise{seed}/profile{pid}'
                                rx=rt.receive(observed,snr,counter,rt.profiles,meter,event);reuse=False
                            else:
                                h.require(all(previous[k]==v for k,v in expected.items()),'Exact N2048 entropy pilot observation differs');rx=previous['actual_RX'];reuse=True
                            packet=dict(expected,request_sha256=rh,actual_RX=rx,pilot_actual_RX_reused=reuse);h.save(pp,packet)
                        frames.append(dict(source_index=i,source_id=entry['source_id'],snr_db=snr,noise_seed=seed,family=r['family'],candidate_id=c['candidate_id'],
                            target_m=c['target_m'],actual_m=m,q=c['q'],nominal_rate=c['nominal_rate'],fallback_attempts=selection['attempts'],physical_frame=h.desc(pp)))
                h.save(cp,dict(status='T6_ENTROPY_CPU_SOURCE_COMPLETE',request_sha256=rh,source_id=entry['source_id'],frames=frames));completed.append(h.desc(cp))
                print(h.canonical(dict(stage='T6_ENTROPY_CPU',worker=index,source=i,source_count=r['source_count'],logical_frames=len(frames))),flush=True)
            h.save(out/'cpu_workers'/f'{index:02d}.json',dict(status='COMPLETE',request_sha256=rh,sources=completed))
        finally:meter.close()


class NeedsIndependentDecode(Exception):pass
class NoModelCodec:
    def decode(self,*_):raise NeedsIndependentDecode()


def source_cache(r,rec,i):
    # Old raw and T2 score identities are checked in the GPU constructor. Actual
    # entropy pilot states are obtained from their sealed recovery proofs.
    cache=raw.cache_records(dict(r,pilot_execution_request=None),rec)
    if r['pilot_execution_request']and i<100:
        pr=h.checked(r['pilot_execution_request']);pd=h.checked(r['pilot_completion']);pc=raw.prior_checkpoint(pr,pd,'gpu_sources',i)
        states={}
        for d in pc['source_decode_checkpoints']:
            v=h.checked(d);states[h.digest(v['state'])]=v['state']
        for row in pc['rows']:
            k=row['received_state_sha256'];h.require(k in states,'Pilot image requires actual source-decode proof')
            value=dict(state=states[k],image_path=row['image_path'],image_key=row['image_key'],image_slot=row['image_slot'],
                image_file_sha256=row['image_file_sha256'],image_sha256=row['image_sha256'],metrics={m:row[m]for m in h.METRICS},reuse='N2048_ENTROPY_PILOT_EXACT_STATE')
            if k in cache:h.require(cache[k]['image_sha256']==value['image_sha256']and cache[k]['metrics']==value['metrics'],'Conflicting exact-state scores')
            else:cache[k]=value
    return cache


def plan_gpu(path):
    r,rh=registered(path);out=Path(r['out']);outputs={};newrenders=newmetrics=decodes=hits=0
    for entry,rec in zip(r['records'],r['old_records']):
        i=entry['source_index'];cp=out/'cpu_sources'/f'{i:04d}.json';cpu=h.read(cp)
        h.require(cpu['request_sha256']==rh and len(cpu['frames'])==len(r['schedule'])*len(r['noise_seeds']),'Complete actual CPU source first')
        sr,arrays=cal.load_source(entry);known={};pending={};recoveries={};cache=source_cache(r,rec,i)
        for f in cpu['frames']:
            packet=h.checked(f['physical_frame']);identity=cal.recovery_input(packet,r['source_model_id']);key=h.digest(identity)
            if key in recoveries:continue
            recoveries[key]=f['physical_frame']
            try:state,evidence=cal.recover(sr,arrays,packet,NoModelCodec(),r['source_model_id']);known[h.digest(state)]=state
            except NeedsIndependentDecode:pending[key]=identity
        matches={k:cache[k]for k in known if k in cache};nr=sum(s['kind']!='gray'for k,s in known.items()if k not in matches)+len(pending)
        nm=len(known)-len(matches)+len(pending);nd=sum(v['actual_received_profile']['family']=='EC_VAR_WHOLE'for v in pending.values())
        p=out/'gpu_plan_sources'/f'{i:04d}.json';value=dict(request_sha256=rh,source_id=entry['source_id'],cpu_source=h.desc(cp),
            known_states=known,pending_independent_decodes=pending,recovery_inputs=recoveries,cache=cache,
            new_VAR_render_cap=nr,new_metric_cap=nm,new_VAR_source_decode_cap=nd,known_cache_state_hits=len(matches))
        h.save(p,value);outputs[str(p)]=h.sha(p);newrenders+=nr;newmetrics+=nm;decodes+=nd;hits+=len(matches)
    ledger=cal.ledger_snapshot(out/'packet_ledger.sqlite',rh,r['packet_cap'])
    value=dict(status='T6_ENTROPY_ACTUAL_RX_GPU_BUDGET_REGISTERED',request=h.desc(path),request_sha256=rh,
        new_VAR_render_cap=newrenders+r['gpu_workers'],new_metric_cap=newmetrics+r['gpu_workers'],new_VAR_source_decode_cap=decodes,
        reference_preparations=r['source_count']+r['gpu_workers'],qualification_VAR_calls=r['gpu_workers'],known_cache_state_hits=hits,
        unknown_actual_payloads_require_independent_decode=True,packet_ledger=ledger,outputs=outputs)
    h.save(out/'gpu_plan.json',value);return {k:v for k,v in value.items()if k!='outputs'}


def gpu_worker(path,cohort_config=None):
    r,rh=registered(path);out=Path(r['out']);began=time.monotonic();plan=h.read(out/'gpu_plan.json')
    h.require(plan['request_sha256']==rh and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Registered post-CPU budget and GPU0 required')
    parallel=None
    if cohort_config is None:
        h.require(r['gpu_workers']==1,'Two workers require owner');shared=h.lock(r['visual_config']['visual_lock']);own=h.lock(out/'gpu.lock');indices=range(r['source_count'])
    else:
        import t6_entropy_gpu_parallel as parallel
        cohort,probe=parallel.admit(r,rh,cohort_config);shared=contextlib.nullcontext();own=h.lock(out/f'gpu_worker_{cohort_config["worker_id"]}.lock');indices=cohort_config['source_indices']
    for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
    with shared,own:
        native,source,scorer=h.gpu_build(r,out);codec=SourceCodec10(r['root'],r['static_completion']['path'],native)
        if parallel:
            h.require(native.assets.old.b is probe,'Owned process admission differs');parallel.qualify(r,rh,native,source,scorer,cohort_config,cohort)
        else:h.qualify_gpu(dict(r,records=r['old_records']),rh,out,native,source,scorer)
        for i in indices:
            h.guard(r,began)
            if parallel:cohort.require_available()
            entry=r['records'][i];rec=r['old_records'][i];cp=out/'gpu_sources'/f'{i:04d}.json'
            if cp.exists():h.require(h.read(cp)['request_sha256']==rh,'Saved GPU source differs');continue
            pp=out/'gpu_plan_sources'/f'{i:04d}.json';h.require(plan['outputs'][str(pp)]==h.sha(pp),'GPU source budget changed');budget=h.read(pp);cpu=h.checked(budget['cpu_source'])
            sr,arrays=cal.load_source(entry);cache=dict(budget['cache']);checked={};olddecodes={}
            if r['pilot_execution_request']and i<100:
                previous=raw.prior_checkpoint(h.checked(r['pilot_execution_request']),h.checked(r['pilot_completion']),'gpu_sources',i)
                for d in previous['source_decode_checkpoints']:
                    v=h.checked(d);olddecodes[h.digest(v['recovery_input'])]=d
            _,pixels=h.source_assets(rec);record=h.score_record(rec);raw.prepare_reference_once(out,r,rh,scorer,record,pixels,i)
            rows=[];pins={};newstates={};decoded={}
            for f in cpu['frames']:
                h.guard(r,began);packet=h.checked(f['physical_frame']);key=h.digest(cal.recovery_input(packet,r['source_model_id']))
                if key not in decoded:
                    h.require(key in budget['recovery_inputs'],'Actual recovery identity absent from sealed CPU budget')
                    previous_path=out/'source_decodes'/f'{i:04d}'/(key+'.json')
                    if key in budget['pending_independent_decodes']and key not in olddecodes and not previous_path.exists():
                        used=sum(v[1]['new_VAR_source_decode']for v in decoded.values())
                        h.require(used<budget['new_VAR_source_decode_cap'],'No unbudgeted actual entropy decode')
                    state,evidence,dp=cal.recover_once(out/'source_decodes'/f'{i:04d}',rh,sr,arrays,packet,codec,r['source_model_id'],olddecodes)
                    pins[dp['path']]=dp;decoded[key]=(state,evidence)
                state,evidence=decoded[key];sk=h.digest(state)
                if sk not in cache:
                    jp=out/'gpu_states'/f'{i:04d}'/(sk+'.json');ip=jp.with_suffix('.npz');res=jp.with_suffix('.reserved.json')
                    if jp.exists():
                        v=h.read(jp);h.require(v['request_sha256']==rh and v['state']==state,'Saved state differs');raw.checked_image(v,{})
                    else:
                        h.require(len(newstates)<budget['new_metric_cap']and
                            (state['kind']=='gray'or sum(x['new_VAR_calls']for x in newstates.values())<budget['new_VAR_render_cap']),
                            'No unbudgeted render or metric call')
                        h.require(not res.exists(),'Unresolved render requires explicit audit');h.save(res,dict(request_sha256=rh,state=state))
                        image=h.render_state(native,source,state);metrics=scorer(record,[image])[0]
                        with ip.open('xb')as handle:np.savez(handle,image=image)
                        v=dict(request_sha256=rh,state=state,image_path=str(ip),image_key='image',image_slot=0,image_file_sha256=h.sha(ip),image_sha256=h.image_sha(image),
                            metrics={m:metrics[m]for m in h.METRICS},reuse='NEW_N2048_ENTROPY_STATE',new_VAR_calls=int(state['kind']!='gray'))
                        h.save(jp,v)
                    cache[sk]=v;newstates[sk]=v
                v=cache[sk];raw.checked_image(v,checked);rx=packet['actual_RX'];body=rx['body'];tx=packet['transmission']
                rows.append(dict(source_index=i,source_id=entry['source_id'],family=r['family'],candidate_id=f['candidate_id'],snr_db=f['snr_db'],noise_seed=f['noise_seed'],
                    N=2048,status='T6_ENTROPY_ACTUAL_CALIBRATION_FRAME_COMPLETE',target_m=f['target_m'],actual_m=f['actual_m'],K=0,q=f['q'],nominal_rate=f['nominal_rate'],
                    fallback_attempts=h.canonical(f['fallback_attempts']),header_ok=rx['header']['header_ok'],body_crc_accept=None if body is None else body['crc_accepted'],
                    parser_accepted=None if body is None else body['parser_accepted'],received_m=None if rx['rx_profile']is None else rx['rx_profile']['m'],
                    source_status=evidence['source_status'],source_decode_cache_hit=evidence['source_decode_cache_hit'],gray=state['kind']=='gray',
                    E_frame=tx['E_frame'],rho=tx['rho'],header_symbols=68,body_symbols=1980,padding_symbols=0,k=tx['k'],n=tx['n'],
                    arithmetic_bits=tx['arithmetic_bits'],known_information_padding_bits=tx['known_information_padding_bits'],
                    pilot_actual_RX_reused=packet['pilot_actual_RX_reused'],received_state_sha256=sk,image_path=v['image_path'],image_key=v['image_key'],image_slot=v['image_slot'],
                    image_file_sha256=v['image_file_sha256'],image_sha256=v['image_sha256'],visual_reuse=v['reuse'],**v['metrics']))
            actual_decode=sum(h.checked(d)['evidence']['new_VAR_source_decode']for d in pins.values());actual_renders=sum(x['new_VAR_calls']for x in newstates.values())
            h.require(actual_decode<=budget['new_VAR_source_decode_cap']and actual_renders<=budget['new_VAR_render_cap']and len(newstates)<=budget['new_metric_cap'],'Finite post-CPU GPU cap exceeded')
            h.save(cp,dict(status='T6_ENTROPY_GPU_SOURCE_COMPLETE',request_sha256=rh,source_id=entry['source_id'],rows=rows,cpu_source=budget['cpu_source'],
                source_decode_checkpoints=list(pins.values()),new_VAR_source_decode_calls=actual_decode,new_VAR_render_calls=actual_renders,new_metric_calls=len(newstates)))
            print(h.canonical(dict(stage='T6_ENTROPY_GPU',source=i,new_VAR_decode_calls=actual_decode,new_VAR_render_calls=actual_renders)),flush=True)
        native.frozen()
        if parallel:cohort.require_available()


def close(path,receipt):
    r,rh=registered(path);out=Path(r['out']);owner=h.read(receipt)
    h.require(owner['request_sha256']==rh and owner['actual_children_waited']and owner['worker_exit_codes']==[0]*r['gpu_workers'],'Actual owner child waits required')
    rows=[];outputs={str(Path(receipt).resolve()):h.sha(receipt)};decodes=renders=metrics=packet_calls=0;seen_packets=set()
    gp=out/'gpu_plan.json';plan=h.read(gp);h.require(plan['request_sha256']==rh,'Exact sealed post-CPU GPU budget required')
    outputs[str(gp)]=h.sha(gp)
    for p,digest in plan['outputs'].items():h.require(h.sha(p)==digest,'Changed GPU budget input');outputs[p]=digest
    qualifications=[out/'qualification_gpu.json']if r['gpu_workers']==1 else[out/'gpu_parallel_qualification'/f'{i:04d}.json'for i in range(2)]
    for qp in qualifications:
        q=h.read(qp);h.require(q['status']=='PASS'and q['request_sha256']==rh,'Actual GPU replay qualification required');outputs[str(qp)]=h.sha(qp)
    for e in r['records']:
        p=out/'gpu_sources'/f'{e["source_index"]:04d}.json';cp=h.read(p);h.require(cp['request_sha256']==rh and cp['source_id']==e['source_id'],'All source rows required')
        outputs[str(p)]=h.sha(p);cpu=h.checked(cp['cpu_source']);outputs[cp['cpu_source']['path']]=cp['cpu_source']['sha256'];rows.extend(cp['rows'])
        for f in cpu['frames']:
            packet=h.checked(f['physical_frame']);pp=f['physical_frame']['path']
            h.require(packet['request_sha256']==rh,'Packet belongs to another request')
            if pp not in seen_packets and not packet['pilot_actual_RX_reused']:packet_calls+=1+int(packet['actual_RX']['header']['header_ok'])
            seen_packets.add(pp);outputs[pp]=f['physical_frame']['sha256']
        for d in cp['source_decode_checkpoints']:h.checked(d);outputs[d['path']]=d['sha256']
        ref=out/'metric_references'/f'{e["source_index"]:04d}.json';archive=ref.with_suffix('.pt');reference=h.read(ref)
        h.require(reference['request_sha256']==rh and h.sha(archive)==reference['archive_sha256'],'Exact persistent source feature reference required')
        outputs[str(ref)]=h.sha(ref);outputs[str(archive)]=h.sha(archive)
        for row in cp['rows']:
            ip=row['image_path'];ih=row['image_file_sha256']
            if ip in outputs:h.require(outputs[ip]==ih,'Conflicting image seals')
            else:h.require(h.sha(ip)==ih,'Actual reconstructed or reused image changed');outputs[ip]=ih
            sp=out/'gpu_states'/f'{e["source_index"]:04d}'/(row['received_state_sha256']+'.json')
            if sp.exists():outputs[str(sp)]=h.sha(sp)
        decodes+=cp['new_VAR_source_decode_calls'];renders+=cp['new_VAR_render_calls'];metrics+=cp['new_metric_calls']
    expected={(i,p['snr_db'],p['candidate_id'],seed)for i in range(r['source_count'])for p in r['schedule']for seed in r['noise_seeds']}
    h.require(len(rows)==r['frame_count']and{(x['source_index'],x['snr_db'],x['candidate_id'],x['noise_seed'])for x in rows}==expected,'All actual conditions including failures retained')
    rankings=[];winners=[]
    for s in SNRS:
        group=[]
        for p in[x for x in r['schedule']if x['snr_db']==s]:
            sub=[x for x in rows if x['snr_db']==s and x['candidate_id']==p['candidate_id']];by={i:[]for i in range(r['source_count'])}
            for x in sub:by[x['source_index']].append(x)
            h.require(all(len(x)==len(r['noise_seeds'])for x in by.values()),'All source/noise conditions')
            means={m:math.fsum(math.fsum(float(x[m])for x in by[i])/len(r['noise_seeds'])for i in range(r['source_count']))/r['source_count']for m in h.METRICS}
            h.require(all(math.isfinite(v)for v in means.values()),'Finite complete source means required')
            group.append(dict(family=r['family'],snr_db=s,candidate_id=p['candidate_id'],source_count=r['source_count'],noise_count=len(r['noise_seeds']),
                target_m=p['candidate']['target_m'],q=p['candidate']['q'],nominal_rate=p['candidate']['nominal_rate'],
                gray_fraction=sum(x['gray']for x in sub)/len(sub),fallback_fraction=sum(x['actual_m']<x['target_m']for x in sub)/len(sub),
                body_crc_rejected_fraction=sum(x['body_crc_accept']is False for x in sub)/len(sub),**means))
        group.sort(key=lambda x:(-x['dinov2_vitl14_cosine'],x['candidate_id']));rankings.extend(dict(x,rank=i+1)for i,x in enumerate(group));winners.append(group[0])
    for name,values in [('per_frame.csv',rows),('calibration_rankings.csv',rankings)]:write_csv(out/name,values);outputs[str(out/name)]=h.sha(out/name)
    policy=dict(status='T6_ENTROPY_CALIBRATION_CANDIDATE_WINNERS',phase=r['phase'],family=r['family'],winners=winners,selection_used_confirmation=False,holdout_used_for_selection=False)
    h.save(out/'entropy_policy.json',policy);outputs[str(out/'entropy_policy.json')]=h.sha(out/'entropy_policy.json')
    ledger=cal.ledger_snapshot(out/'packet_ledger.sqlite',rh,r['packet_cap'])
    h.require(ledger['total']==packet_calls,'Actual physical packet proofs and ledger disagree')
    h.require(decodes<=plan['new_VAR_source_decode_cap']and renders+r['gpu_workers']<=plan['new_VAR_render_cap']and metrics+r['gpu_workers']<=plan['new_metric_cap'],'Actual total GPU calls exceed sealed plan')
    value=dict(status='T6_ENTROPY_PILOT_CALIBRATION_COMPLETE'if r['phase']=='pilot'else'T6_ENTROPY_FULL1000_CALIBRATION_COMPLETE',request=h.desc(path),request_sha256=rh,
        family=r['family'],source_count=r['source_count'],source_ids=r['source_ids'],noise_count=len(r['noise_seeds']),noise_seeds=r['noise_seeds'],frame_count=len(rows),
        rankings_path=str(out/'calibration_rankings.csv'),policy=h.desc(out/'entropy_policy.json'),packet_ledger=ledger,
        new_VAR_source_decodes=decodes,new_VAR_renders=renders,new_metric_calls=metrics,qualification_VAR_calls=r['gpu_workers'],qualification_metric_calls=r['gpu_workers'],
        actual_children_waited=True,worker_exit_codes=owner['worker_exit_codes'],selection_used_confirmation=False,holdout_used_for_selection=False,outputs=outputs)
    h.save(out/'completion.json',value);return {k:v for k,v in value.items()if k not in('outputs','source_ids')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);q=s.add_parser('prepare')
    for x in('source-completion','entropy-family-selection','phy-qualification','environment-request','t2-completion','out'):q.add_argument('--'+x,required=True)
    q.add_argument('--phase',choices=('pilot','full'),required=True);q.add_argument('--pilot-completion');q.add_argument('--t1-completion');q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--packet-cap',type=int,required=True);q.add_argument('--workers',type=int,default=8);q.add_argument('--gpu-workers',type=int,choices=(1,2),default=2)
    for cmd in('ledger-init','cpu-worker','plan-gpu','gpu-worker','close'):
        q=s.add_parser(cmd);q.add_argument('--request',required=True)
        if cmd=='cpu-worker':q.add_argument('--index',type=int,required=True)
        if cmd=='close':q.add_argument('--owner-receipt',required=True)
    a=p.parse_args();f={'ledger-init':ledger_init,'plan-gpu':plan_gpu,'gpu-worker':gpu_worker}
    value=prepare(a)if a.command=='prepare'else cpu_worker(a.request,a.index)if a.command=='cpu-worker'else close(a.request,a.owner_receipt)if a.command=='close'else f[a.command](a.request)
    if value is not None:print(h.canonical(value),flush=True)
