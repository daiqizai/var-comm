"""Actual finite N2048 raw calibration, with exact received-state reuse.

CPU owners finish first. A sealed, exact state inventory then fixes the GPU
budget before neural execution. Old N1024 receptions are never relabelled.
"""
import argparse,contextlib,csv,io,json,math,os,signal,sqlite3,subprocess,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import t2_pilot as h
import t6_raw_phy as phy
from t2_ledger import Ledger
from t6_calibration_plan import SNRS
from t6_plan import write_csv

SCHEMA='WCL_T6_N2048_RAW_CALIBRATION_EXECUTION_V1'
OFFSETS=[0,1,5,14,30,55,91,155,255,424,680]


def prepare(a):
    plan=h.read(a.plan_request);h.require(plan['schema']=='WCL_T6_N2048_RAW_CALIBRATION_V1','Frozen finite plan required')
    env=h.checked(plan['environment_request']);original=h.checked(plan['original_calibration_request'])
    h.require(env['records']==original['records'] and env['old_score_identity']==original['old_score_identity'],'Exact original calibration environment required')
    prior=h.read(a.t2_completion);h.require(prior['status']=='T2_FULL1000_CALIBRATION_COMPLETE_NOT_HOLDOUT'
        and prior['request_sha256']==plan['environment_request']['sha256'],'Completed exact T2 calibration cache required')
    r=dict(plan);r.update(schema=SCHEMA,calibration_plan=h.desc(a.plan_request),t2_completion=h.desc(a.t2_completion),
        source_bindings=dict(env['source_bindings']),native_source_bindings=dict(env['native_source_bindings']),
        old_visual_identity=env['old_visual_identity'],old_score_identity=env['old_score_identity'],old_numerical_runtime=env['old_numerical_runtime'],
        visual_config=env['visual_config'],input_bindings=env['input_bindings'],python_gpu=env['python_gpu'],python_cpu=env['python_cpu'],
        qualification=env['qualification'],gpu_workers=a.gpu_workers,gpu_qualification_VAR_calls=a.gpu_workers,
        gpu_cohort_runtime=env['gpu_cohort_runtime'],execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r['root'],text=True).strip())
    r['t1_calibration_cache']=None
    if a.t1_completion:
        t1=h.read(a.t1_completion);t1r=h.checked(dict(path=t1['request_path'],sha256=t1['request_sha256']));t1env=h.checked(t1r['environment_request'])
        h.require(t1['status']=='T1_FULL_CALIBRATION_COMPLETE'and t1['source_count']==1000 and t1['noise_count']==3
            and not t1['holdout_used']and t1['policy_selection_metric']=='dinov2_vitl14_cosine_only','Completed original1000 T1 DINO-L calibration required')
        for key in('old_visual_identity','old_score_identity','old_numerical_runtime'):
            h.require(t1env[key]==env[key],'T1 cache uses a different renderer/scorer/numerical identity')
        for old,current in zip(t1r['old_records'],env['records']):
            h.require(all(old[k]==current[k]for k in('source_id','source_index','preprocessing_id','tokens_sha256','archive','archive_sha256')),'T1 cache original source differs')
        h.require(len(t1r['old_records'])==1000,'Complete original1000 T1 record bindings required')
        r['t1_calibration_cache']=dict(completion=h.desc(a.t1_completion),request=dict(path=t1['request_path'],sha256=t1['request_sha256']))
    v=r['visual_config'];closure=h.load(v['static_closure_module'],'_t6_raw_closure',r['source_bindings'])
    bindings=closure.collect_bindings(r['root'],v['native_runtime'],v['var_source'],v['dino_source'],v['uep_runtime'])
    for p,value in r['native_source_bindings'].items():h.require(bindings.get(p)==value,'Previously admitted native source changed')
    r['native_source_bindings']=bindings;r['source_bindings'].update(bindings)
    for name in('t6_raw_calibrate.py','t6_raw_gpu_parallel.py','t6_raw_phy.py','t6_calibration_plan.py','t2_pilot.py','t2_ledger.py'):
        p=str(Path(__file__).with_name(name).resolve());r['source_bindings'][p]=h.sha(p)
    # The public N2048 catalogue/session are new. Only its own pilot RX can be
    # reused by the full stage, with waveform/observation/counter equality.
    if plan['pilot_completion']:
        pc=h.checked(plan['pilot_completion']);pr=h.checked(pc['request'])
        h.require(pr['schema']==SCHEMA and pr['phase']=='pilot' and pr['phy_qualification']==r['phy_qualification']
            and pr['old_score_identity']==r['old_score_identity'] and pr['old_visual_identity']==r['old_visual_identity']
            and pr['old_numerical_runtime']==r['old_numerical_runtime'],'Only exact same-protocol N2048 pilot may be reused')
        r['pilot_execution_request']=pc['request']
    else:r['pilot_execution_request']=None
    validate(r);p=Path(r['out'])/'execution_request.json';h.save(p,r)
    return dict(status='T6_RAW_EXECUTION_REGISTERED_NOT_RUN',request=h.desc(p),frame_count=r['frame_count'],packet_cap=r['packet_cap'],gpu_workers=r['gpu_workers'])


def validate(r):
    n=100 if r['phase']=='pilot' else 1000;seeds=[4101]if n==100 else[4101,4102,4103]
    h.require(r['schema']==SCHEMA and r['N']==2048 and r['source_count']==n and r['noise_seeds']==seeds and r['snrs']==SNRS,'Fixed finite N2048 calibration scope')
    h.require([x['source_index']for x in r['records']]==list(range(n))and r['source_ids']==[x['source_id']for x in r['records']],'Original ordered calibration sources')
    h.require(r['frame_count']==len(r['schedule'])*n*len(seeds) and r['packet_cap']==2*r['frame_count'],'Finite cap')
    h.require(r['gpu_workers']in(1,2)and 1<=r['workers']<=16,'Finite workers')
    h.require(len({(p['snr_db'],p['profile_id'])for p in r['schedule']})==len(r['schedule']),'Unique source/SNR/wire conditions')
    h.require(not r['holdout_used_for_selection']and not r['confirmation_images_opened'],'Calibration only')


def registered(path):
    r=h.read(path);validate(r)
    for name in('t6_raw_calibrate.py','t6_raw_gpu_parallel.py','t6_raw_phy.py','t6_calibration_plan.py','t2_pilot.py','t2_ledger.py'):
        p=str(Path(__file__).with_name(name).resolve());h.require(r['source_bindings'][p]==h.sha(p),'Bound execution source changed')
    return r,h.sha(path)


def ledger_init(path):
    r,rh=registered(path);out=Path(r['out'])
    with h.lock(out/'ledger_initialize.lock'):
        meter=Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap']);snap=meter.snapshot();meter.close()
        h.require(snap['unresolved']==0,'Unresolved packet operation requires explicit audit')
        h.save(out/'ledger_initialized.json',dict(request_sha256=rh,packet_cap=r['packet_cap']))
    return snap


def open_ledger(r,rh):
    import fcntl
    out=Path(r['out']);h.require(h.read(out/'ledger_initialized.json')==dict(request_sha256=rh,packet_cap=r['packet_cap']),'Parent must initialize WAL before spawning children')
    # Only WAL constructors serialize; decoder callbacks run independently.
    with(out/'ledger_open.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:return Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap'])
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def counter(i,s,n):return(SNRS.index(s)*1000+i)*3+n-4101
def frame_path(out,i,s,p,n):return out/'cpu_frames'/f'{i:04d}'/f'{s}_{p:03d}_{n}.json'
def identity(row):return(row['snr_db'],row['candidate_id'],row['noise_seed'])


@lru_cache(maxsize=16)
def sealed_document(path,digest):
    """Authenticate immutable stage metadata once per process; assets separately."""
    return h.checked(dict(path=path,sha256=digest))


def document(descriptor):return sealed_document(descriptor['path'],descriptor['sha256'])


def prior_checkpoint(request,done,kind,i):
    p=str(Path(request['out'])/kind/f'{i:04d}.json')
    h.require(done['outputs'].get(p)==h.sha(p),'Completed prior checkpoint changed');return h.read(p)


def cpu_worker(path,index):
    r,rh=registered(path);out=Path(r['out']);began=time.monotonic()
    h.require(0<=index<r['workers']and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU worker')
    for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
    with h.lock(out/f'cpu_worker_{index}.lock'):
        meter=open_ledger(r,rh);rt=phy.create_runtime(r['phy_qualification']['path']);completed=[]
        pilot=h.checked(r['pilot_execution_request'])if r['pilot_execution_request']else None
        pd=h.checked(r['pilot_completion'])if pilot else None
        try:
            for rec in r['records'][index::r['workers']]:
                h.guard(r,began);i=rec['source_index'];cp=out/'cpu_sources'/f'{i:04d}.json';count=len(r['schedule'])*len(r['noise_seeds'])
                if cp.exists():
                    d=h.read(cp);h.require(d['request_sha256']==rh and len(d['frames'])==count,'Prior CPU source differs');completed.append(h.desc(cp));continue
                tokens,_=h.source_assets(rec);scales=[tokens[OFFSETS[j]:OFFSETS[j+1]]for j in range(10)];old={}
                if pilot and i<100:old={identity(x):x for x in prior_checkpoint(pilot,pd,'cpu_sources',i)['frames']}
                frames=[]
                for point in r['schedule']:
                    p=point['profile'];snr=point['snr_db'];pid=p['profile_id']
                    h.require(p==rt.catalogue.entry(pid),'Schedule differs from qualified public catalogue')
                    for seed in r['noise_seeds']:
                        h.guard(r,began);fp=frame_path(out,i,snr,pid,seed)
                        if fp.exists():
                            row=h.read(fp);h.require(row['request_sha256']==rh,'Saved frame differs');frames.append(row);continue
                        ctr=counter(i,snr,seed);wave,tx=rt.transmit(pid,scales,ctr)
                        z=phy.standard_noise(rec['source_id'],seed);observed=wave+z/math.sqrt(10**(snr/10))
                        expected=dict(source_index=i,source_id=rec['source_id'],preprocessing_id=rec['preprocessing_id'],snr_db=snr,noise_seed=seed,
                            candidate_id=p['candidate_id'],profile_id=pid,wire_key=p['wire_key'],noise_sha256=h.image_sha(z),
                            waveform_sha256=h.image_sha(wave),received_sha256=h.image_sha(observed),public_frame_counter=ctr,catalogue_sha256=rt.catalogue.digest)
                        prior=old.get((snr,p['candidate_id'],seed))
                        if prior:
                            h.require(all(prior[k]==v for k,v in expected.items())and prior['transmission']==tx,'Exact N2048 pilot received observation differs')
                            actual=rt.original.present_actual(prior['header'],prior['body'],rt.catalogue)
                            h.require(actual['receiver_state']==prior['receiver_state'],'Pilot actual hard state differs')
                            row=dict(prior,RX_reuse='EXACT_N2048_PILOT',new_packet_calls=0)
                        else:
                            event=f'{SCHEMA}/{rh[:16]}/source{i:04d}/snr{snr}/profile{pid}/noise{seed}'
                            row=rt.receive(observed,float(snr),ctr,meter,event,'calibration')
                            row.update(RX_reuse='NEW_N2048_ACTUAL_RX',new_packet_calls=row['logical_packet_calls'])
                        row.update(expected,request_sha256=rh,transmission=tx,image_reconstruction_complete=False,visual_quality_scored=False)
                        h.save(fp,row);frames.append(row)
                h.save(cp,dict(status='T6_RAW_CPU_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],frames=frames,
                    unique_states=len({h.state_key(x)for x in frames}),reused_frames=sum(x['new_packet_calls']==0 for x in frames)))
                completed.append(h.desc(cp));print(h.canonical(dict(stage='T6_RAW_CPU',worker=index,source=i,source_count=r['source_count'])),flush=True)
            h.save(out/'cpu_workers'/f'{index:02d}.json',dict(status='COMPLETE',request_sha256=rh,worker=index,sources=completed))
        finally:meter.close()


def cache_records(r,rec):
    """Metadata-only source/state lookup. RGB bytes are verified on actual use."""
    result={};i=rec['source_index']
    for row in h.old_rows(rec):
        key=h.state_key(row);p=row['image_archive']
        item=dict(state=row['receiver_state'],image_path=p,image_key='images',image_slot=row['image_slot'],
            image_file_sha256=rec['old_image_archives'][p],image_sha256=row['image_sha256'],metrics={m:row[m]for m in h.METRICS},reuse='ORIGINAL_RAW_EXACT_STATE')
        if key in result:h.require(result[key]['image_sha256']==item['image_sha256']and result[key]['metrics']==item['metrics'],'Conflicting original exact state scores')
        else:result[key]=item
    env=document(r['environment_request']);done=document(r['t2_completion'])
    stages=[(env,done,'T2_FULL_EXACT_STATE')]
    if r['pilot_execution_request']and i<100:stages.append((document(r['pilot_execution_request']),document(r['pilot_completion']),'N2048_PILOT_EXACT_STATE'))
    for previous,completion,label in stages:
        cpu=prior_checkpoint(previous,completion,'cpu_sources',i);states={h.state_key(x):x['receiver_state']for x in cpu['frames']}
        gpu=prior_checkpoint(previous,completion,'gpu_sources',i)
        h.require(cpu['source_id']==gpu['source_id']==rec['source_id'],'Same-source cache required')
        for row in gpu['rows']:
            key=row['received_state_sha256'];h.require(key in states,'Saved score lacks actual received-state proof')
            if key in result:
                h.require(result[key]['image_sha256']==row['image_sha256']and all(result[key]['metrics'][m]==row[m]for m in h.METRICS),'Exact-state cache conflict');continue
            p=row['image_path'];sp=Path(previous['out'])/'gpu_states'/f'{i:04d}'/(key+'.json')
            if not sp.exists()and label=='T2_FULL_EXACT_STATE'and i<100:
                pp=document(previous['parent_pilot_request']);pd=document(previous['parent_pilot_completion'])
                pg=prior_checkpoint(pp,pd,'gpu_sources',i)
                matching=[x for x in pg['rows']if x['received_state_sha256']==key]
                h.require(matching and all(x['image_sha256']==row['image_sha256']and all(x[m]==row[m]for m in h.METRICS)for x in matching),'T2 parent pilot scores differ')
                sp=Path(pp['out'])/'gpu_states'/f'{i:04d}'/(key+'.json')
            if sp.exists():
                v=h.read(sp);h.require(v['state']==states[key]and v['image_sha256']==row['image_sha256']
                    and all(v['metrics'][m]==row[m]for m in h.METRICS),'Saved image/state/score proof differs')
                filehash=v['image_file_sha256'];imagekey=v.get('image_key','image');slot=v['image_slot']
            else:
                # A prior stage can itself refer to an older image. Its sealed
                # metadata must carry the file identity, never guess by source.
                h.require('image_file_sha256'in row and'image_key'in row,'Missing exact inherited cache provenance')
                filehash=row['image_file_sha256'];imagekey=row['image_key'];slot=row['image_slot']
            result[key]=dict(state=states[key],image_path=p,image_key=imagekey,image_slot=slot,image_file_sha256=filehash,
                image_sha256=row['image_sha256'],metrics={m:row[m]for m in h.METRICS},reuse=label)
    if r.get('t1_calibration_cache'):
        add_t1_cache(r,rec,result)
    return result


def add_t1_cache(r,rec,result):
    """Reuse only sealed same-source actual decoded-token states and DINO-L scores."""
    item=r['t1_calibration_cache'];done=document(item['completion']);pr=document(item['request']);i=rec['source_index']
    cp=prior_checkpoint(pr,done,'gpu_sources',i);h.require(cp['source_id']==rec['source_id'],'T1 cache source differs');states={}
    for descriptor in cp['source_decode_checkpoints']:
        h.require(done['outputs'].get(descriptor['path'])==descriptor['sha256'],'Actual T1 source-decode proof must be sealed')
        proof=h.checked(descriptor);h.require(proof['request_sha256']==item['request']['sha256'],'T1 recovery request differs')
        state=proof['state'];states[h.digest(state)]=state
    for row in cp['rows']:
        key=row['received_state_sha256'];h.require(key in states and row['source_id']==rec['source_id'],'T1 row lacks same-source received-token proof')
        if key in result:
            h.require(result[key]['image_sha256']==row['image_sha256']and all(result[key]['metrics'][m]==row[m]for m in h.METRICS),'T1 exact-state image/metric conflict');continue
        ip=Path(row['image_path']);sp=ip.with_suffix('.json');v=h.read(sp)
        h.require(v['state']==states[key]and v['image_path']==str(ip)and v['image_sha256']==row['image_sha256']
            and all(v['metrics'][m]==row[m]for m in h.METRICS),'T1 image/actual token/metric binding differs')
        result[key]=dict(state=states[key],image_path=str(ip),image_key=v.get('image_key','image'),image_slot=row['image_slot'],
            image_file_sha256=v['image_file_sha256'],image_sha256=row['image_sha256'],metrics={m:row[m]for m in h.METRICS},reuse='T1_CALIBRATION_EXACT_SOURCE_STATE_SCORE')


def plan_gpu(path):
    r,rh=registered(path);out=Path(r['out']);pins={};totals=dict(unique_states=0,cache_states=0,new_VAR_renders=0,new_metric_images=0,logical_frames=0)
    for rec in r['records']:
        i=rec['source_index'];cp=out/'cpu_sources'/f'{i:04d}.json';cpu=h.read(cp)
        h.require(cpu['request_sha256']==rh and len(cpu['frames'])==len(r['schedule'])*len(r['noise_seeds']),'Complete CPU source required')
        cache=cache_records(r,rec);states={h.state_key(x):x['receiver_state']for x in cpu['frames']};hits={k:cache[k]for k in states if k in cache}
        value=dict(request_sha256=rh,source_id=rec['source_id'],cpu_source=h.desc(cp),states=states,cache=hits,
            new_VAR_renders=sum(s['kind']!='gray'for k,s in states.items()if k not in hits),new_metric_images=len(states)-len(hits))
        sp=out/'gpu_plan_sources'/f'{i:04d}.json';h.save(sp,value);pins[str(sp)]=h.sha(sp)
        totals['unique_states']+=len(states);totals['cache_states']+=len(hits);totals['new_VAR_renders']+=value['new_VAR_renders'];totals['new_metric_images']+=value['new_metric_images'];totals['logical_frames']+=len(cpu['frames'])
    from t1_calibrate import ledger_snapshot
    ledger=ledger_snapshot(out/'packet_ledger.sqlite',rh,r['packet_cap']);h.require(totals['logical_frames']==r['frame_count'],'Exact full CPU grid')
    value=dict(status='T6_RAW_EXACT_POST_CPU_GPU_BUDGET_REGISTERED',request=h.desc(path),request_sha256=rh,
        **totals,new_VAR_cap=totals['new_VAR_renders']+r['gpu_workers'],new_metric_cap=totals['new_metric_images']+r['gpu_workers'],
        reference_preparations=r['source_count']+r['gpu_workers'],qualification_VAR_calls=r['gpu_workers'],
        packet_ledger=ledger,outputs=pins,source_only_cache_reuse=False,old_RX_relabelled=False)
    h.save(out/'gpu_plan.json',value);return {k:v for k,v in value.items()if k!='outputs'}


def checked_image(v,archives):
    p=v['image_path']
    if p not in archives:
        h.require(h.sha(p)==v['image_file_sha256'],'Cached image file changed')
        with np.load(p,allow_pickle=False)as z:archives[p]={k:z[k].copy()for k in z.files}
    a=archives[p][v['image_key']];image=a if v['image_key']=='image'else a[v['image_slot']]
    h.require(image.shape==(3,256,256)and image.dtype==np.float32 and h.image_sha(image)==v['image_sha256'],'Exact float32 cached image changed')
    return image


def prepare_reference_once(out,r,rh,scorer,record,pixels,index):
    """Persist exact original scorer reference features; no retry recomputation."""
    base=Path(out)/'metric_references'/f'{index:04d}';cp=base.with_suffix('.json');ap=base.with_suffix('.pt');reserved=base.with_suffix('.reserved.json')
    wanted=(record['image_id'],record['preprocessing_id'],index);target=pixels.astype(np.float32)/np.float32(255)
    if cp.exists():
        meta=h.read(cp);h.require(meta['request_sha256']==rh and meta['record_identity']==list(wanted)
            and meta['metric_identity']==r['old_score_identity']and h.sha(ap)==meta['archive_sha256'],'Cached reference feature identity differs')
        # This is our own SHA-sealed feature archive, never an external model or
        # arbitrary user pickle. Preserve original tensor devices exactly.
        value=scorer.torch.load(ap,weights_only=False)
        h.require(tuple(value['record_identity'])==wanted and np.array_equal(value['target'],target),'Cached reference pixels differ')
        scorer.target=value['target'];scorer.reference=value['reference'];scorer.prepared=value['prepared'];scorer.record_identity=wanted
        return
    h.require(not reserved.exists(),'Unresolved reference model preparation; no automatic repeated call')
    h.save(reserved,dict(request_sha256=rh,record_identity=list(wanted),new_reference_preparations=1))
    scorer.prepare_source(record,target,index)
    h.require(tuple(scorer.record_identity)==wanted,'Original metric source identity differs')
    with ap.open('xb')as handle:
        scorer.torch.save(dict(target=scorer.target,reference=scorer.reference,prepared=scorer.prepared,record_identity=scorer.record_identity),handle)
        handle.flush();os.fsync(handle.fileno())
    h.save(cp,dict(request_sha256=rh,record_identity=list(wanted),metric_identity=r['old_score_identity'],archive_sha256=h.sha(ap),new_reference_preparations=1))


def gpu_worker(path,cohort_config=None):
    r,rh=registered(path);out=Path(r['out']);began=time.monotonic();plan=h.read(out/'gpu_plan.json')
    h.require(plan['request_sha256']==rh and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Sealed actual CPU state budget and explicit GPU0 required')
    parallel=None
    if cohort_config is None:
        h.require(r['gpu_workers']==1,'Two-worker run requires owned cohort');shared=h.lock(r['visual_config']['visual_lock']);own=h.lock(out/'gpu.lock');records=r['records']
    else:
        import t6_raw_gpu_parallel as parallel
        cohort,probe=parallel.admit(r,rh,cohort_config);shared=contextlib.nullcontext();own=h.lock(out/f'gpu_worker_{cohort_config["worker_id"]}.lock')
        records=[r['records'][i]for i in cohort_config['source_indices']]
    for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
    with shared,own:
        native,source,scorer=h.gpu_build(r,out)
        if parallel:
            h.require(native.assets.old.b is probe,'Owned native admission differs');parallel.qualify(r,rh,native,source,scorer,cohort_config,cohort)
        else:h.qualify_gpu(r,rh,out,native,source,scorer)
        for rec in records:
            h.guard(r,began)
            if parallel:cohort.require_available()
            i=rec['source_index'];cp=out/'gpu_sources'/f'{i:04d}.json'
            if cp.exists():h.require(h.read(cp)['request_sha256']==rh,'Prior GPU source differs');continue
            sp=out/'gpu_plan_sources'/f'{i:04d}.json';h.require(plan['outputs'][str(sp)]==h.sha(sp),'Post-CPU exact state plan changed');pp=h.read(sp);cpu=h.checked(pp['cpu_source'])
            h.require(pp['request_sha256']==rh and pp['source_id']==rec['source_id'],'State plan source differs')
            cache=dict(pp['cache']);archives={}
            for v in cache.values():checked_image(v,archives)
            _,pixels=h.source_assets(rec);sr=h.score_record(rec);prepare_reference_once(out,r,rh,scorer,sr,pixels,i)
            for key,state in pp['states'].items():
                h.guard(r,began)
                if key in cache:continue
                jp=out/'gpu_states'/f'{i:04d}'/(key+'.json');ip=jp.with_suffix('.npz');res=jp.with_suffix('.reserved.json')
                if jp.exists():
                    v=h.read(jp);h.require(v['request_sha256']==rh and v['state']==state,'Saved state differs');checked_image(v,{})
                else:
                    h.require(not res.exists(),'Unresolved state render must be audited, never automatically repeated')
                    h.save(res,dict(request_sha256=rh,state=state,new_VAR_calls=int(state['kind']!='gray'),new_metric_images=1))
                    image=h.render_state(native,source,state);metrics=scorer(sr,[image])[0]
                    with ip.open('xb')as f:np.savez(f,image=image)
                    v=dict(request_sha256=rh,state=state,image_path=str(ip),image_key='image',image_slot=0,image_file_sha256=h.sha(ip),image_sha256=h.image_sha(image),
                        metrics={m:metrics[m]for m in h.METRICS},reuse='NEW_N2048_RECEIVED_STATE',new_VAR_calls=int(state['kind']!='gray'))
                    h.save(jp,v)
                cache[key]=v
            rows=[]
            for frame in cpu['frames']:
                key=h.state_key(frame);v=cache[key];p=next(x['profile']for x in r['schedule']if x['snr_db']==frame['snr_db']and x['candidate_id']==frame['candidate_id']);tx=frame['transmission']
                rows.append(dict(source_index=i,source_id=rec['source_id'],preprocessing_id=rec['preprocessing_id'],snr_db=frame['snr_db'],noise_seed=frame['noise_seed'],
                    candidate_id=frame['candidate_id'],profile_id=p['profile_id'],N=2048,m=p['m'],K=p['K'],eligible_WHOLE=p['K']==0,eligible_PARTIAL=True,
                    status='T6_RAW_ACTUAL_CALIBRATION_FRAME_COMPLETE',header_ok=frame['header_ok'],body_crc_accept=frame['body_crc_accept'],gray=frame['gray'],
                    E_frame=tx['E_frame'],rho=tx['rho'],header_symbols=68,body_symbols=tx['body_symbols'],padding_symbols=tx['padding_symbols'],
                    source_bits=tx['source_bits'],k=tx['k'],n=tx['n'],q=tx['q'],actual_code_rate=tx['actual_code_rate'],
                    received_state_sha256=key,RX_reuse=frame['RX_reuse'],visual_reuse=v['reuse'],image_path=v['image_path'],image_key=v['image_key'],image_slot=v['image_slot'],
                    image_sha256=v['image_sha256'],image_file_sha256=v['image_file_sha256'],**v['metrics']))
            h.save(cp,dict(status='T6_RAW_GPU_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],rows=rows,cpu_source=pp['cpu_source'],gpu_plan_source=h.desc(sp)))
            print(h.canonical(dict(stage='T6_RAW_GPU',source=i,source_count=r['source_count'],new_VAR_budget=pp['new_VAR_renders'])),flush=True)
        native.frozen()
        if parallel:cohort.require_available()


def aggregate(r,rows):
    rankings=[];winners=[]
    for snr in SNRS:
        group=[]
        for point in[x for x in r['schedule']if x['snr_db']==snr]:
            cid=point['candidate_id'];sub=[x for x in rows if x['snr_db']==snr and x['candidate_id']==cid];p=point['profile'];by={i:[]for i in range(r['source_count'])}
            for x in sub:by[x['source_index']].append(x)
            h.require(all(len(v)==len(r['noise_seeds'])for v in by.values()),'Every source has all registered noises')
            means={m:math.fsum(math.fsum(float(x[m])for x in by[i])/len(r['noise_seeds'])for i in range(r['source_count']))/r['source_count']for m in h.METRICS}
            group.append(dict(snr_db=snr,candidate_id=cid,profile_id=p['profile_id'],m=p['m'],K=p['K'],source_count=r['source_count'],noise_count=len(r['noise_seeds']),
                body_crc_rejected_fraction=sum(x['body_crc_accept']is False for x in sub)/len(sub),header_rejected_fraction=sum(not x['header_ok']for x in sub)/len(sub),**means))
        order=sorted(group,key=lambda x:(-x['dinov2_vitl14_cosine'],x['candidate_id']));whole=[x for x in order if x['K']==0]
        for x in group:x.update(partial_rank=1+order.index(x),whole_rank=1+whole.index(x)if x in whole else'')
        rankings.extend(group)
        for family,chosen in [('RAW_WHOLE',whole[0]),('RAW_PARTIAL',order[0])]:winners.append(dict(family=family,**chosen))
    return rankings,winners


def close(path,receipt):
    r,rh=registered(path);out=Path(r['out']);owner=h.read(receipt)
    h.require(owner['request_sha256']==rh and owner['actual_children_waited']is True and owner['worker_exit_codes']==[0]*r['gpu_workers'],'Actual owner wait/exit receipt required')
    rows=[];outputs={str(Path(receipt).resolve()):h.sha(receipt)};actual_packets=0
    for rec in r['records']:
        p=out/'gpu_sources'/f'{rec["source_index"]:04d}.json';cp=h.read(p)
        h.require(cp['request_sha256']==rh and cp['source_id']==rec['source_id'],'Complete source identity required');outputs[str(p)]=h.sha(p)
        cpu=h.checked(cp['cpu_source']);outputs[cp['cpu_source']['path']]=cp['cpu_source']['sha256'];rows.extend(cp['rows'])
        actual_packets+=sum(x['new_packet_calls']for x in cpu['frames'])
    expected={(i,p['snr_db'],p['candidate_id'],seed)for i in range(r['source_count'])for p in r['schedule']for seed in r['noise_seeds']}
    h.require(len(rows)==r['frame_count']and{(x['source_index'],x['snr_db'],x['candidate_id'],x['noise_seed'])for x in rows}==expected,'Complete fixed action/source/noise grid')
    rankings,winners=aggregate(r,rows)
    for name,values in [('per_frame.csv',rows),('calibration_rankings.csv',rankings)]:write_csv(out/name,values);outputs[str(out/name)]=h.sha(out/name)
    from t1_calibrate import ledger_snapshot
    snap=ledger_snapshot(out/'packet_ledger.sqlite',rh,r['packet_cap']);h.require(snap['total']==actual_packets,'Actual packet ledger and completed frame receipts differ')
    gp=out/'gpu_plan.json';outputs[str(gp)]=h.sha(gp)
    policy=dict(status='T6_RAW_CALIBRATION_CANDIDATE_WINNERS',phase=r['phase'],selection_used_confirmation=False,holdout_used_for_selection=False,winners=winners)
    h.save(out/'raw_policy.json',policy);outputs[str(out/'raw_policy.json')]=h.sha(out/'raw_policy.json')
    done=dict(status='T6_RAW_PILOT_CALIBRATION_COMPLETE'if r['phase']=='pilot'else'T6_RAW_FULL1000_CALIBRATION_COMPLETE',
        request=h.desc(path),request_sha256=rh,source_count=r['source_count'],source_ids=r['source_ids'],noise_count=len(r['noise_seeds']),noise_seeds=r['noise_seeds'],frame_count=len(rows),
        rankings_path=str(out/'calibration_rankings.csv'),policy=h.desc(out/'raw_policy.json'),packet_ledger=snap,actual_children_waited=True,
        worker_exit_codes=owner['worker_exit_codes'],selection_used_confirmation=False,outputs=outputs)
    h.save(out/'completion.json',done);return {k:v for k,v in done.items()if k not in('outputs','source_ids')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);q=s.add_parser('prepare-execution')
    for x in('plan-request','t2-completion'):q.add_argument('--'+x,required=True)
    q.add_argument('--t1-completion',help='Optional completed original1000 entropy calibration, exact renderer/state/score reuse only')
    q.add_argument('--gpu-workers',type=int,choices=(1,2),default=2)
    for cmd in('ledger-init','cpu-worker','plan-gpu','gpu-worker','close'):
        q=s.add_parser(cmd);q.add_argument('--request',required=True)
        if cmd=='cpu-worker':q.add_argument('--index',type=int,required=True)
        if cmd=='close':q.add_argument('--owner-receipt',required=True)
    a=p.parse_args();f={'ledger-init':ledger_init,'plan-gpu':plan_gpu,'gpu-worker':gpu_worker}
    value=prepare(a)if a.command=='prepare-execution'else cpu_worker(a.request,a.index)if a.command=='cpu-worker'else close(a.request,a.owner_receipt)if a.command=='close'else f[a.command](a.request)
    if value is not None:print(h.canonical(value),flush=True)
