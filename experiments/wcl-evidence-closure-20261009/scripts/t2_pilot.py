"""Independent, finite whole-action calibration audit; no original ledger writes.

prepare is metadata-only. cpu-worker uses original paid RAW PHY. gpu-worker uses
the original renderer and metric scorer. No test/holdout image enters this file.
"""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, importlib.util, json, math, os
from pathlib import Path
import shutil, signal, sqlite3, subprocess, sys, time

SCHEMA='WCL_T2_WHOLE_PILOT_V1'
SNRS=[10,19]
SEED=4101
COUNT=100
CAP=42400
STOP=False
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine']
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def desc(p):return dict(path=str(Path(p).absolute()),sha256=sha(p))
def checked(d):
    require(sha(d['path'])==d['sha256'],'Changed input: '+d['path']);return read(d['path'])
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=(canonical(v)+'\n').encode()
    if p.exists():require(p.read_bytes()==data,'Existing checkpoint changed: '+str(p));return
    tmp=p.with_name(p.name+'.pending')
    with tmp.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def load(p,name,bound):
    require(bound.get(str(p))==sha(p),'Unbound executable: '+str(p))
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def stop(*_):
    global STOP;STOP=True
@contextlib.contextmanager
def lock(p):
    import fcntl
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def guard(r,began):
    require(not STOP and time.time()<r['deadline_unix'] and time.monotonic()-began<r['max_seconds'], 'Stopped/deadline; completed checkpoints preserved')
    require(not any(Path(p).exists()for p in r['stop_files']),'Explicit STOP file')
def validate_request(r):
    require(r['schema']==SCHEMA and r['snrs']==SNRS and r['noise_seed']==SEED and r['source_count']==COUNT,'Fixed first100/10,19/4101 scope')
    require(len(r['profiles'])==106 and len({p['wire_key']for p in r['profiles']})==106 and all(p['K']==0 and p['N']==1024 for p in r['profiles']),'All106 unique whole actions')
    require(sum('full_budget'in p['allocation_modes']for p in r['profiles'])==7,'All seven whole full-budget actions retained')
    require(len(r['records'])==100 and [x['source_index']for x in r['records']]==list(range(100)),'Original first100 order')
    require(r['packet_cap']==CAP and r['frame_count']==21200,'Independent finite cap')
    require(type(r['workers'])is int and 1<=r['workers']<=16,'Explicit bounded CPU workers')

def prepare(a):
    vr=read(a.visual_request);cc=read(vr['cpu_batch']['config'])
    cat=read(vr['catalogue']);manifest=read(cc['asset_manifest']);old=read(Path(vr['out'])/'completion.json')
    require(old['status']=='MAIN_RAW64_ACTUAL_VISUAL_COMPLETE_V1' or (old.get('images_scored') and old.get('source_count')==1000),'Original complete calibration visual receipt')
    require(len(cat['profiles'])==433 and len(manifest['records'])==1000,'Frozen433 and originalcal1000')
    require(manifest['source_ids']==[x['source_id']for x in manifest['records']],'Source manifest ordering')
    require(sha(vr['catalogue'])=='af772451d33a5750d77a57ba7f296b851cdff2baf098cd96896141818b3d2f03','Original catalogue file bytes')
    records=[]
    for rec in manifest['records'][:COUNT]:
        cp=read(rec['checkpoint']);archive=rec['archive']
        require(cp['source_id']==rec['source_id'] and cp['source_index']==rec['source_index'] and cp['tokens_sha256']==rec['tokens_sha256'],'Original token checkpoint identity')
        require(sha(rec['checkpoint'])==rec['checkpoint_sha256'],'Original source checkpoint pin')
        i=rec['source_index'];oldcp=str(Path(vr['out'])/'source_checkpoints'/f'{i:04d}.json')
        oldrows=str(Path(vr['out'])/'sources'/f'{i:04d}.json')
        require(old['outputs'].get(oldcp)==sha(oldcp),'Original visual source checkpoint seal')
        oc=read(oldcp);require(oc['source_id']==rec['source_id'] and oc['frame_count']==90 and oc['images_scored'],'Original scored90-frame source')
        require(oc['outputs'].get(oldrows)==old['outputs'].get(oldrows),'Old source rows binding')
        records.append(dict(rec,archive_sha256=cp['outputs'][archive],class_index=cp['evaluation_class_index'],
            old_visual_checkpoint=desc(oldcp),old_visual_rows=dict(path=oldrows,sha256=oc['outputs'][oldrows]),
            old_image_archives={p:h for p,h in oc['outputs'].items()if p.endswith('.npz')}))
    sources=dict(cc['source_bindings']);sources.update(vr['source_bindings'])
    for p,h in sources.items():require(sha(p)==h,'Frozen original source changed: '+p)
    closure=load(vr['static_closure_module'],'_t2_closure',sources)
    native=closure.collect_bindings(vr['root'],vr['native_runtime'],vr['var_source'],vr['dino_source'],vr['uep_runtime'])
    for p,h in vr['native_source_bindings'].items():require(native.get(p)==h,'Original native source changed or removed: '+p)
    # Only explicit additional tracked source files may extend the old graph.
    sources.update(native);sources[str(Path(__file__).absolute())]=sha(__file__);sources[str(Path(__file__).with_name('t2_ledger.py').absolute())]=sha(Path(__file__).with_name('t2_ledger.py'))
    input_pins={p:h for p,h in cc['input_bindings'].items() if p in [cc['original_cpu_plan'],vr['catalogue'],cc['adapter_config']['legacy_qualification']]}
    proxy_path=str(Path(vr['shortlist']).with_name('proxy_scores.json'))
    for p in [vr['legacy_profiles'],vr['legacy_aliases'],vr['metric_batch_qualification'],cc['asset_manifest'],proxy_path]:input_pins[p]=sha(p)
    root=Path(vr['root']);out=Path(a.out).absolute()
    require(out.is_relative_to(root/'outputs') and 'WCL' in str(out).upper(),'New outputs WCL namespace required')
    r=dict(schema=SCHEMA,original_visual_request=desc(a.visual_request),original_cpu_config=desc(vr['cpu_batch']['config']),
        original_visual_completion=desc(Path(vr['out'])/'completion.json'),original_source_manifest=desc(cc['asset_manifest']),
        scientific_snapshot='14b09ecd72984fb39c683d1a62bcd3221c69142f',execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        root=str(root),out=str(out),python_cpu=cc['python'],python_gpu=vr['python'],source_count=COUNT,
        source_rule='first100 in original calibration manifest, no quality selection',records=records,
        profiles=[p for p in cat['profiles']if p['K']==0],snrs=SNRS,noise_seed=SEED,N=1024,
        frame_count=21200,packet_cap=CAP,workers=a.workers,threads=2,
        cpu_config=cc,visual_config={k:v for k,v in vr.items()if k not in ['source_bindings','input_bindings','native_source_bindings','cpu_batch']},
        source_bindings=sources,input_bindings=input_pins,native_source_bindings=native,
        original_proxy_scores=desc(proxy_path),
        old_visual_identity=old['frozen_visual_identity'],old_numerical_runtime=old['numerical_runtime'],old_score_identity=old['score_identity'],
        checkpoint_module=vr['checkpoint_module'],deadline_unix=a.deadline_unix,max_seconds=a.max_seconds,
        stop_files=[str(root/'STOP'),str(out/'STOP')],original_ledgers_opened=False,policy_selection=False,
        noise_rule='Original main_raw64_calibration_checkpoints.standard_noise(source_id,4101)',
        counter_rule='Original six-SNR order, 1000-source counter: (SNRindex*1000+source_index)*3',
        reuse_rule='Exact old RX metadata plus regenerated TX/noise hashes; rendered state+frozen model+metric identity',
        GPU_upper_bound_new_renders=21200,GPU_expected_new_renders='Determined from actual RX state deduplication after CPU; no optimistic BLER substitution',
        qualification=dict(new_VAR_calls=1,new_metric_images=1,tolerances=dict(psnr_db=1e-4,lpips_alex=1e-5,dinov2_vitl14_cosine=1e-5)))
    validate_request(r);save(a.request,r)
    return dict(status='PREPARED_METADATA_ONLY',request=desc(a.request),whole_actions=106,frame_count=21200,packet_cap=CAP,new_scientific_calls=0)

from t2_ledger import Ledger

def registered(path):
    r=read(path);validate_request(r);require(r['source_bindings'][str(Path(__file__).absolute())]==sha(__file__),'Runner changed after preparation')
    lp=str(Path(__file__).with_name('t2_ledger.py').absolute());require(r['source_bindings'][lp]==sha(lp),'Ledger changed after preparation')
    return r,sha(path)
def add_paths(r):
    for p in reversed(r['visual_config']['pythonpath']):sys.path.insert(0,p)
def cpu_runtime(r,ledger):
    add_paths(r);cc=r['cpu_config'];b=dict(r['source_bindings'],**r['input_bindings'])
    load(cc['adapter_config']['plan_module'],'main_raw64_plan_only',b)
    packet=load(cc['adapter_module'],'main_raw64_packet_adapter',b)
    cfg=dict(cc,plan=cc['original_cpu_plan'],source_bindings=r['source_bindings'],input_bindings=r['input_bindings'])
    adapter=packet.create(cfg)
    wire=load(cc['original_receiver_module'],'_t2_original_wire',b)
    rx=load(cc['receiver_module'],'_t2_mixed_wire',b)
    catalogue=rx.MixedPublicCatalogue(read(cc['adapter_config']['catalogue']),read(r['visual_config']['legacy_profiles']),read(r['visual_config']['legacy_aliases']),legacy_backend_identity=adapter.backend.identity,original_receiver=wire)
    receiver=rx.Receiver(catalogue,backend=adapter.backend,legacy_phy=adapter.legacy,header=adapter.header,charge=ledger.call)
    core=load(r['checkpoint_module'],'_t2_original_noise',b)
    return adapter,wire,rx,catalogue,receiver,core
def source_assets(rec):
    import numpy as np
    require(sha(rec['archive'])==rec['archive_sha256'],'Original source NPZ changed')
    with np.load(rec['archive'],allow_pickle=False)as z:tokens=z['tokens'].copy();pixels=z['pixels'].copy()
    require(tokens.shape==(680,) and tokens.dtype==np.int64 and ((tokens>=0)&(tokens<4096)).all(),'Original680 tokens')
    require(hashlib.sha256(b'int64:680\0'+tokens.astype('<i8',copy=False).tobytes()).hexdigest()==rec['tokens_sha256'],'Original token content SHA')
    require(pixels.shape==(3,256,256) and pixels.dtype==np.uint8 and hashlib.sha256(pixels.tobytes()).hexdigest()==rec['preprocessing_id'],'Original pixels SHA')
    return tokens,pixels
def state_key(row):return digest(row['receiver_state'])
def old_rows(rec):
    cp=checked(rec['old_visual_checkpoint']);rows=checked(rec['old_visual_rows'])
    require(cp['outputs'][rec['old_visual_rows']['path']]==rec['old_visual_rows']['sha256'],'Original source closure changed')
    require(len(rows)==90 and all(x['source_id']==rec['source_id'] and x['visual_quality_scored'] and x['primary_metric']=='dinov2_vitl14_cosine'for x in rows),'Old complete90-frame quality rows')
    return rows
def cpu_check(path):
    r,rh=registered(path);out=Path(r['out']);out.mkdir(parents=True,exist_ok=True)
    class NoDecode:
        def call(self,*_):raise RuntimeError('CPU construction check must never decode')
    with lock(out/'cpu_check.lock'):
        a,wire,rx,cat,receiver,core=cpu_runtime(r,NoDecode())
        for p in r['profiles']:
            g=p['groups'][0];q={'QPSK':2,'16QAM':4,'64QAM':6}[g['modulation']]
            require(a.backend.plan(g['information_bits'],g['transmitted_bits'],q)==g['layout'],'Frozen legal layout changed')
        token,pixels=source_assets(r['records'][0]);old_rows(r['records'][0])
        done=dict(status='CPU_RUNTIME_ALL106_LAYOUTS_VALIDATED_NO_DECODE',request_sha256=rh,whole_candidates=106,
            source0_tokens_shape=list(token.shape),source0_pixels_shape=list(pixels.shape),new_packet_calls=0,GPU_used=False)
        save(out/'cpu_check.json',done);return done
def cpu_worker(path,index):
    import numpy as np
    r,rh=registered(path);out=Path(r['out']);out.mkdir(parents=True,exist_ok=True)
    require(0<=index<r['workers'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU worker scope')
    for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,stop)
    began=time.monotonic()
    with lock(out/f'cpu_worker_{index}.lock'):
        ledger=Ledger(out/'packet_ledger.sqlite',rh,CAP);a,wire,rx,cat,receiver,core=cpu_runtime(r,ledger)
        prefix=[0,1,5,14,30,55,91,155,255,424,680];completed=[]
        for rec in r['records'][index::r['workers']]:
            guard(r,began);i=rec['source_index'];cp=out/'cpu_sources'/f'{i:04d}.json'
            if cp.exists():
                d=read(cp);require(d['request_sha256']==rh and len(d['frames'])==212,'Completed CPU source differs');completed.append(desc(cp));continue
            tokens,_=source_assets(rec);scales=[tokens[prefix[j]:prefix[j+1]]for j in range(10)]
            prior=old_rows(rec);pm={(x['snr_db'],x['candidate_id'],x['noise_seed']):x for x in prior};frames=[]
            for snr in SNRS:
                for profile in r['profiles']:
                    guard(r,began);pid=profile['profile_id'];point=dict(snr_db=snr,candidate_id=profile['candidate_id'],profile_id=pid,wire_key=profile['wire_key'])
                    fp=out/'cpu_frames'/f'{i:04d}'/f'{snr}_{pid:03d}.json'
                    if fp.exists():
                        row=read(fp);require(row['t2_request_sha256']==rh and row['source_id']==rec['source_id'],'Existing CPU frame differs');frames.append(row);continue
                    payload=wire.serialize_raw(scales,profile);counter=core.frame_counter(i,snr,SEED)
                    wave,tx=rx.transmit_frame(cat,a.backend,a.legacy,a.header,pid,payload,counter)
                    expected,observed=core.prepare_frame(rec,point,SEED,payload,wave,tx)
                    old=pm.get((snr,profile['candidate_id'],SEED))
                    if old is not None:
                        require(all(old[k]==v for k,v in expected.items()),'Old RX does not match original transmitted observation')
                        require(old['effective_catalogue_digest']==cat.digest and old['effective_aliases_digest']==cat.aliases_digest,'Old effective receiver differs')
                        actual=wire.present_actual(old['header'],old['body'],cat)
                        require(actual['receiver_state']==old['receiver_state'],'Old hard receiver state inconsistent')
                        row={k:v for k,v in old.items()if k not in METRICS and k not in ['visual_evidence','image_archive','image_slot']}
                        row.update(t2_reuse='ORIGINAL_EXACT_RX',new_packet_calls=0,original_rows=rec['old_visual_rows'])
                    else:
                        event=f'{SCHEMA}/source{i:04d}/snr{snr}/profile{pid}/noise{SEED}'
                        row=receiver.receive(observed,float(snr),counter,phase='actual_calibration',event_prefix=event)
                        row.update(expected,t2_reuse='NEW_ACTUAL_RX',new_packet_calls=row['logical_packet_calls'])
                    row.update(t2_request_sha256=rh,population_role='calibration',new_quality_scored=False,
                        image_reconstruction_complete=False,quality_scored=False,visual_quality_scored=False)
                    save(fp,row);frames.append(row)
            old_states={state_key(x)for x in prior};states={state_key(x)for x in frames}
            save(cp,dict(status='T2_CPU_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],source_index=i,
                frames=frames,reused_frames=sum(x['t2_reuse']=='ORIGINAL_EXACT_RX'for x in frames),
                unique_received_states=len(states),old_scored_state_matches=len(states&old_states),
                estimated_new_VAR_renders=sum(k not in old_states and not next(x for x in frames if state_key(x)==k)['gray']for k in states),
                source=rec));completed.append(desc(cp))
            print(canonical(dict(stage='cpu',worker=index,source=i,completed_sources=len(completed),new_state_keys=len(states-old_states))),flush=True)
        save(out/'cpu_workers'/f'{index:02d}.json',dict(status='COMPLETE',request_sha256=rh,worker_index=index,sources=completed,seconds=time.monotonic()-began))

def gpu_build(r,out):
    add_paths(r);v=r['visual_config'];b=r['source_bindings']
    quality=load(v['quality_driver_module'],'_t2_quality',b)
    source=load(v['source_driver_module'],'_t2_source',b)
    native=quality.build_native(Path(r['root']),v['native_runtime'],stop)
    require(native.loaded['identity']==r['old_visual_identity'] and native.flags==r['old_numerical_runtime'],'Frozen model/numerical identity differs')
    require(native.driver_bindings==r['native_source_bindings'],'Current native executable graph differs from registered old-plus-additions')
    qdir=out/'metric_qualification';qdir.mkdir(parents=True,exist_ok=True)
    target=qdir/'metric_batch8_qualification.json'
    if not target.exists():shutil.copyfile(v['metric_batch_qualification'],target)
    require(sha(target)==r['input_bindings'][v['metric_batch_qualification']],'Existing metric qualification changed')
    scorer=quality.MetricScorer(native,Path(r['root']),qdir)
    require(quality.q.jsonable(scorer.identity)==r['old_score_identity'],'Original metric identity differs; old scores cannot be mixed')
    return native,source,scorer
def render_state(native,source,state):
    import numpy as np
    if state['kind']=='gray':return np.full((3,256,256),.5,np.float32)
    flat=np.asarray([v for scale in state['prefix']for v in scale]+list(state['partial_values']),dtype=np.int64)
    with native.torch.no_grad():return source.render_received(native,flat,state['m'],state['K'])
def image_sha(image):
    import numpy as np
    image=np.ascontiguousarray(image)
    return hashlib.sha256(str(image.dtype).encode()+str(image.shape).encode()+image.tobytes()).hexdigest()
def score_record(rec):return dict(image_id=rec['source_id'],preprocessing_id=rec['preprocessing_id'],source_index=rec['source_index'],class_index=rec['class_index'])
def qualify_gpu(r,rh,out,native,source,scorer):
    import numpy as np
    p=out/'qualification_gpu.json'
    if p.exists():
        d=read(p);require(d['status']=='PASS' and d['request_sha256']==rh,'GPU qualification changed');return
    reservation=out/'qualification_gpu.reserved.json';require(not reservation.exists(),'Unresolved GPU qualification; do not repeat')
    rec=r['records'][0];old=next(x for x in old_rows(rec)if not x['gray']);_,pixels=source_assets(rec)
    archive=old['image_archive'];require(sha(archive)==rec['old_image_archives'][archive],'Original image archive changed')
    with np.load(archive,allow_pickle=False)as z:truth=z['images'][old['image_slot']].copy()
    save(reservation,dict(request_sha256=rh,new_VAR_calls=1,new_metric_images=1))
    image=render_state(native,source,old['receiver_state'])
    require(image_sha(image)==old['image_sha256'] and np.array_equal(image,truth),'Original renderer exact RGB replay failed')
    sr=score_record(rec);scorer.prepare_source(sr,pixels.astype(np.float32)/np.float32(255),0);value=scorer(sr,[image])[0]
    differences={m:abs(float(value[m])-float(old[m]))for m in METRICS}
    require(all(differences[m]<=r['qualification']['tolerances'][m]for m in METRICS),'Original metric replay tolerance failed')
    save(p,dict(status='PASS',request_sha256=rh,image_exact=True,image_sha256=image_sha(image),metric_differences=differences,new_VAR_calls=1,new_metric_images=1))

def gpu_worker(path,qualification_only=False):
    import numpy as np
    r,rh=registered(path);out=Path(r['out']);began=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit GPU0')
    for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,stop)
    with lock(r['visual_config']['visual_lock']),lock(out/'gpu.lock'):
        native,source,scorer=gpu_build(r,out);qualify_gpu(r,rh,out,native,source,scorer)
        if qualification_only:return dict(status='GPU_QUALIFICATION_COMPLETE',new_packets=0)
        for rec in r['records']:
            guard(r,began);i=rec['source_index'];cp=out/'gpu_sources'/f'{i:04d}.json'
            if cp.exists():require(read(cp)['request_sha256']==rh,'Existing GPU source differs');continue
            cpu_path=out/'cpu_sources'/f'{i:04d}.json';require(cpu_path.exists(),'CPU source incomplete; rerun GPU later to resume')
            cpu=read(cpu_path);require(cpu['request_sha256']==rh and len(cpu['frames'])==212,'Complete registered CPU source')
            frames=cpu['frames'];prior=old_rows(rec);cache={};archive_cache={};actual_new=0;old_reuse=0
            # Match receiver state exactly, independently of source truth/correctness.
            for old in prior:
                key=state_key(old)
                if key in cache:continue
                archive=old['image_archive']
                if archive not in archive_cache:
                    require(sha(archive)==rec['old_image_archives'][archive],'Old RGB archive changed')
                    with np.load(archive,allow_pickle=False)as z:archive_cache[archive]=z['images'].copy()
                image=archive_cache[archive][old['image_slot']]
                require(image_sha(image)==old['image_sha256'],'Old image slot changed')
                cache[key]=dict(image_sha256=old['image_sha256'],metrics={m:old[m]for m in METRICS},image_path=archive,image_slot=old['image_slot'],reuse='ORIGINAL_STATE_IMAGE_AND_SCORE')
            _,pixels=source_assets(rec);sr=score_record(rec);scorer.prepare_source(sr,pixels.astype(np.float32)/np.float32(255),i)
            rows=[]
            for frame in frames:
                guard(r,began);key=state_key(frame)
                if key not in cache:
                    sp=out/'gpu_states'/f'{i:04d}'/(key+'.json');imagepath=sp.with_suffix('.npz');reserved=sp.with_suffix('.reserved.json')
                    if sp.exists():
                        value=read(sp);require(value['request_sha256']==rh and value['state']==frame['receiver_state'] and sha(imagepath)==value['image_file_sha256'],'New state cache changed')
                    else:
                        require(not reserved.exists(),'Unresolved state render; do not automatically repeat')
                        save(reserved,dict(request_sha256=rh,state=frame['receiver_state'],source_id=rec['source_id']))
                        image=render_state(native,source,frame['receiver_state']);value=scorer(sr,[image])[0]
                        with imagepath.open('xb')as f:np.savez(f,image=image)
                        value=dict(request_sha256=rh,state=frame['receiver_state'],metrics={m:value[m]for m in METRICS},
                            image_sha256=image_sha(image),image_file_sha256=sha(imagepath),image_path=str(imagepath),image_slot=0,
                            reuse='NEW_RECEIVED_STATE',new_VAR_calls=0 if frame['gray']else 1)
                        save(sp,value);actual_new+=value['new_VAR_calls']
                    cache[key]=value
                v=cache[key];old_reuse+=v['reuse']=='ORIGINAL_STATE_IMAGE_AND_SCORE'
                rows.append(dict(source_index=i,source_id=rec['source_id'],snr_db=frame['snr_db'],noise_seed=SEED,
                    candidate_id=frame['candidate_id'],profile_id=frame['profile_id'],m=next(p['m']for p in r['profiles']if p['candidate_id']==frame['candidate_id']),K=0,
                    header_ok=frame['header_ok'],body_crc_accept=frame['body_crc_accept'],gray=frame['gray'],
                    E_frame=frame['transmission']['E'],rho=frame['transmission']['E']/2048,
                    received_state_sha256=key,image_sha256=v['image_sha256'],RX_reuse=frame['t2_reuse'],visual_reuse=v['reuse'],
                    image_path=v['image_path'],image_slot=v['image_slot'],**v['metrics']))
            save(cp,dict(status='T2_GPU_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],source_index=i,
                cpu_source=desc(cpu_path),rows=rows,new_VAR_calls_this_attempt=actual_new,old_score_memberships_reused=old_reuse))
            print(canonical(dict(stage='gpu',source=i,new_VAR_calls=actual_new,old_score_memberships=old_reuse)),flush=True)
        native.frozen();return dict(status='GPU_ALL100_COMPLETE',request_sha256=rh)

def close(path):
    r,rh=registered(path);out=Path(r['out']);rows=[];inputs={};newframes=reuseframes=0
    for rec in r['records']:
        i=rec['source_index'];p=out/'cpu_sources'/f'{i:04d}.json';c=read(p)
        require(c['request_sha256']==rh,'CPU request mismatch');inputs[str(p)]=sha(p)
        reuseframes+=c['reused_frames'];newframes+=212-c['reused_frames']
        p=out/'gpu_sources'/f'{i:04d}.json';g=read(p);require(g['request_sha256']==rh and len(g['rows'])==212,'GPU request mismatch');inputs[str(p)]=sha(p);rows+=g['rows']
    expected={(i,s,p['candidate_id'])for i in range(100)for s in SNRS for p in r['profiles']}
    require(len(rows)==21200 and {(x['source_index'],x['snr_db'],x['candidate_id'])for x in rows}==expected,'Complete pilot all106 actions')
    db=sqlite3.connect('file:'+str(out/'packet_ledger.sqlite')+'?mode=ro',uri=True)
    calls=db.execute('SELECT COUNT(*) FROM events').fetchone()[0];pending=db.execute("SELECT COUNT(*) FROM events WHERE status!='COMPLETE'").fetchone()[0];db.close()
    require(calls<=CAP and pending==0,'Independent packet budget incomplete')
    rankings=[];proxy=checked(r['original_proxy_scores'])
    for snr in SNRS:
        group=[]
        original_order=sorted(r['profiles'],key=lambda p:(-proxy[str(snr)][p['candidate_id']],p['candidate_id']))
        original_rank={p['candidate_id']:i+1 for i,p in enumerate(original_order)}
        for p in r['profiles']:
            rs=[x for x in rows if x['snr_db']==snr and x['candidate_id']==p['candidate_id']]
            group.append(dict(snr_db=snr,candidate_id=p['candidate_id'],profile_id=p['profile_id'],m=p['m'],K=0,source_count=100,noise_count=1,
                modulation=p['groups'][0]['modulation'],allocation_modes=';'.join(p['allocation_modes']),
                original_proxy_score=proxy[str(snr)][p['candidate_id']],original_proxy_rank_whole=original_rank[p['candidate_id']],
                **{m:math.fsum(float(x[m])for x in rs)/100 for m in METRICS},
                body_crc_rejected_fraction=sum(x['body_crc_accept']is False for x in rs)/100,header_rejected_fraction=sum(not x['header_ok']for x in rs)/100))
        group.sort(key=lambda x:(-x['dinov2_vitl14_cosine'],x['candidate_id']))
        rankings.extend(dict(x,pilot_rank=i+1,full_calibration_shortlist_top5=i<5,role='pilot_only_not_test_performance')for i,x in enumerate(group))
    for name,values in [('per_frame.csv',rows),('calibration_rankings.csv',rankings)]:
        p=out/name
        import io
        s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values);data=s.getvalue().encode()
        if p.exists():require(p.read_bytes()==data,'Completed CSV changed')
        else:p.write_bytes(data)
        inputs[str(p)]=sha(p)
    done=dict(status='T2_PILOT_COMPLETE_ONLY',request_sha256=rh,frame_count=21200,source_count=100,noise_count=1,SNRs=SNRS,
        whole_candidate_count=106,original_RX_frames_reused=reuseframes,new_RX_frames=newframes,new_packet_calls=calls,
        packet_cap=CAP,outputs=inputs,full1000_recheck_complete=False,holdout_started=False,original_ledgers_mutated=False)
    save(out/'completion.json',done);return done

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--visual-request',required=True);a.add_argument('--out',required=True);a.add_argument('--request',required=True);a.add_argument('--workers',type=int,default=8);a.add_argument('--deadline-unix',required=True,type=float);a.add_argument('--max-seconds',type=int,default=86400)
    for name in ['cpu-check','cpu-worker','gpu-worker','qualify-gpu','close']:
        a=sub.add_parser(name);a.add_argument('--request',required=True)
        if name=='cpu-worker':a.add_argument('--index',type=int,required=True)
    a=p.parse_args()
    value=prepare(a)if a.command=='prepare'else(cpu_check(a.request)if a.command=='cpu-check'else(cpu_worker(a.request,a.index)if a.command=='cpu-worker'else(gpu_worker(a.request,a.command=='qualify-gpu')if a.command in ['gpu-worker','qualify-gpu']else close(a.request))))
    if value is not None:print(canonical(value),flush=True)
if __name__=='__main__':main()
