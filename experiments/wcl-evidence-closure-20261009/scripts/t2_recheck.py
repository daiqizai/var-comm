"""Separately registered full1000 recheck after the complete106-action pilot.

No automatic launch. No original policy replacement. Only K=0 actions can win
the expanded WHOLE comparison; the frozen PARTIAL arm is a reference only.
"""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, io, json, math, os, subprocess
from pathlib import Path
import signal, sqlite3, sys, time
import t2_pilot as h
from t2_ledger import Ledger

SCHEMA='WCL_T2_FULL1000_RECHECK_V1'
SEEDS=[4101,4102,4103]

def build_schedule(rankings,all_profiles,policy):
    profiles={p['candidate_id']:p for p in all_profiles}
    full=[p['candidate_id']for p in all_profiles if p['K']==0 and'full_budget'in p['allocation_modes']]
    h.require(len(full)==7,'All seven full-budget whole controls')
    whole_ids={p['candidate_id']for p in all_profiles if p['K']==0}
    h.require(len(whole_ids)==106,'Original 106 whole actions required')
    schedule=[]
    for snr in h.SNRS:
        rr=sorted([x for x in rankings if int(x['snr_db'])==snr],key=lambda x:int(x['pilot_rank']))
        h.require(len(rr)==106 and[int(x['pilot_rank'])for x in rr]==list(range(1,107)),'All pilot ranks required')
        h.require({x['candidate_id']for x in rr}==whole_ids,'All unique original whole identities required')
        selected={x['candidate_id']for x in rr[:5]}|set(full)
        winners={x['family']:x['candidate_id']for x in policy['winners']if x['snr_db']==snr}
        h.require(set(winners)=={'WHOLE','PARTIAL'},'Original two-granularity policies')
        selected.update(winners.values())
        for cid in sorted(selected):
            p=profiles[cid];reasons=[]
            if cid in {x['candidate_id']for x in rr[:5]}:reasons.append('pilot_top5')
            if cid in full:reasons.append('full_budget_whole')
            reasons.extend('original_'+f.lower()+'_winner'for f,c in winners.items()if c==cid)
            schedule.append(dict(snr_db=snr,candidate_id=cid,profile=p,reasons=reasons,eligible_for_expanded_whole=p['K']==0))
    return schedule

def prepare(a):
    pilot,rph=h.registered(a.pilot_request);po=Path(pilot['out']);done=h.read(po/'completion.json')
    h.require(done['status']=='T2_PILOT_COMPLETE_ONLY'and done['request_sha256']==rph and done['frame_count']==21200,'Complete106-action pilot required')
    rankpath=str(po/'calibration_rankings.csv');h.require(done['outputs'][rankpath]==h.sha(rankpath),'Pilot rankings changed')
    with Path(rankpath).open(newline='')as f:rankings=list(csv.DictReader(f))
    vr=pilot['visual_config'];cat=h.read(vr['catalogue']);profiles={p['candidate_id']:p for p in cat['profiles']}
    policy=h.checked(h.desc(a.original_policy));h.require(policy['status']=='POLICIES_FROZEN_ON_CALIBRATION1000','Original frozen policy required')
    h.require(h.sha(a.original_policy)=='7871ac7f9c619bd21c63ba31157738cd56c0dda9344c993e99b935f60ce70c0c','Original PARTIAL policy cannot change')
    schedule=build_schedule(rankings,cat['profiles'],policy)
    manifest=h.checked(pilot['original_source_manifest']);old=h.checked(pilot['original_visual_completion']);records=[]
    for rec in manifest['records']:
        cp=h.read(rec['checkpoint']);h.require(h.sha(rec['checkpoint'])==rec['checkpoint_sha256'],'Original source checkpoint changed')
        h.require(cp['source_id']==rec['source_id']and cp['source_index']==rec['source_index']and cp['tokens_sha256']==rec['tokens_sha256'],'Original source token identity')
        i=rec['source_index'];ocp=str(Path(vr['out'])/'source_checkpoints'/f'{i:04d}.json');orp=str(Path(vr['out'])/'sources'/f'{i:04d}.json')
        h.require(old['outputs'].get(ocp)==h.sha(ocp),'Old visual checkpoint changed')
        oc=h.read(ocp);h.require(oc['source_id']==rec['source_id']and oc['frame_count']==90 and oc['images_scored'],'Original visual population')
        h.require(oc['outputs'][orp]==old['outputs'][orp],'Original visual rows seal')
        records.append(dict(rec,archive_sha256=cp['outputs'][rec['archive']],class_index=cp['evaluation_class_index'],
            old_visual_checkpoint=h.desc(ocp),old_visual_rows=dict(path=orp,sha256=oc['outputs'][orp]),old_image_archives={p:v for p,v in oc['outputs'].items()if p.endswith('.npz')}))
    r=dict(pilot);r.update(schema=SCHEMA,parent_pilot_request=h.desc(a.pilot_request),parent_pilot_completion=h.desc(po/'completion.json'),
        parent_pilot_qualification=h.desc(po/'qualification_gpu.json'),original_policy=h.desc(a.original_policy),source_count=1000,records=records,
        out=str(Path(a.out).absolute()),schedule=schedule,profiles=[profiles[cid]for cid in sorted({x['candidate_id']for x in schedule})],noise_seeds=SEEDS,
        frame_count=len(schedule)*1000*3,packet_cap=len(schedule)*1000*3*2,deadline_unix=a.deadline_unix,max_seconds=a.max_seconds,workers=a.workers,
        stop_files=[str(Path(pilot['root'])/'STOP'),str(Path(a.out).absolute()/'STOP')],policy_selection='K0 only by full1000 source-mean DINO-L; PARTIAL unchanged',
        source_rule='all1000 in original calibration manifest, no quality selection',
        noise_rule='Original main_raw64_calibration_checkpoints.standard_noise(source_id, each of4101/4102/4103)',
        counter_rule='Original six-SNR order,1000-source counter: (SNRindex*1000+source_index)*3+noise_index',
        GPU_upper_bound_new_renders=len(schedule)*3000,
        GPU_expected_new_renders='Determined from actual received-state deduplication; exact old/pilot cache first',
        gpu_workers=a.gpu_workers,gpu_parallel_qualification_new_VAR_calls=2 if a.gpu_workers==2 else 0,
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=pilot['root'],text=True).strip())
    r.pop('noise_seed',None)
    r['source_bindings']=dict(pilot['source_bindings']);r['source_bindings'][str(Path(__file__).absolute())]=h.sha(__file__)
    parallel=Path(__file__).with_name('t2_gpu_parallel.py').absolute();r['source_bindings'][str(parallel)]=h.sha(parallel)
    r['gpu_cohort_runtime']=str(Path(pilot['root'])/'experiments/m1-gpu-workers-20261004/cohort_runtime.py')
    h.require(str(r['gpu_cohort_runtime'])in r['source_bindings'],'Original cohort implementation must already be bound')
    closure=h.load(vr['static_closure_module'],'_t2_full_closure',r['source_bindings'])
    native=closure.collect_bindings(r['root'],vr['native_runtime'],vr['var_source'],vr['dino_source'],vr['uep_runtime'])
    for p,v in pilot['native_source_bindings'].items():h.require(native.get(p)==v,'Previously admitted native source changed')
    r['native_source_bindings']=native;r['source_bindings'].update(native)
    validate(r);h.save(a.request,r)
    summary=dict(status='FULL1000_RECHECK_PREPARED_NOT_RUN',request=h.desc(a.request),candidate_counts={s:sum(x['snr_db']==s for x in schedule)for s in h.SNRS},
        frame_count=r['frame_count'],packet_cap=r['packet_cap'],new_scientific_calls=0,schedule=[{k:v for k,v in x.items()if k!='profile'}for x in schedule])
    h.save(Path(a.out)/'prepared_plan.json',summary);return summary

def validate(r):
    h.require(r['schema']==SCHEMA and r['source_count']==1000 and r['noise_seeds']==SEEDS and r['snrs']==h.SNRS,'Fixed full1000/three-noise/twoSNR scope')
    h.require([x['source_index']for x in r['records']]==list(range(1000)),'All original1000 calibration sources')
    h.require(1<=r['workers']<=16 and r['frame_count']==len(r['schedule'])*3000 and r['packet_cap']==r['frame_count']*2<=168000,'Finite full recheck budget')
    h.require(r.get('gpu_workers',1)in(1,2),'Only one or two original-numerical GPU workers')
    h.require(len({(x['snr_db'],x['profile']['wire_key'])for x in r['schedule']})==len(r['schedule']),'Deduplicate actual wires')
    for snr in h.SNRS:h.require(sum(x['snr_db']==snr and'full_budget'in x['profile']['allocation_modes']and x['profile']['K']==0 for x in r['schedule'])==7,'Retain all seven whole full-budget controls')
    h.require(Path(r['out']).is_relative_to(Path(r['root'])/'outputs')and'WCL'in r['out'].upper(),'New independent output namespace')
def registered(path):
    r=h.read(path);validate(r)
    for p in [__file__,h.__file__,str(Path(h.__file__).with_name('t2_ledger.py')),str(Path(__file__).with_name('t2_gpu_parallel.py'))]:h.require(r['source_bindings'][str(Path(p).absolute())]==h.sha(p),'Bound runner changed')
    return r,h.sha(path)
def frame_path(out,i,s,p,n):return out/'cpu_frames'/f'{i:04d}'/f'{s}_{p:03d}_{n}.json'

def pilot_checkpoint(pilot,completion,kind,i,request_sha):
    p=str(Path(pilot['out'])/kind/f'{i:04d}.json')
    h.require(completion['outputs'].get(p)==h.sha(p),'Completed parent pilot output changed')
    cp=h.read(p);h.require(cp['request_sha256']==request_sha,'Parent pilot request changed')
    return cp

def cpu_worker(path,index):
    import numpy as np
    r,rh=registered(path);out=Path(r['out']);out.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    h.require(0<=index<r['workers']and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only valid worker')
    for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,h.stop)
    with h.lock(out/f'cpu_worker_{index}.lock'):
        ledger=Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap']);a,wire,rx,cat,receiver,core=h.cpu_runtime(r,ledger)
        pilot=h.checked(r['parent_pilot_request']);pilot_done=h.checked(r['parent_pilot_completion'])
        prefix=[0,1,5,14,30,55,91,155,255,424,680];completed=[]
        for rec in r['records'][index::r['workers']]:
            h.guard(r,began);i=rec['source_index'];cp=out/'cpu_sources'/f'{i:04d}.json';expected_count=len(r['schedule'])*3
            if cp.exists():
                d=h.read(cp);h.require(d['request_sha256']==rh and len(d['frames'])==expected_count,'Completed source differs');completed.append(h.desc(cp));continue
            tokens,_=h.source_assets(rec);scales=[tokens[prefix[j]:prefix[j+1]]for j in range(10)]
            old=h.old_rows(rec);pm={(x['snr_db'],x['candidate_id'],x['noise_seed']):x for x in old}
            if i<100:
                pc=pilot_checkpoint(pilot,pilot_done,'cpu_sources',i,r['parent_pilot_request']['sha256'])
                for x in pc['frames']:pm.setdefault((x['snr_db'],x['candidate_id'],x['noise_seed']),x)
            frames=[]
            for point in r['schedule']:
                snr=point['snr_db'];p=point['profile'];pid=p['profile_id'];payload=wire.serialize_raw(scales,p)
                for seed in SEEDS:
                    h.guard(r,began);fp=frame_path(out,i,snr,pid,seed)
                    if fp.exists():
                        row=h.read(fp);h.require(row['recheck_request_sha256']==rh,'Existing frame differs');frames.append(row);continue
                    counter=core.frame_counter(i,snr,seed);wave,tx=rx.transmit_frame(cat,a.backend,a.legacy,a.header,pid,payload,counter)
                    pure=dict(snr_db=snr,candidate_id=p['candidate_id'],profile_id=pid,wire_key=p['wire_key'])
                    expected,observed=core.prepare_frame(rec,pure,seed,payload,wave,tx);prior=pm.get((snr,p['candidate_id'],seed))
                    if prior is not None:
                        h.require(all(prior[k]==v for k,v in expected.items()),'Prior exact reception mismatch')
                        if 'effective_catalogue_digest'in prior:h.require(prior['effective_catalogue_digest']==cat.digest,'Prior effective catalogue mismatch')
                        else:h.require(prior['codebook_sha256']==cat.digest and prior['aliases_sha256']==cat.aliases_digest,'Prior pilot catalogue mismatch')
                        actual=wire.present_actual(prior['header'],prior['body'],cat);h.require(actual['receiver_state']==prior['receiver_state'],'Prior actual state inconsistent')
                        row=dict(prior,recheck_RX_reuse='EXACT_ORIGINAL_OR_PILOT',new_packet_calls=0)
                    else:
                        event=f'{SCHEMA}/{rh[:16]}/source{i:04d}/snr{snr}/profile{pid}/noise{seed}'
                        row=receiver.receive(observed,float(snr),counter,phase='actual_calibration',event_prefix=event)
                        row.update(expected,recheck_RX_reuse='NEW_ACTUAL_RX',new_packet_calls=row['logical_packet_calls'])
                    row.update(recheck_request_sha256=rh,image_reconstruction_complete=False,visual_quality_scored=False)
                    h.save(fp,row);frames.append(row)
            oldstates={h.state_key(x)for x in old};states={h.state_key(x)for x in frames}
            h.save(cp,dict(status='FULL_RECHECK_CPU_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],frames=frames,
                reused_frames=sum(x['recheck_RX_reuse']!='NEW_ACTUAL_RX'for x in frames),unique_states=len(states),
                maximum_new_state_renders=len(states-oldstates),source=rec));completed.append(h.desc(cp))
            print(h.canonical(dict(stage='full_cpu',worker=index,source=i,completed_sources=len(completed))),flush=True)
        h.save(out/'cpu_workers'/f'{index:02d}.json',dict(status='COMPLETE',request_sha256=rh,worker=index,sources=completed));ledger.close()

def gpu_worker(path,cohort_config=None):
    import numpy as np
    r,rh=registered(path);out=Path(r['out']);began=time.monotonic();pilot=h.checked(r['parent_pilot_request']);po=Path(pilot['out'])
    h.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit GPU0')
    pilot_done=h.checked(r['parent_pilot_completion'])
    qual=h.checked(r['parent_pilot_qualification']);h.require(qual['status']=='PASS'and qual['request_sha256']==r['parent_pilot_request']['sha256'],'Reuse actual parent GPU qualification')
    for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,h.stop)
    if cohort_config is None:
        h.require(r['gpu_workers']==1,'Two-worker request must use the sealed GPU owner')
        shared=h.lock(r['visual_config']['visual_lock']);own=h.lock(out/'gpu.lock');records=r['records'];parallel=None
    else:
        import t2_gpu_parallel as parallel
        cohort,probe=parallel.admit(r,rh,cohort_config)
        shared=contextlib.nullcontext();own=h.lock(out/f'gpu_worker_{cohort_config["worker_id"]}.lock')
        records=[r['records'][i]for i in cohort_config['source_indices']]
    with shared,own:
        native,source,scorer=h.gpu_build(r,out)
        if parallel is not None:
            h.require(native.assets.old.b is probe,'Native used a different cohort admission probe')
            parallel.qualify(r,rh,native,source,scorer,cohort_config,cohort)
        for rec in records:
            if parallel is not None:cohort.require_available()
            h.guard(r,began);i=rec['source_index'];cp=out/'gpu_sources'/f'{i:04d}.json'
            if cp.exists():h.require(h.read(cp)['request_sha256']==rh,'Completed full GPU source differs');continue
            cpu_path=out/'cpu_sources'/f'{i:04d}.json';cpu=h.read(cpu_path);h.require(cpu['request_sha256']==rh,'Full CPU request mismatch')
            prior=h.old_rows(rec);cache={};archives={}
            for old in prior:
                key=h.state_key(old)
                if key in cache:continue
                p=old['image_archive']
                if p not in archives:
                    h.require(h.sha(p)==rec['old_image_archives'][p],'Original image archive changed')
                    with np.load(p,allow_pickle=False)as z:archives[p]=z['images'].copy()
                h.require(h.image_sha(archives[p][old['image_slot']])==old['image_sha256'],'Original scored image changed')
                cache[key]=dict(image_sha256=old['image_sha256'],image_path=p,image_slot=old['image_slot'],metrics={m:old[m]for m in h.METRICS},reuse='ORIGINAL_STATE_IMAGE_AND_SCORE')
            if i<100:
                pc=pilot_checkpoint(pilot,pilot_done,'gpu_sources',i,r['parent_pilot_request']['sha256'])
                for row in pc['rows']:
                    key=row['received_state_sha256']
                    if key in cache:continue
                    p=Path(row['image_path']);sp=po/'gpu_states'/f'{i:04d}'/(key+'.json');state=h.read(sp)
                    h.require(state['request_sha256']==r['parent_pilot_request']['sha256']and h.sha(p)==state['image_file_sha256']and h.digest(state['state'])==key,'Pilot state image changed')
                    h.require(state['image_sha256']==row['image_sha256']and all(state['metrics'][m]==row[m]for m in h.METRICS),'Pilot state scores differ from sealed checkpoint')
                    cache[key]=dict(state,reuse='PILOT_STATE_IMAGE_AND_SCORE')
            _,pixels=h.source_assets(rec);sr=h.score_record(rec);scorer.prepare_source(sr,pixels.astype(np.float32)/np.float32(255),i);rows=[];newcalls=0
            for frame in cpu['frames']:
                h.guard(r,began);key=h.state_key(frame)
                if key not in cache:
                    sp=out/'gpu_states'/f'{i:04d}'/(key+'.json');ip=sp.with_suffix('.npz');res=sp.with_suffix('.reserved.json')
                    if sp.exists():
                        v=h.read(sp);h.require(v['request_sha256']==rh and v['state']==frame['receiver_state']and h.sha(ip)==v['image_file_sha256'],'Full state cache changed')
                    else:
                        h.require(not res.exists(),'Unresolved full state render; no automatic repeat')
                        h.save(res,dict(request_sha256=rh,state=frame['receiver_state']))
                        image=h.render_state(native,source,frame['receiver_state']);metrics=scorer(sr,[image])[0]
                        with ip.open('xb')as f:np.savez(f,image=image)
                        v=dict(request_sha256=rh,state=frame['receiver_state'],image_path=str(ip),image_slot=0,image_file_sha256=h.sha(ip),image_sha256=h.image_sha(image),
                            metrics={m:metrics[m]for m in h.METRICS},reuse='NEW_RECEIVED_STATE',new_VAR_calls=0 if frame['gray']else 1)
                        h.save(sp,v);newcalls+=v['new_VAR_calls']
                    cache[key]=v
                v=cache[key];p=next(x['profile']for x in r['schedule']if x['snr_db']==frame['snr_db']and x['candidate_id']==frame['candidate_id'])
                rows.append(dict(source_index=i,source_id=rec['source_id'],snr_db=frame['snr_db'],noise_seed=frame['noise_seed'],candidate_id=frame['candidate_id'],profile_id=frame['profile_id'],
                    N=1024,population='original_calibration1000',protocol_version=SCHEMA,code_commit=r['execution_commit'],
                    method_id='RAW_WHOLE_EXPANDED_CANDIDATE_V1'if p['K']==0 else'RAW_PARTIAL_ORIGINAL_POLICY_REFERENCE',
                    m=p['m'],K=p['K'],eligible_for_expanded_whole=p['K']==0,header_ok=frame['header_ok'],body_crc_accept=frame['body_crc_accept'],gray=frame['gray'],
                    E_frame=frame['transmission']['E'],rho=frame['transmission']['E']/2048,received_state_sha256=key,image_sha256=v['image_sha256'],
                    RX_reuse=frame['recheck_RX_reuse'],visual_reuse=v['reuse'],image_path=v['image_path'],image_slot=v['image_slot'],**v['metrics']))
            h.save(cp,dict(status='FULL_RECHECK_GPU_SOURCE_COMPLETE',request_sha256=rh,source_id=rec['source_id'],rows=rows,new_VAR_calls_this_attempt=newcalls,cpu_source=h.desc(cpu_path)))
            print(h.canonical(dict(stage='full_gpu',source=i,new_VAR_calls=newcalls)),flush=True)
        native.frozen()
        if parallel is not None:cohort.require_available()

def close(path):
    r,rh=registered(path);out=Path(r['out']);rows=[];pins={};logical_per_source=len(r['schedule'])*3
    if r['gpu_workers']==2:
        for i in range(2):
            p=out/'gpu_parallel_qualification'/f'{i:04d}.json';q=h.read(p)
            h.require(q['status']=='PASS'and q['request_sha256']==rh,'Completed parallel qualification required');pins[str(p)]=h.sha(p)
    for i in range(1000):
        p=out/'gpu_sources'/f'{i:04d}.json';d=h.read(p);h.require(d['request_sha256']==rh and len(d['rows'])==logical_per_source,'Complete full1000 source')
        h.checked(d['cpu_source']);pins[d['cpu_source']['path']]=d['cpu_source']['sha256'];pins[str(p)]=h.sha(p);rows.extend(d['rows'])
    expected={(i,x['snr_db'],x['candidate_id'],n)for i in range(1000)for x in r['schedule']for n in SEEDS}
    h.require(len(rows)==r['frame_count']and{(x['source_index'],x['snr_db'],x['candidate_id'],x['noise_seed'])for x in rows}==expected,'Full calibrated action/source/noise grid')
    db=sqlite3.connect('file:'+str(out/'packet_ledger.sqlite')+'?mode=ro',uri=True);calls=db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    unresolved=db.execute("SELECT COUNT(*) FROM events WHERE status!='COMPLETE'").fetchone()[0];db.close();h.require(calls<=r['packet_cap']and unresolved==0,'Full recheck meter unresolved')
    rankings=[];winners=[];original=h.checked(r['original_policy']);proxy=h.checked(r['original_proxy_scores'])
    pilot=h.checked(r['parent_pilot_request']);pilot_done=h.checked(r['parent_pilot_completion'])
    rankingpath=str(Path(pilot['out'])/'calibration_rankings.csv')
    h.require(pilot_done['outputs'][rankingpath]==h.sha(rankingpath),'Parent pilot rankings changed')
    with Path(rankingpath).open(newline='')as f:pilot_ranks={(int(x['snr_db']),x['candidate_id']):x for x in csv.DictReader(f)}
    for snr in h.SNRS:
        group=[]
        for point in[x for x in r['schedule']if x['snr_db']==snr]:
            cid=point['candidate_id'];sub=[x for x in rows if x['snr_db']==snr and x['candidate_id']==cid];p=point['profile']
            bysource={i:[]for i in range(1000)}
            for x in sub:bysource[x['source_index']].append(x)
            h.require(all(len(v)==3 for v in bysource.values()),'Exactly three noises for each source')
            means={m:math.fsum(math.fsum(float(x[m])for x in bysource[i])/3 for i in range(1000))/1000 for m in h.METRICS}
            oldrank=pilot_ranks.get((snr,cid),{});g=p['groups'][0]
            group.append(dict(snr_db=snr,candidate_id=cid,profile_id=p['profile_id'],m=p['m'],K=p['K'],eligible_for_expanded_whole=p['K']==0,source_count=1000,noise_count=3,
                original_proxy_score=proxy[str(snr)][cid],original_proxy_rank_whole=oldrank.get('original_proxy_rank_whole',''),pilot_rank=oldrank.get('pilot_rank',''),
                modulation=g['modulation'],information_bits=g['information_bits'],transmitted_bits=g['transmitted_bits'],allocation_modes=';'.join(p['allocation_modes']),
                body_symbols=p['used_body_symbols'],header_symbols=p['header_symbols'],padding_symbols=p['idle_symbols'],
                reasons=';'.join(point['reasons']),**means,body_crc_rejected_fraction=sum(x['body_crc_accept']is False for x in sub)/3000,
                header_rejected_fraction=sum(not x['header_ok']for x in sub)/3000))
        ranked=sorted([x for x in group if x['eligible_for_expanded_whole']],key=lambda x:(-x['dinov2_vitl14_cosine'],x['candidate_id']))
        rank={x['candidate_id']:i+1 for i,x in enumerate(ranked)};rankings.extend(dict(x,whole_rank=rank.get(x['candidate_id'],''))for x in group)
        win=ranked[0];old=next(x for x in original['winners']if x['family']=='WHOLE'and x['snr_db']==snr)
        winners.append(dict(snr_db=snr,candidate_id=win['candidate_id'],profile_id=win['profile_id'],mean_DINO_L=win['dinov2_vitl14_cosine'],original_whole_candidate_id=old['candidate_id'],changed=win['candidate_id']!=old['candidate_id']))
    for name,values in [('per_frame.csv',rows),('calibration_rankings.csv',rankings)]:
        buf=io.StringIO(newline='');w=csv.DictWriter(buf,list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values);p=out/name;data=buf.getvalue().encode()
        if p.exists():h.require(p.read_bytes()==data,'Completed full CSV changed')
        else:p.write_bytes(data)
        pins[str(p)]=h.sha(p)
    policy=dict(status='EXPANDED_WHOLE_FROZEN_ON_ORIGINAL_CALIBRATION1000',method_id='RAW_WHOLE_EXPANDED_CALIBRATION_V1',objective='dinov2_vitl14_cosine',noise_reduction='source_mean_first',winners=winners,
        original_partial_policy_unchanged=r['original_policy'],same_source_posthoc_holdout_required_at=[x['snr_db']for x in winners if x['changed']],holdout_started=False,systematic_prescreen_followup='Assess all audited points; if systematic issue appears, separately register same-rule other4SNR audit')
    h.save(out/'expanded_whole_policy.json',policy);pins[str(out/'expanded_whole_policy.json')]=h.sha(out/'expanded_whole_policy.json')
    done=dict(status='T2_FULL1000_CALIBRATION_COMPLETE_NOT_HOLDOUT',request_sha256=rh,source_count=1000,noise_count=3,frame_count=r['frame_count'],new_packet_calls=calls,packet_cap=r['packet_cap'],outputs=pins,original_ledgers_mutated=False,original_policy_overwritten=False)
    h.save(out/'completion.json',done);return done

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for x in ['pilot-request','original-policy','out','request']:a.add_argument('--'+x,required=True)
    a.add_argument('--workers',type=int,default=8);a.add_argument('--gpu-workers',type=int,choices=[1,2],default=1);a.add_argument('--deadline-unix',type=float,required=True);a.add_argument('--max-seconds',type=int,default=172800)
    for cmd in ['cpu-worker','gpu-worker','close']:
        a=sub.add_parser(cmd);a.add_argument('--request',required=True)
        if cmd=='cpu-worker':a.add_argument('--index',type=int,required=True)
    a=p.parse_args();value=prepare(a)if a.command=='prepare'else(cpu_worker(a.request,a.index)if a.command=='cpu-worker'else(gpu_worker(a.request)if a.command=='gpu-worker'else close(a.request)))
    if value is not None:print(h.canonical(value),flush=True)
if __name__=='__main__':main()
