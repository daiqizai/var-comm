"""B2 actual-pixel DINO-only gaps and exactly four frozen policies; no bootstrap."""
from analysis_common import *
import argparse,csv,collections

def canonical_digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def compatible_dino(a,b,flags,runtime):
    assert dino_identity(a['DINO_L_metadata'])==dino_identity(b['DINO_L_metadata']), 'BLOCKED_DINO_SUBIDENTITY'
    assert a['numeric_flags']==b['numeric_flags']==flags, 'BLOCKED_DINO_FLAGS'
    # The complete metadata carries the full package/runtime identity; the old
    # four-metric suite also explicitly binds the original numerical runtime.
    if 'runtime_identity' in a:assert a['runtime_identity']['expected_runtime']==runtime
    if 'runtime_identity' in b:assert b['runtime_identity']['expected_runtime']==runtime

def check_cached_state(state,saved):
    for k in ('source_id','source_index','codec','endpoint','source','image','array_key','image_float_sha256','dino'):
        assert state[k]==saved[k],('BLOCKED_CAL100_CACHE_CHANGED',k)

def check_pair(record,sid,source_sha,image_sha,value,allowed_identities):
    assert record['metric_identity'] in allowed_identities, 'BLOCKED_UNBOUND_METRIC_IDENTITY'
    if 'reference_archive' in record:
        assert record['reference_archive']['sha256']==source_sha
        assert record['reconstruction_archive']['sha256']==image_sha
        assert float(record['metrics']['dinov2_vitl14_cosine'])==float(value)
    else:
        assert record['source']['sha256']==source_sha
        if record.get('image') is not None:assert record['image']['sha256']==image_sha
        assert float(record['scores']['dinov2_vitl14_cosine'])==float(value)
    if 'source_id' in record:assert record['source_id']==sid

def csvread(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def csvwrite(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())

def prepare(expected,gpu):
    r=registration('B2',expected);root=L/'B2/dino';root.mkdir()
    npth=O/'B2/native_completion.json';nc=read(npth)
    assert nc['status']=='COMPLETE' and nc['process_closure_status']=='ALL_NATIVE_PROCESSES_GONE'
    for d in nc['process_closure_evidence']:check_observation(d)
    tr=load(r['transport_registration']);sources=load(tr['private_sources'])
    assert len(sources)==1000;source_map={s['source_id']:s for s in sources}
    logical=load(nc['logical_pixels']);assert len(logical)==27000
    pairs={}
    for row in logical:
        assert row['source_id'] in source_map
        k=(row['source_id'],row['image_float_sha256'])
        if k not in pairs:
            image(row['image'],row['array_key'],row['image_float_sha256'])
            pairs[k]=dict(source_id=row['source_id'],source_index=row['source_index'],image=row['image'],array_key=row['array_key'],image_float_sha256=row['image_float_sha256'])
    # Numerical identity is the original MetricEvaluator metadata with its
    # population-label field removed, exactly as in the previous closed worker.
    oldreg=read(OLD_O/'a2_metrics_v1/registration.json');meta=read(OLD_O/'a2_metrics_v1/metric_identity.json')
    met2=read(OLD_O/'a2b_metrics_v1/metric_identity.json')
    assert dino_identity(meta['DINO_L_metadata'])==dino_identity(met2['DINO_L_metadata'])
    routes=load(r['model_routes']);runtime=routes['native_qualification']['numeric_settings']['runtime']
    compatible_dino(meta,met2,routes['numeric_flags'],runtime)
    # The original pilot and T1-B pairs use one four-metric suite identity. The
    # published T1-B identity contains that exact suite metadata (not a new ID).
    suitepath=BASE/'baseline_strength_20261011/t1b_m10_calibration_20261016_v1/metrics_recovery_v2/metric_identity.json'
    suite=load(descriptor(suitepath,'d1bdb801caeb461dc9cddc048bc4be9b9bf60b761fcbf6c261c84ee05646a04b',53830))
    assert canonical_digest(suite['metadata'])==suite['identity']
    compatible_dino(suite['metadata'],meta,routes['numeric_flags'],runtime)
    assert suite['observed_runtime']=={k:runtime[k] for k in ('python','torch','numpy','cuda','cudnn')}
    reg2=read(OLD_O/'a2b_metrics_v1/registration.json');done2=read(OLD_O/'a2b_metrics_v1/completion.json')
    assert done2['unresolved']==0 and done2['new_channel']==0
    for d in done2['outputs']:verify(d)
    done1=read(OLD_O/'a2_metrics_v1/completion.json')
    assert done1['unresolved']==0 and done1['metric_identity']==done2['old_metric_identity']
    assert load(next(d for d in done1['outputs'] if d['path']==str(OLD_O/'a2_metrics_v1/metric_identity.json')))==meta
    assert load(next(d for d in done2['outputs'] if d['path']==str(OLD_O/'a2b_metrics_v1/metric_identity.json')))==met2
    allowed={suite['identity'],done1['metric_identity'],done2['metric_identity']}
    prior_pairs={x['pair_key']:x['file'] for x in reg2['prior_pair_files']}
    def old_pair(key):
        if key in prior_pairs:return prior_pairs[key]
        p=OLD_L/'a2_metrics_v1'/(key+'.json')
        return stable_pin(p if p.exists() else OLD_L/'a2b_metrics_v1'/(key+'.json'))
    refs={s['source_id']:s for s in reg2['prior_references']};plan2=load(reg2['pair_plan'])
    assert len(refs)==100 and len(plan2['sources'])==1000
    for s in plan2['sources']:
        sid=s['source_id'];assert s['archive']['sha256']==source_map[sid]['source']['sha256']
        if sid not in refs:refs[sid]=dict(source_id=sid,source=s['archive'],file=stable_pin(OLD_L/'a2b_metrics_v1'/(str(s['source_index'])+'_reference.npz')))
    assert len(refs)==1000
    cached={};cache_evidence=[stable_pin(suitepath),stable_pin(OLD_O/'a2_metrics_v1/metric_identity.json'),
        stable_pin(OLD_O/'a2b_metrics_v1/metric_identity.json'),stable_pin(OLD_O/'a2_metrics_v1/completion.json'),
        stable_pin(OLD_O/'a2b_metrics_v1/completion.json')]
    def add(sid,h,v,proof,source_sha):
        k=(sid,h)
        if k not in pairs:return
        assert source_map[sid]['source']['sha256']==source_sha and math.isfinite(v)
        if k in cached:assert cached[k]['dino']==v,('BLOCKED_DINO_CACHE_CONFLICT',k)
        cached[k]=dict(**pairs[k],dino=v,score_evidence=proof,source=source_map[sid]['source'])
    # Every saved cal100 state is closed, including the 1,793 actual new scores.
    calreg=read(old.O/'cal100_score_registration.json')
    assert calreg['logical_pixel_pair_reuse']==0
    calclean=read(old.O/'cal100_clean_registration.json');verified=load(calclean['exact_cal100_cache'])
    assert verified['status']=='EXACT_CLOSED_CAL100_CACHE_VERIFIED'
    for d in verified['input_files']:verify(d)
    saved_cache={(x['source_id'],x['codec'],x['endpoint']):x for x in load(verified['cache'])}
    caldone=read(old.L/'cal100_scores_v1/worker/completion.json')
    assert caldone['ledger']['unresolved']==0
    calmeta=load(caldone['metric_identity'])
    assert calmeta['current_DINO_metadata']==dino_identity(meta['DINO_L_metadata'])
    assert calmeta['runtime']==runtime and calmeta['flags']==routes['numeric_flags']
    assert calreg['expected_runtime']==runtime and load(calreg['prior_metadata'])==meta
    cache_evidence.extend([stable_pin(old.O/'cal100_score_registration.json'),stable_pin(old.O/'cal100_clean_registration.json'),
        calclean['exact_cal100_cache'],verified['cache'],calreg['all_states'],
        stable_pin(old.L/'cal100_scores_v1/worker/completion.json'),caldone['scores'],caldone['metric_identity']])
    calnew={(x['source_id'],x['image_float_sha256']):x for x in load(caldone['scores'])}
    for x in load(calreg['all_states']):
        k=(x['source_id'],x['image_float_sha256'])
        if k not in pairs:continue
        if x['dino'] is None:
            new=calnew[k];record=load(new['pair']);assert record['image_float_sha256']==k[1] and record['source']['sha256']==x['source']['sha256']
            assert new['dino']==record['dinov2_vitl14_cosine']
            v=new['dino'];proof=new['pair']
        else:
            saved=saved_cache[x['source_id'],x['codec'],x['endpoint']];check_cached_state(x,saved)
            v=x['dino'];proof=saved['score_evidence'];verify(proof)
            if x['codec']==0:
                record=load(proof);check_pair(record,k[0],x['source']['sha256'],x['image']['sha256'],v,{suite['identity']})
            else:
                key=saved['cache_provenance']['score']['values']['pair_key'];record=load(old_pair(key))
                check_pair(record,k[0],x['source']['sha256'],x['image']['sha256'],v,{done1['metric_identity'],done2['metric_identity']})
                a,_=image(x['image'],x['array_key'],k[1])
                assert record['image_float_sha256']==hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
        add(k[0],k[1],float(v),proof,x['source']['sha256'])
    # Reuse the exact B0 NeST source/pixel-pair scores attached to the accepted
    # native object. A legal wrong source input never borrows its donor's score.
    import numpy as np
    for obj in load(nc['actual_native_objects']):
        ev=obj.get('cache_evidence',{})
        if ev.get('kind')!='B0_EXACT_INPUT':continue
        s,c=ev['source'],ev['cache'];score=c.get('score')
        if s['codec']!=0 or score is None:continue
        sid=s['source_id'];h=obj['image_float_sha256']
        if (sid,h) not in pairs:continue
        rec=load(score['pair'])
        assert rec['reference_archive']['sha256']==s['source']['sha256']
        assert rec['reconstruction_archive']['sha256']==obj['image']['sha256']
        assert rec['metrics']==score['metrics']
        check_pair(rec,sid,s['source']['sha256'],obj['image']['sha256'],rec['metrics']['dinov2_vitl14_cosine'],{suite['identity']})
        add(sid,h,float(rec['metrics']['dinov2_vitl14_cosine']),score['pair'],s['source']['sha256'])
    # All ARPC cal1000 saved pairs include gray failures. The CSV state is the
    # raw float SHA; its original pair receipt binds the structured float SHA.
    for x in csvread(OLD_O/'a2b_metrics_v1/actual_pair_scores.csv'):
        sid=x['source_id'];h=x['state'];k=(sid,h)
        if k not in pairs:continue
        proof=old_pair(x['pair_key']);record=load(proof)
        a,_=image(pairs[k]['image'],pairs[k]['array_key'],h)
        structured=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
        assert record['image_float_sha256']==structured and record['metric_identity'] in {done2['metric_identity'],done2['old_metric_identity']}
        v=float(record['scores']['dinov2_vitl14_cosine']);assert v==float(x['dinov2_vitl14_cosine'])
        add(sid,h,v,proof,record['source']['sha256'])
    groups={};new=[]
    for k,row in sorted(pairs.items()):
        if k in cached:continue
        sid=k[0];s=source_map[sid];ref=refs[sid]
        assert ref['source']['sha256']==s['source']['sha256'];verify(ref['file'])
        with np.load(ref['file']['path'],allow_pickle=False) as z:f=z['features'].copy()
        assert f.dtype==np.float32 and f.shape==(1,1024) and np.isfinite(f).all()
        group=groups.setdefault(sid,dict(source_id=sid,source_index=s['source_index'],archive=s['source'],reference=ref['file'],pairs=[]))
        group['pairs'].append(row);new.append(row)
    put(root/'plan.json',list(groups.values()));put(root/'cached_pairs.json',list(cached.values()))
    put(root/'source_references.json',list(refs.values()))
    put(root/'cache_metric_compatibility.json',dict(status='EXACT_DINO_SUBIDENTITIES_AND_CLOSED_CACHE_PROVENANCE',
        evidence=cache_evidence,allowed_suite_identities=sorted(allowed),DINO_subidentity=dino_identity(meta['DINO_L_metadata']),
        numeric_flags=routes['numeric_flags'],runtime=runtime,new_metric_calls=0,global_suite_identity_equality_claimed=False))
    vals=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=pci.bus_id,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert int(vals[2])>=8192
    cfg=dict(status='B2_DINO_ONLY_ACTUAL_PIXELS_REGISTERED',phase='B2',
        plan=stable_pin(root/'plan.json'),all_states=nc['logical_pixels'],cached_pairs=stable_pin(root/'cached_pairs.json'),
        source_references=stable_pin(root/'source_references.json'),metric_config=oldreg['metric_config'],
        cache_metric_compatibility=stable_pin(root/'cache_metric_compatibility.json'),
        prior_metadata=stable_pin(OLD_O/'a2_metrics_v1/metric_identity.json'),
        prior_reference_registration=stable_pin(OLD_O/'a2b_metrics_v1/registration.json'),
        source=stable_pin(old.E/'cal100_dino.py'),code=r['code']+[stable_pin(old.E/'cal100_dino.py')],
        native_wait=stable_pin(npth),native_registration=stable_pin(O/'B2/registration.json'),
        expected_runtime=load(r['model_routes'])['native_qualification']['numeric_settings']['runtime'],
        new_DINO_pairs=len(new),cached_DINO_pairs=len(cached),unique_actual_pixel_pairs=len(pairs),
        source_reference_reuse=len(groups),reference_forward_new=0,source_images=1000,logical_frames=27000,
        gpu_index=gpu,gpu_pci=vals[0].strip(),gpu_uuid=vals[1].strip(),execution_host=os.uname().nodename,
        max_seconds=21600,bootstrap_calls=0,source_forward_calls=0,additional_candidates=0)
    put(O/'B2/score_registration.json',cfg)
    print(json.dumps(dict(registration=stable_pin(O/'B2/score_registration.json'),new_pairs=len(new),reuse=len(cached),new_reference_forward=0)),flush=True)

def run(expected):
    p=O/'B2/score_registration.json';assert sha(p)==expected;r=read(p)
    assert r['execution_host']==os.uname().nodename
    for d in r['code']:verify(d)
    for d in load(r['cache_metric_compatibility'])['evidence']:verify(d)
    root=L/'controls/B2_dino';root.mkdir(parents=True)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(r['gpu_index']),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None)
    import fcntl
    with (old.L/'adaptive_execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wait_child(root,[str(PY),'-B',str(E/'dino_worker.py'),expected],env,r['max_seconds']+300)

def select(expected):
    p=O/'B2/score_registration.json';assert sha(p)==expected;r=read(p)
    for d in r['code']:verify(d)
    load(r['cache_metric_compatibility'])
    w,launch=check_wait(L/'controls/B2_dino/actual_wait.json',True)
    root=L/'B2/dino';done=read(root/'worker/completion.json')
    assert done['status']=='NEW_DINO_ONLY_COMPLETE' and done['ledger']['unresolved']==0
    assert not(root/'worker/failure.json').exists()
    scores={}
    for x in load(r['cached_pairs'])+load(done['scores']):
        k=x['source_id'],x['image_float_sha256']
        if k in scores:assert scores[k]==x['dino']
        scores[k]=x['dino']
    logical=load(r['all_states']);scored=[]
    for x in logical:
        v=scores[x['source_id'],x['image_float_sha256']]
        scored.append({k:x[k] for k in ('key','codec','policy_id','source_id','source_index','snr_db','noise_seed','image_float_sha256','receiver_status','accepted_wrong_post_exit')}|dict(dino=v))
    assert len(scores)==r['unique_actual_pixel_pairs'] and len(scored)==27000
    rows,means=source_means(scored)
    csvwrite(root/'logical_DINO.csv',scored);csvwrite(root/'source_noise_means.csv',rows)
    put(root/'candidate_means.json',means)
    put(O/'B2/score_completion.json',dict(status='COMPLETE',logical_frames=27000,source_count=1000,noise_per_source=3,
        metric='dinov2_vitl14_cosine',new_DINO_pairs=r['new_DINO_pairs'],exact_pair_reuse=r['cached_DINO_pairs'],
        ledger=done['ledger'],source_forward_calls=0,bootstrap_calls=0,
        actual_wait=stable_pin(L/'controls/B2_dino/actual_wait.json'),worker_completion=stable_pin(root/'worker/completion.json'),
        metric_identity=done['metric_identity'],outputs=[stable_pin(root/n) for n in ('logical_DINO.csv','source_noise_means.csv','candidate_means.json')]))
    put(O/'B2/selection.json',dict(schema='N2048_B2_DINO_FOUR_POLICIES_V1',status='FOUR_POLICIES_FROZEN',metric='dinov2_vitl14_cosine',
        candidate_means=means,rows=transport_plan.select_four(means),no_additional_candidates=True,calibration_bootstrap_calls=0,
        B2_registration=stable_pin(TO/'B2/registration.json'),B2_transport_completion=stable_pin(TO/'B2/transport_completion.json'),
        source_evidence=r['source_references'],metric_evidence=done['metric_identity'],selection_evidence=stable_pin(root/'candidate_means.json'),
        score_completion=stable_pin(O/'B2/score_completion.json'),score_actual_wait=stable_pin(L/'controls/B2_dino/actual_wait.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['prepare','run','select']);ap.add_argument('--expected-registration',required=True);ap.add_argument('--gpu',type=int)
    a=ap.parse_args()
    try:
        if a.role=='prepare':prepare(a.expected_registration,a.gpu)
        else:globals()[a.role](a.expected_registration)
    except BaseException:fail(L/'B2/dino');raise
