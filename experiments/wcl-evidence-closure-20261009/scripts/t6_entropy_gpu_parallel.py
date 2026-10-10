"""Two owned N2048 source shards, exact original-state qualification and real waits.

This independent wrapper preserves the original numerical flags and common visual
lock; workers own disjoint source assignments and retain original FP32 flags.
"""
from __future__ import annotations
import argparse,importlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import t6_entropy_calibrate as f
h=f.h

def assignments(r):return [list(range(k,r["source_count"],2))for k in range(2)]

def cohort_module(r):return h.load(r['gpu_cohort_runtime'],'_wcl_t6_entropy_cohort',r['source_bindings'])

def validate_config(r,rh,c):
    h.require(r['gpu_workers']==2 and c['request_sha256']==rh and c['worker_id']in(0,1),'Registered two-worker cohort')
    h.require(c['stage']=='production'and c['source_indices']==assignments(r)[c['worker_id']],'Fixed disjoint source assignment')
    session=Path(c['session']);h.require(session.parent==Path(r['out'])/'gpu_cohorts','Cohort output location')
    h.require(Path(c['cohort_manifest'])==session/'cohort.json'and Path(c['stop_file'])==session/'STOP','Cohort control locations')
    return c

def admit(r,rh,c):
    validate_config(r,rh,c);cohort=cohort_module(r).await_cohort(c,stopped=lambda:h.STOP)
    # Reuse the original process-local admission implementation. All thermal
    # checks, deterministic flags and actual model code remain in place.
    old=Path(r['root'])/'experiments/rx-posterior-step1-20260929';sys.path.insert(0,str(old))
    environment=importlib.import_module('run_preflight')
    h.require(Path(environment.__file__).resolve()==old/'run_preflight.py','Original preflight module differs')
    original=importlib.import_module('rx_v3_common');probe=original.b
    runtime=importlib.import_module('latent_enhancement.runtime');cohort.install(probe,runtime)
    return cohort,probe

def wait_for(path,r,began,cohort,session):
    while not Path(path).exists():
        h.guard(r,began);h.require(not(Path(session)/'STOP').exists(),'GPU cohort stopped')
        cohort.require_available();time.sleep(.2)

def qualify(r,rh,native,source,scorer,c,cohort):
    import numpy as np
    began=time.monotonic();session=Path(c['session']);wid=c['worker_id'];i=wid;rec=r['old_records'][i]
    # Every actual N2048 cohort keeps the same numerical identity.
    h.save(session/f'ready_{wid}.json',dict(request_sha256=rh,worker_id=wid,pid=os.getpid(),flags=native.flags))
    wait_for(session/'begin_qualification.json',r,began,cohort,session)
    result=Path(r['out'])/'gpu_parallel_qualification'/f'{i:04d}.json'
    with h.lock(result.with_suffix('.lock')):
        if result.exists():
            value=h.read(result);h.require(value['request_sha256']==rh and value['status']=='PASS','Prior parallel qualification changed')
        else:
            frame=next(x for x in h.old_rows(rec)if not x['gray']);key=h.state_key(frame);row=frame
            archives={};cached=dict(image_path=row['image_archive'],image_key='images',image_slot=row['image_slot'],
                image_file_sha256=rec['old_image_archives'][row['image_archive']],image_sha256=row['image_sha256'])
            expected=f.raw.checked_image(cached,archives)
            reservation=result.with_suffix('.reserved.json')
            h.require(not reservation.exists(),'Unresolved parallel replay; no automatic additional call')
            h.save(reservation,dict(request_sha256=rh,source_index=i,received_state_sha256=key,new_VAR_calls=1,new_metric_images=1))
            _,pixels=h.source_assets(rec);sr=h.score_record(rec)
            scorer.prepare_source(sr,pixels.astype(np.float32)/np.float32(255),i)
            native.torch.cuda.synchronize();compute_start=time.time()
            image=h.render_state(native,source,frame['receiver_state']);metrics=scorer(sr,[image])[0]
            native.torch.cuda.synchronize();compute_end=time.time()
            h.require(np.array_equal(image,expected)and h.image_sha(image)==row['image_sha256'],'Concurrent replay differs from exact original float32 RGB')
            diffs={m:abs(float(metrics[m])-float(row[m]))for m in h.METRICS}
            h.require(all(diffs[m]<=r['qualification']['tolerances'][m]for m in h.METRICS),'Concurrent metric replay differs')
            value=dict(status='PASS',request_sha256=rh,source_index=i,received_state_sha256=key,image_exact=True,image_sha256=h.image_sha(image),
                metric_differences=diffs,numerical_runtime=native.flags,new_VAR_calls=1,new_metric_images=1,cohort_manifest_sha256=cohort.manifest_sha256,
                compute_start_unix=compute_start,compute_end_unix=compute_end)
            h.save(result,value)
    h.save(session/f'qualified_{wid}.json',dict(request_sha256=rh,result=h.desc(result)))
    wait_for(session/'begin_production.json',r,began,cohort,session)

def gpu_pids():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return {int(x.strip())for x in raw.splitlines()if x.strip()}

def terminate_children(children):
    for child in children:
        if child.poll()is None:child.terminate()
    until=time.monotonic()+20
    while any(child.poll()is None for child in children)and time.monotonic()<until:time.sleep(.2)
    for child in children:
        if child.poll()is None:child.kill()
    for child in children:child.wait()

def owner(path,session_name):
    r,rh=f.registered(path);out=Path(r['out']);began=time.monotonic()
    h.require(r['gpu_workers']==2 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit registered GPU0 two-worker mode')
    h.require(session_name and all(x.isalnum()or x in'_-.'for x in session_name)and session_name not in('.','..'),'Simple new cohort session name')
    session=out/'gpu_cohorts'/session_name;h.require(not session.exists(),'Use a fresh session name; scientific source checkpoints resume')
    for i in range(r["source_count"]):
        cp=h.read(out/'cpu_sources'/f'{i:04d}.json');h.require(cp['request_sha256']==rh and len(cp['frames'])==len(r['schedule'])*len(r['noise_seeds']),'All registered full CPU sources required before GPU')
    for sig in[signal.SIGINT,signal.SIGTERM]:signal.signal(sig,h.stop)
    children=[];logs=[]
    with h.lock(r['visual_config']['visual_lock']),h.lock(out/'gpu.lock'):
        h.require(not gpu_pids(),'GPU belongs to an existing process')
        session.mkdir(parents=True);cohort=cohort_module(r)
        target=out/'metric_qualification'/'metric_batch8_qualification.json';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(r['visual_config']['metric_batch_qualification'],target)
        h.require(h.sha(target)==r['input_bindings'][r['visual_config']['metric_batch_qualification']],'Existing metric qualification changed')
        configs=[]
        for wid,indices in enumerate(assignments(r)):
            c=dict(request_sha256=rh,stage='production',worker_id=wid,source_indices=indices,session=str(session),cohort_manifest=str(session/'cohort.json'),stop_file=str(session/'STOP'))
            p=session/f'worker_{wid}.json';h.save(p,c);configs.append(p)
        try:
            for wid,p in enumerate(configs):
                log=(session/f'worker_{wid}.log').open('x');logs.append(log)
                child=subprocess.Popen([r['python_gpu'],'-u',str(Path(__file__).absolute()),'worker','--request',str(Path(path).absolute()),'--config',str(p)],
                    cwd=r['root'],env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT);children.append(child)
            workers=[dict(cohort.process_identity(p.pid),worker_id=wid,source_indices=assignments(r)[wid])for wid,p in enumerate(children)]
            manifest=dict(schema='M1_GPU_COHORT_V1',status='READY',stage='production',owner=cohort.process_identity(os.getpid()),workers=workers)
            h.save(session/'cohort.json',manifest)
            phase='loading';lastcheck=0.
            while any(p.poll()is None for p in children):
                h.guard(r,began)
                h.require(all(p.poll()in(None,0)for p in children),'Owned GPU child failed; stop peers and retain checkpoints')
                if time.monotonic()-lastcheck>=2:
                    observed=gpu_pids();h.require(observed<={p.pid for p in children},'Unregistered GPU process appeared')
                    for row in workers:
                        if row['pid']in observed:cohort.assert_process(row)
                    lastcheck=time.monotonic()
                if phase=='loading'and all((session/f'ready_{i}.json').exists()for i in range(2)):
                    h.save(session/'begin_qualification.json',dict(request_sha256=rh,cohort_manifest_sha256=h.sha(session/'cohort.json')));phase='qualification'
                if phase=='qualification'and all((session/f'qualified_{i}.json').exists()for i in range(2)):
                    checks=[h.checked(h.read(session/f'qualified_{i}.json')['result'])for i in range(2)]
                    h.require(all(x['status']=='PASS'and x['request_sha256']==rh for x in checks),'Both exact concurrent source replays must pass')
                    h.require(checks[0]['cohort_manifest_sha256']==checks[1]['cohort_manifest_sha256']and max(x['compute_start_unix']for x in checks)<min(x['compute_end_unix']for x in checks),'Two qualification compute windows must overlap in the same cohort')
                    h.save(session/'begin_production.json',dict(request_sha256=rh,qualification_sources=[0,1]));phase='production'
                time.sleep(.5)
            codes=[p.wait()for p in children]
            h.require(phase=='production'and codes==[0,0],'Two-worker run incomplete')
            for i in range(r["source_count"]):h.require(h.read(out/'gpu_sources'/f'{i:04d}.json')['request_sha256']==rh,'Missing full GPU source')
            h.save(session/'completion.json',dict(status='T6_ENTROPY_ALL_GPU_SOURCES_COMPLETE',request_sha256=rh,source_count=r['source_count'],actual_children_waited=True,worker_exit_codes=codes))
        except BaseException as error:
            if not(session/'STOP').exists():h.save(session/'STOP',dict(reason=repr(error)))
            terminate_children(children)
            h.save(session/'failure.json',dict(status='STOPPED_REQUIRES_REVIEW',request_sha256=rh,error=repr(error)));raise
        finally:
            for log in logs:log.close()

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('owner');a.add_argument('--request',required=True);a.add_argument('--session',required=True)
    a=sub.add_parser('worker');a.add_argument('--request',required=True);a.add_argument('--config',required=True)
    a=p.parse_args()
    if a.command=='owner':owner(a.request,a.session)
    else:f.gpu_worker(a.request,h.read(a.config))
if __name__=='__main__':main()
