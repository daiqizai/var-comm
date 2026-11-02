"""Qualification-gated, single-use paired-prefill timing continuation."""
from pathlib import Path
import os, sys, subprocess, time, traceback
from run_once import read, save, sha, ident, same_process

def pin(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def main(path, expected):
    assert sha(path)==expected
    r=read(path);root=Path(r['root']);control=root/'chain';control.mkdir(exist_ok=False)
    save(control/'owner.json',dict(owner=ident(os.getpid()),registration=pin(path)))
    used=0.;stages=[]
    try:
        for phase in ['tx','rx']:
            if phase!='qualification':assert read(root/'qualification/qualification.json')['status']=='PASS'
            available=min(r['GPU_remaining_seconds']-used,r['deadline_unix']-time.time()-r['closure_reserve_seconds'])
            assert available>=120, 'NO_REMAINING_REGISTERED_BUDGET'
            worker=dict(r['worker_template'],phase=phase,output=str(root/phase),seconds=min(available-30,r['phase_seconds'][phase]))
            if phase!='qualification':worker['qualification']=pin(root/'qualification/qualification.json')
            if phase=='rx':worker['received_symbols']=pin(root/'tx/received_symbols.json')
            wp=Path(r['registration_dir'])/(phase+'.json');save(wp,worker)
            q=dict(study_root=str(root/(phase+'_queue')),gpu_uuid=r['NVML_UUID'],deadline_unix=r['deadline_unix'],
                   GPU_device_seconds=available,CPU_process_wall_seconds=0,worker_seconds=worker['seconds'],
                   jobs=[dict(name=phase,request=str(wp),request_sha256=sha(wp),python=r['PY'],gpu=True)],
                   worker=r['worker'],environment=r['environment'])
            # run_once locates completion relative to its queue root. The
            # scientific output is the same path through this symlink.
            qr=Path(q['study_root']);qr.mkdir();(qr/phase).symlink_to(root/phase,target_is_directory=True)
            qp=Path(r['registration_dir'])/(phase+'_queue.json');save(qp,q)
            start=time.time()
            with(control/(phase+'.log')).open('xb')as f:
                child=subprocess.Popen([r['CPY'],'-B',r['queue_worker'],str(qp),sha(qp)],
                    cwd=r['R'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                identity=ident(child.pid);save(control/(phase+'_child.json'),identity);rc=child.wait()
            wait=dict(returncode=rc,child=identity,process_gone=not same_process(identity),actual_wait=True,elapsed=time.time()-start)
            save(control/(phase+'_actual_wait.json'),wait)
            result=read(qr/'queue_control/completion.json');used+=result['GPU_device_seconds']
            stages.append(dict(phase=phase,wait=wait,result=result))
            assert rc==0 and result['status']=='COMPLETE', 'PRESERVED_STAGE_FAILURE_NO_RETRY'
        save(control/'completion.json',dict(status='COMPLETE',stages=stages,GPU_device_seconds=used,
             aggregate_GPU_device_seconds=r['prior_GPU_seconds']+used,aggregate_CPU_seconds=r['prior_CPU_seconds']))
    except BaseException:
        save(control/'failure.json',dict(traceback=traceback.format_exc(),stages=stages,GPU_device_seconds=used,
             aggregate_GPU_device_seconds=r['prior_GPU_seconds']+used,automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
