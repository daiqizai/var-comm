"""Once-only serial EMAIN chain with actual child waits and no failure retries."""
import argparse
import fcntl
import signal
import subprocess
import traceback
from common import *

def main(path):
    r=registration(path);ART.mkdir(parents=True,exist_ok=True)
    lock=(ART/'owner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (ART/'owner_launch.json').exists(),'Owner already attempted; preserve scene')
    write(ART/'owner_launch.json',dict(pid=os.getpid(),registration=descriptor(path),started_unix=time.time()))
    statuses=[];child=None
    try:
        for stage in ('source','phy','visual','score'):
            require(not (ART/stage).exists(),'Previously attempted stage requires reviewed recovery: '+stage)
            env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_DEVICE_ORDER='PCI_BUS_ID')
            n='2' if stage=='phy' else '6'
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=n
            env['CUDA_VISIBLE_DEVICES']='' if stage=='phy' else str(r['target_gpu'])
            env['TMPDIR']=str(ART/'cache');(ART/'cache').mkdir(exist_ok=True)
            if stage!='phy':
                gate=ART/('admission_'+stage);gate.mkdir();device,g=resource_device(r,gate)
                # Resource WAIT before models/packets is not a scientific retry.
                while True:
                    require(time.time()<r['deadline_unix'] and not (OUT/'STOP').exists(),'Owner waiting deadline/STOP')
                    try:
                        snapshot,admission=device.resource_snapshot(g.helper(),prelaunch=True)
                        status(OUT/'resource_wait.json',dict(stage=stage,status='READY',snapshot=snapshot));break
                    except Exception as e:
                        if type(e).__name__!='ResourceBusy':raise
                        status(OUT/'resource_wait.json',dict(stage=stage,status='WAIT',reason=str(e),observed_unix=time.time()));time.sleep(30)
            py=RT/'envs'/('frozen_ldpc_py311' if stage=='phy' else 'frozen_visual_py310')/'bin/python'
            command=[str(py),str(Path(__file__).with_name(stage+'.py')),'--registration',str(path),'--parent',str(os.getpid())]
            status(OUT/'status.json',dict(status='RUNNING',stage=stage,completed_stages=statuses,owner_pid=os.getpid()))
            cpus=sorted(os.sched_getaffinity(0))[:2]
            start=time.time()
            with (ART/(stage+'.stdout.log')).open('xb') as stdout,(ART/(stage+'.stderr.log')).open('xb') as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,cwd=ROOT,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
                code=child.wait()
            receipt=dict(actual_wait=True,actual_child_waited=True,returncode=code,exit_code=code,pid=child.pid,stage=stage,elapsed_seconds=time.time()-start)
            child=None
            if (ART/stage).exists():write(ART/stage/'actual_wait.json',receipt)
            write(ART/(stage+'_actual_wait.json'),receipt)
            require(code==0,'Actual stage failure: '+stage)
            completion=read(ART/stage/'completion.json');statuses.append(dict(stage=stage,completion=descriptor(ART/stage/'completion.json'),actual_wait=receipt))
        write(OUT/'scoring_completion.json',dict(status='COMPLETE_EMAIN4500_SCORING',logical_frames=4500,stages=statuses,
            frame_rows=descriptor(OUT/'frame_rows.csv'),source_streams=descriptor(OUT/'source_streams.csv'),statistics_completed=False))
        write(OUT/'actual_wait_scoring.json',statuses[-1]['actual_wait'])
        write(ART/'owner_completion.json',dict(status='COMPLETE_EMAIN4500_SCORING',stages=statuses,all_actual_exit_zero=True))
        status(OUT/'status.json',dict(status='COMPLETE_EMAIN4500_SCORING_STATISTICS_PENDING',completed_stages=statuses))
    except BaseException as e:
        if child is not None and child.poll() is None:
            child.terminate();code=child.wait();write(ART/'interrupted_actual_wait.json',dict(actual_wait=True,returncode=code,pid=child.pid))
        write(ART/'owner_failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),completed_stages=statuses))
        status(OUT/'status.json',dict(status='STOPPED_NO_RETRY',error=repr(e),completed_stages=statuses));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);a=p.parse_args();main(Path(a.registration))
