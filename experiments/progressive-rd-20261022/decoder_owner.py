"""Three matched single-GPU pilots with actual child.wait and finite timeout."""
from common import *
import argparse,subprocess,signal,concurrent.futures,traceback

def main(regsha):
    assert sha(O/'decoder_pilot_registration.json')==regsha;reg=read(O/'decoder_pilot_registration.json')
    for d in reg['sources']:checked(d)
    assert read(O/'decoder_qualification.json')['returncode']==0
    out=L/'decoder_dispatch_v1';out.mkdir(exist_ok=False);jobs=[]
    assert not(L/'decoder_pilot_v1').exists()
    for arm,row in reg['arms'].items():
        gpu=subprocess.check_output(['nvidia-smi','-i',str(row['index']),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert gpu[0].strip()==row['uuid'] and int(gpu[1])>=60000 and int(gpu[2])<=10,gpu
    for arm,row in reg['arms'].items():
        d=out/arm;d.mkdir();so=(d/'stdout.log').open('xb');se=(d/'stderr.log').open('xb')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(row['index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
        argv=[str(PY),'-B',str(E/'decoder_pilot.py'),'--registration-sha256',regsha,'--arm',arm]
        cpus=row['cpu_affinity'];p=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=lambda cpus=cpus:os.sched_setaffinity(0,set(cpus)))
        identity=process(p.pid);assert identity
        save(d/'launch.json',dict(argv=argv,process=identity,parent_process=process(os.getpid()),started_unix=time.time(),resource=row));jobs.append((p,identity,d,so,se,time.perf_counter()))
    def wait(job):
        p,identity,d,so,se,t=job;terminated=False;killed=False
        try:rc=p.wait(timeout=reg['walltime_seconds']+180)
        except subprocess.TimeoutExpired:
            assert process(p.pid)['start_ticks']==identity['start_ticks'];os.killpg(p.pid,signal.SIGTERM);terminated=True
            try:rc=p.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);killed=True;rc=p.wait()
        so.close();se.close();v=dict(returncode=rc,process=identity,elapsed_seconds=time.perf_counter()-t,terminated=terminated,killed=killed,timed_out=terminated,stdout=pin(d/'stdout.log'),stderr=pin(d/'stderr.log'))
        save(d/'actual_wait.json',v);return v
    with concurrent.futures.ThreadPoolExecutor(3)as pool:results=list(pool.map(wait,jobs))
    save(out/'completion.json',dict(status='ALL_ARMS_EXITED',actual_waits=results,all_exit0=all(x['returncode']==0 for x in results),parent_process=process(os.getpid())))
    assert all(x['returncode']==0 for x in results),'An arm failed; retain evidence, no automatic retry'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);a=p.parse_args()
    try:main(a.registration_sha256)
    except BaseException as e:
        out=L/'decoder_dispatch_v1';out.mkdir(parents=True,exist_ok=True)
        if not(out/'failure.json').exists():save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
