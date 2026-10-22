"""Eight disjoint inference shards; three wait for this study's owned training."""
from common import *
import argparse,subprocess,concurrent.futures,signal,traceback

def main(regsha):
    assert sha(O/'expansion_registration.json')==regsha;reg=read(O/'expansion_registration.json')
    for d in reg['sources']:checked(d)
    assert read(O/'expansion_qualification.json')['status']=='PASS'
    out=L/'expansion_dispatch_v1';out.mkdir(exist_ok=False)
    assert not(L/'expansion_cache_v1').exists()
    def shard(i):
        row=reg['resources'][i];d=out/f'shard_{i:02d}';d.mkdir()
        try:
            arm=reg['wait_for_decoder_arms'].get(str(i))
            if arm:
                p=L/'decoder_dispatch_v1'/arm/'actual_wait.json'
                save(d/'waiting_for_owned_decoder.json',dict(arm=arm,actual_wait_path=str(p),started_unix=time.time()))
                while not p.exists():
                    assert time.time()<reg['latest_worker_start_unix'],'No safe resource window remains'
                    launch=read(L/'decoder_dispatch_v1'/arm/'launch.json');current=process(launch['process']['pid'])
                    if current is None:
                        time.sleep(3);assert p.exists(),'Owned decoder disappeared without actual wait'
                    elif current['start_ticks']!=launch['process']['start_ticks']:raise RuntimeError('Decoder PID reused')
                    time.sleep(15)
                w=read(p);assert w['returncode']==0 and process(w['process']['pid'])is None
                assert not(L/'decoder_pilot_v1'/arm/'failure.json').exists()
                save(d/'dependency_closed.json',dict(actual_wait=pin(p),completion=pin(L/'decoder_pilot_v1'/arm/'completion.json')))
            assert time.time()<reg['latest_worker_start_unix']
            gpu=subprocess.check_output(['nvidia-smi','-i',str(i),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            assert gpu[0].strip()==row['uuid'] and int(gpu[1])>=60000 and int(gpu[2])<=10,gpu
            argv=[str(PY),'-B',str(E/'expansion_worker.py'),'--registration-sha256',regsha,'--shard',str(i)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
            with(d/'stdout.log').open('xb')as so,(d/'stderr.log').open('xb')as se:
                cpus=row['cpu_affinity'];t=time.perf_counter();p=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cpus)));identity=process(p.pid);assert identity
                save(d/'launch.json',dict(argv=argv,process=identity,parent_process=process(os.getpid()),resource=row,started_unix=time.time()))
                terminated=False;killed=False
                try:rc=p.wait(timeout=reg['shard_walltime_seconds']+180)
                except subprocess.TimeoutExpired:
                    assert process(p.pid)['start_ticks']==identity['start_ticks'];os.killpg(p.pid,signal.SIGTERM);terminated=True
                    try:rc=p.wait(timeout=90)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);killed=True;rc=p.wait()
            v=dict(returncode=rc,process=identity,elapsed_seconds=time.perf_counter()-t,terminated=terminated,killed=killed,timed_out=terminated,stdout=pin(d/'stdout.log'),stderr=pin(d/'stderr.log'));save(d/'actual_wait.json',v);return v
        except BaseException as e:
            save(d/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));return dict(prelaunch_or_control_failure=True,error=repr(e))
    with concurrent.futures.ThreadPoolExecutor(8)as pool:results=list(pool.map(shard,range(8)))
    save(out/'completion.json',dict(status='ALL_SHARDS_TERMINAL',results=results,all_exit0=all(x.get('returncode')==0 for x in results),parent_process=process(os.getpid())))
    assert all(x.get('returncode')==0 for x in results),'Retain failed shard; no automatic retry'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);a=p.parse_args();main(a.registration_sha256)
