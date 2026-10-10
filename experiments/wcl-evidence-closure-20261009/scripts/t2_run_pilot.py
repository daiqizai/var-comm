"""Bounded one-shot T2 supervisor. Existing scientific checkpoints are immutable."""
import argparse, fcntl, hashlib, json, os
from pathlib import Path
import signal, subprocess, sys, time

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); os.replace(temp,path)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--request',required=True)
    args=parser.parse_args(); request=Path(args.request)
    r=json.loads(request.read_text()); out=Path(r['out']); runner=Path(__file__).with_name('t2_pilot.py')
    rh=hashlib.sha256(request.read_bytes()).hexdigest()
    check=json.loads((out/'cpu_check.json').read_text())
    assert check['request_sha256']==rh and check['status']=='CPU_RUNTIME_ALL106_LAYOUTS_VALIDATED_NO_DECODE'
    assert not (out/'supervisor_exit.json').exists(), 'Completed/failed supervisor already recorded; inspect before any resume'
    lock=(out/'supervisor.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'logs').mkdir(exist_ok=True); children=[]; handles=[]
    def terminate_owned(*_):
        for child in children:
            if child.poll() is None: child.send_signal(signal.SIGTERM)
        write(out/'supervisor_status.json',dict(status='STOP_REQUESTED',pid=os.getpid(),request_sha256=rh))
        raise SystemExit(130)
    signal.signal(signal.SIGTERM,terminate_owned);signal.signal(signal.SIGINT,terminate_owned)
    base=os.environ.copy();base['PYTHONDONTWRITEBYTECODE']='1'
    cpu=dict(base,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    try:
        for index in range(r['workers']):
            log=(out/'logs'/f'cpu_{index}.log').open('xb');handles.append(log)
            child=subprocess.Popen([r['python_cpu'],'-B',str(runner),'cpu-worker','--request',str(request),'--index',str(index)],
                cwd=r['root'],env=cpu,stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
        write(out/'supervisor_status.json',dict(status='CPU_RUNNING',pid=os.getpid(),request_sha256=rh,
            cpu_pids=[p.pid for p in children],started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        while any(p.poll() is None for p in children):
            if time.time()>r['deadline_unix']:raise RuntimeError('Registered deadline reached')
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('A CPU worker failed; inspect preserved logs and ledger')
            time.sleep(2)
        assert all(p.returncode==0 for p in children), 'CPU worker failed'
        gpu=dict(base,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log=(out/'logs/gpu.log').open('xb');handles.append(log)
        child=subprocess.Popen([r['python_gpu'],'-B',str(runner),'gpu-worker','--request',str(request)],cwd=r['root'],env=gpu,stdout=log,stderr=subprocess.STDOUT)
        children.append(child)
        write(out/'supervisor_status.json',dict(status='GPU_SCORING',pid=os.getpid(),gpu_pid=child.pid,request_sha256=rh,CPU_workers_complete=8))
        code=child.wait();assert code==0, 'GPU stage failed; no automatic retry'
        log=(out/'logs/close.log').open('xb');handles.append(log)
        closed=subprocess.run([r['python_gpu'],'-B',str(runner),'close','--request',str(request)],cwd=r['root'],env=cpu,stdout=log,stderr=subprocess.STDOUT)
        assert closed.returncode==0, 'Pilot closure failed'
        result=dict(status='T2_PILOT_COMPLETE_ONLY',request_sha256=rh,exit_code=0,full_calibration_complete=False)
    except BaseException as error:
        for child in children:
            if child.poll() is None:child.send_signal(signal.SIGTERM)
        result=dict(status='STOPPED_WITH_PRESERVED_CHECKPOINTS',request_sha256=rh,exit_code=1,error=repr(error))
    finally:
        for child in children:
            if child.poll() is None:child.wait()
        for handle in handles:handle.close()
    write(out/'supervisor_exit.json',result);write(out/'supervisor_status.json',result)
    print(json.dumps(result),flush=True)
    return result['exit_code']

if __name__=='__main__':raise SystemExit(main())
