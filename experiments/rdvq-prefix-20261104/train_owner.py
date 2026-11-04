"""One registered training attempt with a long enough checkpoint-stop grace."""
from pathlib import Path
import hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from owned_process import proc_identity

def main():
    rp=Path(sys.argv[1]);r=json.loads(rp.read_text());O=Path(r['output']);O.mkdir(exist_ok=False);L=Path(r['root']);S=Path(__file__).resolve().parent
    assert r['training_seconds']<=32400 and r['reserved_gpu_hours']<=10 and r['global_batch']==16 and r['microbatch']==4
    for p,h in r['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert json.loads(Path(r['p1_gate']).read_text())['passed']
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert r['gpu']['uuid'] not in apps
    identity=subprocess.check_output(['nvidia-smi','-i',str(r['gpu']['index']),'--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True).strip().split(',')
    assert identity[0].strip()==r['gpu']['uuid'] and identity[1].strip().lower()==r['gpu']['pci'].lower()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(r['gpu']['index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONPATH=str(L/'environment/site')+':'+str(L/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184')+':'+str(S),TORCH_EXTENSIONS_DIR=str(L/'extensions_headers_v2'),CPATH=str(L/'dependencies/python310_headers/Include')+':'+str(L/'dependencies/python310_headers'))
    (O/'registration.json').write_bytes(rp.read_bytes());(O/'owner.json').write_text(json.dumps(dict(**proc_identity(os.getpid()),started=time.time())))
    start=time.time();usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
    child=subprocess.Popen([sys.executable,'-u',str(S/'train_matched.py'),'--request',str(rp)],env=env,stdin=subprocess.DEVNULL,stdout=(O/'stdout.log').open('xb'),stderr=(O/'stderr.log').open('xb'),preexec_fn=lambda:os.sched_setaffinity(0,set(r['gpu']['cpus'])))
    (O/'child.json').write_text(json.dumps(dict(**proc_identity(child.pid),started=time.time())))
    stopping=[None]
    def stop(sig,frame):
        if stopping[0] is None:
            stopping[0]=time.time();(O/'requested_stop.json').write_text(json.dumps(dict(signal=sig,time=stopping[0],grace_seconds=600)));child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while child.poll() is None:
        if (L/'STOP_P2').exists() or time.time()-start>r['training_seconds']+120:stop(signal.SIGTERM,None)
        if stopping[0] is not None and time.time()-stopping[0]>600:
            child.kill();break
        time.sleep(2)
    rc=child.wait();u=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.time()-start
    wait=dict(returncode=rc,child_gone=not Path('/proc',str(child.pid)).exists(),started=start,finished=time.time(),elapsed_seconds=elapsed,charged_gpu_device_hours=elapsed/3600,cpu_process_seconds=(u.ru_utime+u.ru_stime)-(usage0.ru_utime+usage0.ru_stime))
    (O/'actual_wait.json').write_text(json.dumps(wait,indent=2))
    if rc:raise RuntimeError(f'Unknown training failure {rc}; no automatic retry')
    assert (O/'complete.json').exists() or (O/'segment_complete.json').exists() or (O/'paused.json').exists()
if __name__=='__main__':main()
