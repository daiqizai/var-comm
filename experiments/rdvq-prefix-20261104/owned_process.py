"""Exclusive bounded process owner; records actual wait and charged resources.

Does not retry failed science, change bound source, or reset a stage deadline.
"""
from pathlib import Path
import hashlib,json,os,resource,signal,subprocess,sys,time,traceback

def proc_identity(pid):
    p=Path('/proc')/str(pid)/'stat'
    return dict(pid=pid,start_ticks=p.read_text().rsplit(')',1)[1].split()[19])

def run_role(root,role,command,env,cpus,gpus,max_seconds,bindings):
    out=Path(root)/role;out.mkdir(exist_ok=False)
    for path,sha in bindings.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    reg=dict(role=role,argv=command,cpus=cpus,gpu_devices=gpus,bindings=bindings,max_seconds=max_seconds,registered=time.time())
    (out/'registration.json').write_text(json.dumps(reg,indent=2))
    (out/'owner.json').write_text(json.dumps(dict(**proc_identity(os.getpid()),started=time.time())))
    start=time.time();usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
    def affinity():os.sched_setaffinity(0,set(cpus))
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=(out/'stdout.log').open('xb'),stderr=(out/'stderr.log').open('xb'),env=env,preexec_fn=affinity)
    def request_stop(signum,frame):
        (out/'requested_stop.json').write_text(json.dumps(dict(signal=signum,time=time.time())))
        child.terminate()
    previous={s:signal.signal(s,request_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    (out/'child.json').write_text(json.dumps(dict(**proc_identity(child.pid),started=time.time())))
    try:rc=child.wait(timeout=max_seconds)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:rc=child.wait(timeout=30)
        except subprocess.TimeoutExpired:child.kill();rc=child.wait()
        (out/'timeout.json').write_text(json.dumps(dict(reason='registered wall limit',time=time.time())))
    usage1=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.time()-start
    result=dict(returncode=rc,started=start,finished=time.time(),elapsed_seconds=elapsed,
                charged_gpu_device_hours=elapsed*len(gpus)/3600,
                cpu_process_seconds=(usage1.ru_utime+usage1.ru_stime)-(usage0.ru_utime+usage0.ru_stime),
                child_gone=not Path('/proc',str(child.pid)).exists())
    (out/'actual_wait.json').write_text(json.dumps(result,indent=2))
    for s,handler in previous.items():signal.signal(s,handler)
    if rc:raise RuntimeError(f'{role} failed with actual wait {rc}; retained, no retry')
    return result

