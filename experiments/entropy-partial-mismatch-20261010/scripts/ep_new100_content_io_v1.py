"""Bounded local file/CPU owner support. No image, model or network imports."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu')
RT=BASE/'var_comm_runtime_20261010'
ENGINEERING=RT/'envs/engineering-py312-v1/bin/python'
FROZEN=RT/'envs/frozen_visual_py310/bin/python'
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def inside(path,base=BASE):
    p=Path(path);b=Path(base).resolve()
    require(p.is_absolute() and p!=b and p.resolve().is_relative_to(b) and
        not any(x.is_symlink() for x in (p,*p.parents)),'Literal personal namespace path required')
    return p

def pin(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def bytes_checked(desc,maximum=64<<20,base=BASE):
    p=inside(desc['path'],base);before=p.stat()
    require(p.is_file() and before.st_size<=maximum,'Bounded regular input required')
    raw=p.read_bytes();after=p.stat()
    require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns) and
        hashlib.sha256(raw).hexdigest()==desc['sha256'] and len(raw)==desc.get('bytes',len(raw)),'Pinned bytes changed')
    return raw

def read(desc,base=BASE,maximum=64<<20):return json.loads(bytes_checked(desc,base=base,maximum=maximum))

def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)

def guard(request,owner_pid=None):
    require(time.time()<request['deadline_unix'],'Finite stage deadline exceeded')
    require(not (BASE/'STOP').exists() and not (Path(request['out'])/'STOP').exists(),'STOP observed')
    if owner_pid is not None:require(os.getppid()==owner_pid,'Owner lost; no continued content access')

def validate_execution(r):
    require(type(r['max_seconds']) is int and 1<=r['max_seconds']<=3600 and
        type(r['deadline_unix']) in (float,int) and 0<r['deadline_unix']-time.time()<=3600,
        'At most3600 seconds, fresh absolute deadline')
    slots=r['cpu_slots'];require(len(slots)==len(set(slots))==2 and all(type(x) is int and x>=0 for x in slots),
        'Exactly two explicitly coordinated CPU slots')
    require(r['automatic_retry'] is False and r['automatic_successor'] is False,'No automatic retry or successor')
    inside(r['out']);return slots

def controlled_environment(request,out):
    if request['worker_python']==str(FROZEN):
        original=read(request['controlled_environment']);env={k:v for k,v in original.items() if isinstance(v,str)}
        old=str(RT/'qualification/h800_native_import_v1/private')
        for k,v in list(env.items()):
            if v==old or v.startswith(old+'/'):env[k]=str(out/'private')+v[len(old):]
    else:
        require(request['worker_python']==str(ENGINEERING),'Only bound engineering or frozen interpreter')
        env={k:v for k,v in os.environ.items() if k in ('LANG','LC_ALL','PATH','TMPDIR')}
    for k in ('LD_PRELOAD','LD_AUDIT','PYTHONHOME','PYTHONPATH'):env.pop(k,None)
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for k in THREADS:env[k]='1'
    for v in env.values():
        if v.startswith(str(out/'private')):inside(v).mkdir(parents=True,exist_ok=True)
    return env

def run_owner(request_pin,script,registration,success):
    """One child, exact current source pins, actual wait, durable global claim."""
    require(sys.platform.startswith('linux'),'Actual owner requires the admitted Linux host')
    r=registration(request_pin);slots=validate_execution(r);guard(r)
    require(set(slots)<=os.sched_getaffinity(0),'Coordinated CPU slots unavailable')
    out=inside(Path(r['out'])/'run');require(not out.exists(),'Owner already attempted; no replay')
    out.mkdir()
    claim=inside(RT/'controls/ep_new100_confirmation_v1'/r['claim_name'])
    save(claim,dict(request=request_pin,owner_pid=os.getpid(),attempt_started_unix=time.time(),no_retry=True))
    started=time.monotonic();child=None;waited=None
    def stop(*unused):raise InterruptedError('Owned stage interrupted')
    previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        env=controlled_environment(r,out);save(out/'controlled_environment.json',env)
        argv=[r['worker_python'],'-B','-u',str(script),'_worker','--request',request_pin['path'],
            '--request-sha256',request_pin['sha256'],'--owner-pid',str(os.getpid())]
        save(out/'intent.json',dict(argv=argv,request=request_pin,cpu_slots=slots,model_calls=0,packet_calls=0))
        def limits():
            import resource
            os.sched_setaffinity(0,set(slots));os.nice(10)
            resource.setrlimit(resource.RLIMIT_CPU,(r['max_seconds']+30,r['max_seconds']+30))
        seconds=min(r['max_seconds'],r['deadline_unix']-time.time());require(seconds>0,'Expired before launch')
        with (out/'child.log').open('xb') as log:
            child=subprocess.Popen(argv,env=env,cwd=out,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,preexec_fn=limits)
            save(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),argv=argv))
            print(json.dumps(dict(status='OWNED_CPU_CHILD_STARTED',pid=child.pid,stage=r['schema'])),flush=True)
            try:rc=child.wait(timeout=seconds);interrupted=False
            except BaseException:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                rc=child.wait();interrupted=True
        waited=dict(actual_wait=True,returncode=rc,interrupted_or_timeout=interrupted,
            elapsed_seconds=time.monotonic()-started,log_sha256=sha(out/'child.log'))
        save(out/'actual_child_wait.json',waited)
        require(rc==0 and not interrupted,'Actual child failed; no retry or successor')
        done=read(pin(out/'worker_completion.json'));require(done['status']==success and done['request_sha256']==request_pin['sha256'],
            'Expected actual worker closure required')
        result=dict(status=success,request=request_pin,actual_children_waited=True,worker_exit_codes=[0],
            actual_child_wait=waited,worker_completion=pin(out/'worker_completion.json'),results=done['results'],
            model_calls=0,packet_calls=0,automatic_successor=False)
        save(out/'completion.json',result);return result
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),actual_child_wait=waited,
            child_waited=child is not None and child.poll() is not None,returncode=None if child is None else child.returncode,
            no_retry=True,automatic_successor=False));raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
