"""Finite once-only CPU delivery parent; scientific completion is a prerequisite."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, signal, subprocess, sys, time
from pathlib import Path

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013')
CPY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
MASTER='ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c'
ENTRY_SHA='f37df9e92520a1f45bdc7954dbbd6eaf5c1a7cab4721d3b5df7ec5df65c6de01'
SCHEMA='R1_RECOVERED_CPU_DELIVERY_DISPATCH_V1'
MAX_SECONDS=7200
sys.dont_write_bytecode=True

def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def pin(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def put(path,value):
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)
    with os.fdopen(os.open(path,flags,0o600),'w',encoding='utf8') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def identity(pid):
    value=(Path('/proc')/str(pid)/'stat').read_text()
    tail=value[value.rfind(')')+2:].split()
    return dict(pid=int(pid),start_ticks=int(tail[19]))

def checked_identity(value,pid):
    require(type(pid) is int and type(value.get('pid')) is int and value['pid']==pid
            and type(value.get('start_ticks')) is int and value['start_ticks']>0,
            'Exact integer PID/start identity required')
    return dict(pid=value['pid'],start_ticks=value['start_ticks'])

def absent(path):
    try:path.lstat()
    except FileNotFoundError:return
    raise FileExistsError('Existing attempt must be inspected, not replayed: '+str(path))

def load_entry(path):
    require(sha(path)==ENTRY_SHA,'Frozen delivery source changed')
    spec=importlib.util.spec_from_file_location('r1_delivery_dispatch_cpu_entry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def run(root=R,temporary=L,python=CPY,popen=subprocess.Popen,clock=time.monotonic,identity_fn=identity):
    require(sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Original Linux CPU environment with CUDA hidden required')
    entry=root/'experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_v1/deliver.py'
    registration=dict(path=str(root/'results/layered_mismatch_raw_20261013/r1_cache_recovery_v1/registered/registration.json'),sha256=MASTER)
    control=temporary/'r1_cache_recovery_delivery_dispatch_v1'
    for path in (control,temporary/'r1_cache_recovery_delivery_v1',temporary/'r1_cache_recovery_delivery_v1.claim.json',temporary/'r1_cache_recovery_delivery_v1_publication_failure.json',root/'results/layered_mismatch_raw_20261013/r1_cache_recovery_delivery_v1'):absent(path)
    module=load_entry(entry)
    original=module.load_original(root)
    module.preflight(original,root,temporary,registration)  # Explicit old failed/new successful segments only.
    require(sha(entry)==ENTRY_SHA,'Entry changed after preflight')
    control.mkdir(mode=0o700,exist_ok=False)
    import fcntl
    lock=os.open(control/'dispatch.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    argv=[python,'-B',str(entry),'--expected-registration',MASTER]
    common=dict(schema=SCHEMA,registration=registration,delivery_entry=pin(entry),argv=argv,
                owner=checked_identity(identity_fn(os.getpid()),os.getpid()),CUDA_VISIBLE_DEVICES='',scientific_calls=0,
                max_seconds=MAX_SECONDS,dispatcher=pin(Path(__file__).resolve()))
    put(control/'prepared.json',common)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):environment[name]='2'
    child=None;started=clock();timed_out=False;terminated=False;killed=False;error=None;child_id=None;returncode=None;waited=False
    def wait_child(timeout=None):
        nonlocal waited
        value=child.wait() if timeout is None else child.wait(timeout=timeout)
        require(type(value) is int,'Child wait must return an integer exit code')
        waited=True
        return value
    try:
        with (control/'stdout.txt').open('x') as stdout,(control/'stderr.txt').open('x') as stderr:
            child=popen(argv,cwd=root,env=environment,stdout=stdout,stderr=stderr,pass_fds=(lock,))
            child_id=checked_identity(identity_fn(child.pid),child.pid)
            put(control/'launch.json',dict(common,child=child_id,created_unix=time.time()))
            try:returncode=wait_child(MAX_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out=True;terminated=True;child.terminate()
                try:returncode=wait_child(15)
                except subprocess.TimeoutExpired:killed=True;child.kill();returncode=wait_child()
    except BaseException as exc:
        error=repr(exc)
        if child is not None:
            if child.poll() is None:
                terminated=True;child.terminate()
                try:returncode=wait_child(15)
                except subprocess.TimeoutExpired:killed=True;child.kill();returncode=wait_child()
            else:returncode=wait_child()
    finally:
        try:
            if child is not None:
                receipt=dict(common,child=child_id,actual_wait=waited,returncode=returncode,
                    terminated=terminated,killed=killed,timed_out=timed_out,elapsed_seconds=clock()-started,
                    stdout=pin(control/'stdout.txt'),stderr=pin(control/'stderr.txt'),metadata_error=error)
                put(control/'actual_wait.json',receipt)
            if error is not None or returncode!=0 or any((timed_out,terminated,killed)):
                put(control/'failure.json',dict(error=error,returncode=returncode,actual_wait=waited,automatic_retry=False))
        finally:os.close(lock)
    require(error is None and waited and type(returncode) is int and returncode==0 and not any((timed_out,terminated,killed)),'CPU delivery failed; original attempt retained')
    print(json.dumps(dict(status='CPU_DELIVERY_CHILD_EXIT_ZERO',control=str(control),actual_wait=pin(control/'actual_wait.json'),scientific_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-registration',required=True)
    args=parser.parse_args();require(args.expected_registration==MASTER,'Explicit master SHA required');run()
