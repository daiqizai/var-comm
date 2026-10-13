"""One finite resource waiter for the reviewed R0 clean5400 owner.

No new scientific entry, periodic scheduler, scientific retry or model import.
The only retryable child result is a proven exit0 WAIT before any owner/stage
launch. An actual scientific owner launch consumes this once-only dispatch.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
ENTRY=Path(__file__).with_name('r0_clean.py')
ENTRY_SHA='f34c6495d0ce0988188a0946d538a46c37bc21ddc7a07595206b324ee73cebed'
REG=ROOT/'results/layered_mismatch_raw_20261013/r0_supplement_preparation/clean_registration.json'
REG_SHA='d2c8116141b1f9ee2dbfb3b0920f29c60aa437cb8fda4335cbcd7f2d538915d8'
POLL_SECONDS=300
WAIT_SECONDS=86400
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013/clean_wait_once_v1')
OUT=ROOT/'results/layered_mismatch_raw_20261013/clean_wait_once_v1'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def status(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    q.write_text(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n',encoding='utf8');os.replace(q,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def load_clean():
    require(sha(ENTRY)==ENTRY_SHA and sha(REG)==REG_SHA,'Frozen clean entry/registration changed')
    spec=importlib.util.spec_from_file_location('_clean_wait_owned_entry',ENTRY);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    r=m.registration(REG,REG_SHA)
    require(r['target_gpu']==3 and r['CPU_slots']==[178,179] and r['max_seconds']==21600,'Reviewed GPU/CPU/wall binding changed')
    return m,r

def live_clean_processes(clean):
    found=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            rec=clean.process_identity(int(p.name))
            if rec and rec['pid']!=os.getpid() and rec['state']!='Z' and str(ENTRY) in rec['argv'] and (' run ' in rec['argv'] or ' worker ' in rec['argv']):found.append(rec)
    return found

def zero_science_scene(clean,live=None):
    """Specific attempt markers, not status text or absence of GPU utilization."""
    paths=[clean.TMP/name for name in ('owner_launch.json','owner_failure.json','owner_completion.json',
        'dispatch.json','render','score','render.stdout.log','render.stderr.log','score.stdout.log','score.stderr.log',
        'render_actual_wait.json','score_actual_wait.json','interrupted_actual_wait.json')]
    paths += [clean.OUT/name for name in ('completion.json','clean_quality.csv','dispatch.json')]
    markers=[str(p) for p in paths if p.exists()]
    active=live_clean_processes(clean) if live is None else live
    return dict(zero_science_proven=not markers and not active,existing_attempt_markers=markers,active_owner_or_worker=active)

def classify_child(actual,stdout,scene,completion=None):
    """No unknown/nonzero exit or partial scene is ever classified as retryable."""
    if actual.get('actual_wait') is not True or actual.get('timed_out') or actual.get('returncode')!=0:
        return 'STOP_UNKNOWN_OR_FAILED_NO_RETRY'
    try:result=json.loads(stdout.strip())
    except (ValueError,TypeError):return 'STOP_UNRECOGNIZED_EXIT_NO_RETRY'
    if not isinstance(result,dict):return 'STOP_UNRECOGNIZED_EXIT_NO_RETRY'
    if result.get('status')=='WAIT' and result.get('model_calls')==0 and scene['zero_science_proven']:
        return 'WAIT_RACE_PROVEN_ZERO_SCIENCE'
    if result.get('status')=='COMPLETE_CLEAN5400' and completion is not None and completion.get('status')=='COMPLETE_CLEAN5400' and completion.get('all_actual_exit_zero') is True:
        return 'COMPLETE'
    return 'STOP_UNCLOSED_OR_AMBIGUOUS_NO_RETRY'

@contextlib.contextmanager
def probe_locks(clean,fcntl):
    """Waiter lock is outermost; original owner and timing locks close race gaps."""
    clean.TMP.mkdir(parents=True,exist_ok=True)
    with clean.TIMING_LOCK.open('a+') as timing,(clean.TMP/'owner.lock').open('a+') as owner:
        try:fcntl.flock(timing,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:yield dict(status='WAIT',reason='timing_exclusive_lock');return
        try:
            try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:yield dict(status='WAIT',reason='clean_owner_lock');return
            try:
                # Release only the owner lock immediately before spawning. The
                # original owner then reacquires it; the persistent waiter lock
                # and frozen owner_launch marker prevent a duplicate waiter.
                yield dict(status='LOCKED',timing=timing,owner=owner)
            finally:fcntl.flock(owner,fcntl.LOCK_UN)
        finally:fcntl.flock(timing,fcntl.LOCK_UN)

def wait_owned_child(child,timeout):
    """Wait the actual owner. Last-resort timeout kills only its new process group."""
    result=dict(actual_wait=True,pid=child.pid,timed_out=False,terminated=False,killed=False)
    try:code=child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        result.update(timed_out=True,terminated=True)
        require(os.getpgid(child.pid)==child.pid,'Owned process group identity changed')
        os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);result['killed']=True;code=child.wait()
    result['returncode']=code;return result

def dispatch(clean,r,locks,index):
    import fcntl
    require(sha(ENTRY)==ENTRY_SHA and sha(REG)==REG_SHA,'Frozen entry/registration changed while waiting')
    before=zero_science_scene(clean);require(before['zero_science_proven'],'Attempt appeared before dispatch')
    p=TMP/'attempts'/f'{index:03d}';p.mkdir(parents=True,exist_ok=False)
    command=[str(clean.PYTHON),str(ENTRY),'run','--registration',str(REG),'--expected-registration',REG_SHA,'--reviewed']
    write(p/'dispatch.json',dict(command=command,registration=pin(REG),entry=pin(ENTRY),created_unix=time.time(),
        dispatch_is_not_scientific_completion=True,precondition=before))
    fcntl.flock(locks['owner'],fcntl.LOCK_UN)
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    child=None;started=time.monotonic()
    try:
        with (p/'stdout.log').open('xb') as stdout,(p/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True,
                pass_fds=(locks['timing'].fileno(),))
            identity=clean.process_identity(child.pid)
            write(p/'child_identity.json',dict(process=identity,pid=child.pid,new_process_group=True,spawned_unix=time.time()))
            actual=wait_owned_child(child,r['max_seconds']+120)
        actual.update(elapsed_seconds=time.monotonic()-started,identity_at_launch=identity,
            process_after_wait=clean.process_identity(child.pid))
        write(p/'actual_wait.json',actual)
        scene=zero_science_scene(clean);completion=read(clean.OUT/'completion.json') if (clean.OUT/'completion.json').exists() else None
        outcome=classify_child(actual,(p/'stdout.log').read_text(encoding='utf8'),scene,completion)
        if outcome=='COMPLETE':
            require(completion['registration']['sha256']==REG_SHA and len(completion['stages'])==2,'Wrong completion registration/stages')
            for stage in completion['stages']:
                require(stage['actual_wait']['returncode']==0 and stage['actual_wait']['actual_wait'] and not stage['actual_wait'].get('timed_out'),'Stage did not actually exit0')
                clean.checked(stage['completion'])
            require(not scene['active_owner_or_worker'],'Scientific owner returned while a worker remains live')
        result=dict(status=outcome,post_wait_scene=scene,actual_wait=pin(p/'actual_wait.json'),
            stdout=pin(p/'stdout.log'),stderr=pin(p/'stderr.log'),completion=None if completion is None else pin(clean.OUT/'completion.json'))
        write(p/'decision.json',result);return result
    except BaseException:
        if child is not None and child.poll() is None:
            # This is our newly created process group, never an unrelated job.
            require(os.getpgid(child.pid)==child.pid,'Owned interruption group identity changed')
            os.killpg(child.pid,signal.SIGTERM)
            actual=wait_owned_child(child,15);write(p/'interrupted_actual_wait.json',actual)
        raise

def run(reviewed):
    require(reviewed,'Root must review this independent once-only waiter before dispatch')
    clean,r=load_clean();TMP.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (TMP/'waiter.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (TMP/'launch.json').exists() and not (TMP/'attempts').exists(),'Waiter already attempted; no implicit restart')
        scene=zero_science_scene(clean);require(scene['zero_science_proven'],'Existing scientific attempt; do not wait or dispatch')
        start=time.time();deadline=start+WAIT_SECONDS;attempt=0;poll=0
        write(TMP/'launch.json',dict(process=clean.process_identity(os.getpid()),entry=pin(__file__),clean_entry=pin(ENTRY),registration=pin(REG),
            started_unix=start,wait_deadline_unix=deadline,poll_seconds=POLL_SECONDS,wait_seconds_cap=WAIT_SECONDS,
            wait_is_not_scientific_wall_budget=True,scientific_owner_wall_cap=r['max_seconds'],periodic_scheduler=False))
        try:
            while True:
                require(not any(p.exists() for p in (OUT/'STOP',TMP/'STOP',clean.OUT/'STOP',clean.TMP/'STOP')),'Requested STOP')
                scene=zero_science_scene(clean);require(scene['zero_science_proven'],'A scientific attempt appeared; stop without dispatch')
                if time.time()>=deadline:
                    result=dict(status='EXPIRED_WAIT_NO_SCIENCE',polls=poll,owner_invocations=attempt,scientific_owner_launch_count=0)
                    write(OUT/'completion.json',result);return result
                with probe_locks(clean,fcntl) as locks:
                    if locks['status']=='WAIT':ready=locks
                    else:
                        scene=zero_science_scene(clean);require(scene['zero_science_proven'],'Attempt appeared while acquiring admission locks')
                        gate=clean.TMP/'waiter_admission';gate.mkdir(exist_ok=True)
                        ready=clean.admission(r,gate)
                        if ready['status']=='READY':
                            result=dispatch(clean,r,locks,attempt);attempt+=1
                            if result['status']=='COMPLETE':
                                write(OUT/'completion.json',dict(status='COMPLETE',clean_completion=result['completion'],actual_owner_wait=result['actual_wait'],
                                    owner_invocations=attempt,scientific_owner_launch_count=1,proven_zero_science_race_waits=attempt-1));return result
                            require(result['status']=='WAIT_RACE_PROVEN_ZERO_SCIENCE','Unknown/actual child failure; preserve scene, no retry')
                            ready=dict(status='WAIT',reason='owner_exit0_proven_zero_science_race')
                poll+=1;status(OUT/'wait_state.json',dict(status='WAIT',observed_unix=time.time(),deadline_unix=deadline,polls=poll,
                    owner_invocations=attempt,scientific_owner_launch_count=0,readiness=ready))
                time.sleep(min(POLL_SECONDS,max(0,deadline-time.time())))
        except BaseException as e:
            write(OUT/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),owner_invocations=attempt,polls=poll));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('check','run'));p.add_argument('--reviewed',action='store_true');a=p.parse_args()
    if a.action=='check':
        clean,r=load_clean();print(json.dumps(dict(status='CHECKED_NO_SCIENCE',registration=pin(REG),entry=pin(ENTRY),scene=zero_science_scene(clean),poll_seconds=POLL_SECONDS,wait_seconds=WAIT_SECONDS)))
    else:print(json.dumps(run(a.reviewed)))

if __name__=='__main__':main()
