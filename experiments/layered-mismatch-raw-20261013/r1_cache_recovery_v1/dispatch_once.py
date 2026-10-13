"""Persistent real owner wait; explicit once-only recovery, not a scheduler."""
from __future__ import annotations
import argparse, os, signal, subprocess, time, traceback
import common as c
import adapter

def terminate_owned(child):
    terminated=killed=False
    if child.poll() is None:
        c.require(os.getpgid(child.pid)==child.pid,'Only the owned process group may be stopped')
        os.killpg(child.pid,signal.SIGTERM);terminated=True
        try:code=child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);killed=True;code=child.wait()
    else:code=child.wait()
    c.require(type(code) is int,'Real child wait did not return integer')
    return dict(actual_wait=True,pid=child.pid,returncode=code,terminated=terminated,killed=killed)

def run(expected):
    registration=dict(path=str(c.OUT/'registered/registration.json'),sha256=expected)
    s,_,_,_=adapter.install(registration,'dispatch');reg=s.validate_registration(registration)
    c.require(time.time()<c.DEADLINE and not c.RUN.exists() and not c.DISPATCH.exists(),'No implicit replay or deadline extension')
    import fcntl
    # Ancestor locks are also held until the actual new child has exited, so
    # unrelated recovery invocations cannot consume the same missing segment.
    with (c.ANCESTOR/'owner.lock').open('rb') as old_lock,(c.BASE/'r1_resource_dispatch_v2/dispatch.lock').open('rb') as old_dispatch:
        fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(old_dispatch,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.closed_ancestors();c.DISPATCH.mkdir()
        with (c.DISPATCH/'dispatch.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            argv=[s.PY_CPU,'-B','-u',str(c.HERE/'owner.py'),'--registration',registration['path'],'--registration-sha256',expected]
            c.write(c.DISPATCH/'launch.json',dict(process=c.identity(os.getpid()),registration=registration,argv=argv,
                deadline_unix=c.DEADLINE,automatic_retry=False,scientific_completion=False))
            child=None;waited=False;started=time.monotonic()
            try:
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
                with (c.DISPATCH/'owner.stdout.log').open('xb') as stdout,(c.DISPATCH/'owner.stderr.log').open('xb') as stderr:
                    child=subprocess.Popen(argv,cwd=c.R,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                        start_new_session=True,pass_fds=(lock.fileno(),old_lock.fileno(),old_dispatch.fileno()))
                    identity=c.identity(child.pid);c.require(identity is not None,'Real owner identity unavailable')
                    c.write(c.DISPATCH/'owner_identity.json',dict(process=identity,registration=registration))
                    timeout=False
                    try:
                        code=child.wait(timeout=max(.001,c.DEADLINE-time.time()+120));c.require(type(code) is int,'Integer wait required')
                        record=dict(actual_wait=True,pid=child.pid,returncode=code,terminated=False,killed=False)
                    except subprocess.TimeoutExpired:timeout=True;record=terminate_owned(child);code=record['returncode']
                    record.update(timed_out=timeout,identity_at_launch=identity,registration=registration,elapsed_seconds=time.monotonic()-started,
                        scientific_owner_launch_exists=(c.RUN/'owner_launch.json').exists(),automatic_retry=False)
                    c.write(c.DISPATCH/'actual_wait.json',record);waited=True
                return code
            except BaseException:
                if child is not None and not waited:c.write(c.DISPATCH/'interrupted_actual_wait.json',terminate_owned(child))
                c.write(c.DISPATCH/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration=registration,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True)
    a=p.parse_args();raise SystemExit(run(a.expected_registration))
