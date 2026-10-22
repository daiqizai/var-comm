"""Durable actual parent wait for the sole bounded scale-up owner."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from rdscale_common import *
import argparse,subprocess,signal,traceback

def main(regsha,authsha):
    assert sha(O/'registration.json')==regsha;reg=read(O/'registration.json')
    for d in reg['sources']:checked(d)
    out=L/'recovery_outer_dispatch_v1';out.mkdir(exist_ok=False)
    assert not(L/'unlaunched_training_recovery_v1').exists()
    assert sha(O/'unlaunched_training_recovery_preparation_v1.json')==authsha
    argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'unlaunched_recovery_v1/recover_unlaunched.py'),'--registration-sha256',regsha,'--authorization-sha256',authsha]
    ident=None;p=None;terminated=False;killed=False;timed_out=False;t=time.perf_counter()
    def stop_owned(*_):
        nonlocal terminated
        if ident:
            cur=process(ident['pid'])
            if cur and cur['start_ticks']==ident['start_ticks']:
                os.killpg(ident['pid'],signal.SIGTERM);terminated=True
    signal.signal(signal.SIGTERM,stop_owned);signal.signal(signal.SIGINT,stop_owned)
    try:
        with(out/'stdout.log').open('xb')as so,(out/'stderr.log').open('xb')as se:
            p=subprocess.Popen(argv,cwd=R,stdout=so,stderr=se,stdin=subprocess.DEVNULL,start_new_session=True)
            ident=process(p.pid);assert ident
            save(out/'launch.json',dict(argv=argv,process=ident,parent_process=process(os.getpid()),registration_sha256=regsha,started_unix=time.time()))
            try:rc=p.wait(timeout=max(1,reg['absolute_deadline_unix']-time.time()+300))
            except subprocess.TimeoutExpired:
                timed_out=True;stop_owned()
                try:rc=p.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    cur=process(p.pid);assert cur and cur['start_ticks']==ident['start_ticks'];os.killpg(p.pid,signal.SIGKILL);killed=True;rc=p.wait()
        save(out/'actual_wait.json',dict(returncode=rc,actual_wait=True,process=ident,parent_process=process(os.getpid()),argv=argv,elapsed_seconds=time.perf_counter()-t,terminated=terminated,killed=killed,timed_out=timed_out,metadata_error=None,stdout=pin(out/'stdout.log'),stderr=pin(out/'stderr.log')))
        if rc!=0:raise RuntimeError('Owner exit '+str(rc))
    except BaseException as e:
        if p and p.poll()is None:
            stop_owned()
            try:p.wait(timeout=180)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),owned_process=ident))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);p.add_argument('--authorization-sha256',required=True);a=p.parse_args();main(a.registration_sha256,a.authorization_sha256)
