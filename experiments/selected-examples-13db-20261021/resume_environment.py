"""Explicit zero-science environment correction; no completed call is repeated."""
import sys,os,time,subprocess,signal,fcntl,traceback
from pathlib import Path
import execute as x
def stage(reg,method,role):
    x.bind(reg)
    f=x.native.replace_function(x,'stage',[("out=T/method/role;out.mkdir(parents=True,exist_ok=False)",
        "out=T/method/(role+'_environment_recovery_v1' if method=='NEST' and role=='phy' else role);out.mkdir(parents=True,exist_ok=False)")])
    f(reg,method,role)
def owner(reg):
    x.bind(reg);d=x.T/'environment_recovery_control_v1';d.mkdir()
    assert x.read(x.T/'dispatch/actual_wait.json')['returncode']==1
    f=x.read(x.T/'NEST/phy/failure.json');assert f['completed']==0 and "No module named 'sionna'" in f['traceback']
    assert not (x.T/'NEST/phy/cases').exists()
    lock=(x.OLD_T/'timing.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    x.save(d/'launch.json',dict(process=x.process(os.getpid()),parent_process=x.process(os.getppid()),argv=sys.argv))
    stages=[(m,'phy',x.CPY) for m in ['NEST','ARPC','BPG']]+[(m,role,x.PY) for m in ['P100000','Swin200000'] for role in ['tx','rx']]+[('ALL','assemble',x.CPY),('ALL','score',x.PY)]
    try:
        for m,role,py in stages:
            q=d/(m+'_'+role);q.mkdir();argv=[py,'-B',str(Path(__file__)),role,reg,m]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if role in ('phy','assemble') else '1',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6');env.pop('PYTHONPATH',None)
            start=time.monotonic();timed_out=killed=False
            with (q/'stdout.log').open('xb') as out,(q/'stderr.log').open('xb') as err:
                p=subprocess.Popen(argv,env=env,cwd=x.E,stdout=out,stderr=err,start_new_session=True);ident=x.process(p.pid)
                x.save(q/'launch.json',dict(process=ident,parent_process=x.process(os.getpid()),argv=argv))
                try:code=p.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                    try:code=p.wait(timeout=30)
                    except subprocess.TimeoutExpired:killed=True;os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=30)
            x.save(q/'actual_wait.json',dict(returncode=code,child=ident,seconds=time.monotonic()-start,timed_out=timed_out,killed=killed,child_gone=x.process(p.pid) is None))
            assert code==0 and not timed_out,(m,role,code)
        x.save(d/'completion.json',dict(status='COMPLETE',stages=len(stages)))
    except BaseException:x.save(d/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
def supervise(reg):
    d=x.T/'environment_recovery_dispatch_v1';d.mkdir();start=time.monotonic()
    argv=[x.CPY,'-B',str(Path(__file__)),'owner',reg]
    with (d/'stdout.log').open('xb') as out,(d/'stderr.log').open('xb') as err:
        p=subprocess.Popen(argv,cwd=x.E,stdout=out,stderr=err,start_new_session=True);ident=x.process(p.pid)
        x.save(d/'launch.json',dict(process=ident,parent_process=x.process(os.getpid()),argv=argv));code=p.wait(timeout=16500)
    x.save(d/'actual_wait.json',dict(returncode=code,child=ident,seconds=time.monotonic()-start,child_gone=x.process(p.pid) is None));assert code==0
if __name__=='__main__':
    role,reg=sys.argv[1:3]
    auth=x.read(x.O/'environment_recovery_v1.json')
    assert auth['registration_sha256']==reg
    x.checked(auth['source'])
    if role in ('phy','tx','rx'):stage(reg,sys.argv[3],role)
    elif role in ('owner','supervise'):globals()[role](reg)
    else:getattr(x,role)(reg)
