"""Single finite owner, one real child.wait per stage; no automatic retries."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from contract import *


def run(expected):
    r = binding(expected)
    out = T/'control'
    out.mkdir()
    save(out/'launch.json', dict(process=process(os.getpid()),
        parent_process=process(os.getppid()), argv=sys.argv, registration_sha256=expected))
    lock = (T/'timing.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    start = time.monotonic()
    try:
        for d in r['original_bindings']:
            checked(d)
        for method in METHODS:
            roles = ('tx','rx') if method in ('P100000','Swin200000') else ('tx','phy','rx')
            for role in roles:
                assert not (T/'STOP').exists()
                label = method+'_'+role
                d = out/label
                d.mkdir()
                python = CPY if role=='phy' or method=='BPG' else '/usr/bin/python3' if method=='ARPC' else PY
                argv = [python,'-B',str(E/'worker.py'),method,role,'--registration',expected]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if role=='phy' or method=='BPG' else str(r['resource']['gpu_index']),
                    CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',
                    NUMEXPR_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                env.pop('PYTHONPATH',None)
                if method=='ARPC' and role!='phy':
                    env['PYTHONPATH']=r['arpc_public']['environment']+':'+str(E)
                before=time.monotonic()
                with (d/'stdout.log').open('xb') as stdout,(d/'stderr.log').open('xb') as stderr:
                    p=subprocess.Popen(argv,env=env,cwd=E,stdout=stdout,stderr=stderr,start_new_session=True)
                    ident=process(p.pid)
                    save(d/'launch.json',dict(process=ident,parent_process=process(os.getpid()),argv=argv,environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONNOUSERSITE']}))
                    timed_out=False
                    try:
                        code=p.wait(timeout=r['stage_wall_seconds'][role]+300)
                    except subprocess.TimeoutExpired:
                        timed_out=True
                        os.killpg(p.pid,signal.SIGTERM)
                        try:code=p.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=30)
                save(d/'actual_wait.json',dict(returncode=code,seconds=time.monotonic()-before,
                    child=ident,timed_out=timed_out,child_gone=process(p.pid) is None))
                assert code==0 and not timed_out, (label,code)
                assert not (T/method/role/'failure.json').exists()
                c=read(T/method/role/'completion.json')
                assert c['registration_sha256']==expected and c['calls']==288 and c['formal']==144
        save(out/'completion.json',dict(status='COMPLETE_ALL_FIVE_METHODS',
            registration_sha256=expected,full_calls=1440,formal_calls=720,
            seconds=time.monotonic()-start,training=0,quality_metrics=0,bootstrap=0))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--registration',required=True)
    a=ap.parse_args()
    run(a.registration)


if __name__=='__main__':
    main()
