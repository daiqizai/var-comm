"""Explicit once-only stage supervisor. No scheduling, successor or blind retry."""
import argparse,signal,time,traceback
from common import *
import a1_owner as parent
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['source']);p.add_argument('role',choices=['prepare','dispatch']);p.add_argument('--expected-registration');a=p.parse_args()
    out=L/(a.stage+'_'+a.role+'_control_v1');assert not out.exists()
    argv=[sys.executable,'-B',str(E/(a.stage+'_control.py')),'prepare'if a.role=='prepare'else'run']
    if a.role=='dispatch':
        assert a.expected_registration and desc(O/'source_v1/registration.json')['sha256']==a.expected_registration
        check_wait(L/'source_prepare_control_v1/child.actual_wait.json')
        assert gone(read(L/'source_prepare_control_v1/supervisor.json')['process'])
        argv+=['--expected-registration',a.expected_registration]
    out.mkdir();parent.RUN=out
    put(out/'supervisor.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'source':desc(__file__),'started_unix':time.time()})
    try:
        parent.run_stage('child',argv,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'),8100 if a.role=='dispatch'else 900)
        put(out/'completion.json',{'status':'PASS','actual_wait':desc(out/'child.actual_wait.json')})
    except BaseException:
        # Only children whose recorded PID/start identity is still the same may
        # be terminated. Never touch unrelated processes or other users' GPUs.
        for name in ('tx','audit')if a.role=='dispatch'else():
            launch=L/'source_v1'/(name+'.launch.json')
            if launch.exists():
                ident=read(launch)['process']
                if not gone(ident):
                    assert os.getpgid(ident['pid'])==ident['pid'];os.killpg(ident['pid'],signal.SIGTERM)
        put(out/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
