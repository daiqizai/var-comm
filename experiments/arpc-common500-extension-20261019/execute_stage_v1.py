"""Explicit once-only subordinate controls, no automatic replay or successor."""
import argparse,signal,traceback,time
from common import *
import a1_owner as parent
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['link','rx','metrics']);p.add_argument('role',choices=['prepare','dispatch','close']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='close':
        from stage_support import close
        close(a.stage);return
    out=L/(a.stage+'_'+a.role+'_control_v1');assert not out.exists()
    python=str(PY)if a.stage=='metrics'else(str(CPY)if a.stage=='link'else'/usr/bin/python3')
    argv=[python,'-B',str(E/(a.stage+'_control.py')),'prepare'if a.role=='prepare'else'run']
    if a.role=='dispatch':
        assert desc(O/(a.stage+'_v1')/'registration.json')['sha256']==a.expected_registration
        check_wait(L/(a.stage+'_prepare_control_v1')/'child.actual_wait.json')
        assert gone(read(L/(a.stage+'_prepare_control_v1')/'supervisor.json')['process'])
        argv+=['--expected-registration',a.expected_registration]
    out.mkdir();parent.RUN=out
    put(out/'supervisor.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'source':desc(__file__),'started_unix':time.time()})
    timeout={'link':5400,'rx':15600,'metrics':15600}[a.stage]if a.role=='dispatch'else 1800
    try:
        parent.run_stage('child',argv,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'),timeout)
        put(out/'completion.json',{'status':'PASS','actual_wait':desc(out/'child.actual_wait.json')})
    except BaseException:
        if a.role=='dispatch':
            for name in ('tx','rx','audit','score'):
                path=L/(a.stage+'_v1')/(name+'.launch.json')
                if path.exists():
                    ident=read(path)['process']
                    if not gone(ident):
                        assert os.getpgid(ident['pid'])==ident['pid'];os.killpg(ident['pid'],signal.SIGTERM)
        put(out/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
