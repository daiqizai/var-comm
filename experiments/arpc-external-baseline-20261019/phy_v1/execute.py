"""Persistent supervisor with finite actual wait; each role can run only once."""
import argparse,json,os,pathlib,signal,sys,time,traceback
import owner
from owner import L,CODE,RUN,REG,desc,put,identity,gone,success
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','dispatch']);p.add_argument('--expected-registration');a=p.parse_args()
    name={'qualify':'qualification','prepare':'prepare','dispatch':'dispatch'}[a.role]
    control=L/f'a2_phy_{name}_control_v1';assert not control.exists(),'Existing attempt must be diagnosed, not repeated'
    argv=[str(owner.CPY),'-B',str(CODE/'owner.py'),'run' if a.role=='dispatch' else a.role]
    if a.role=='dispatch':
        prior=L/'a2_phy_prepare_control_v1';success(prior/'child.actual_wait.json')
        assert gone(json.loads((prior/'supervisor.json').read_text())['process']) and not (prior/'failure.json').exists()
        assert desc(REG)['sha256']==a.expected_registration and not (RUN/'owner.launch.json').exists()
        argv+=['--expected-registration',a.expected_registration]
    control.mkdir();put(control/'supervisor.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'source':desc(__file__),'started_unix':time.time()})
    owner.old.RUN=control;os.chdir(CODE)
    try:
        owner.old.run_stage('child',argv,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'),930 if a.role=='dispatch' else 180)
        put(control/'completion.json',{'status':'PASS','actual_wait':desc(control/'child.actual_wait.json'),'ended_unix':time.time()})
    except BaseException:
        if a.role=='dispatch':
            # Only our registered stage process groups may be stopped by this owner.
            for stage in ['tx','rx','audit']:
                path=RUN/f'{stage}.launch.json'
                if path.exists():
                    d=json.loads(path.read_text())['process']
                    if not gone(d):
                        assert os.getpgid(d['pid'])==d['pid'];os.killpg(d['pid'],signal.SIGTERM)
        put(control/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False,'ended_unix':time.time()});raise
if __name__=='__main__':main()
