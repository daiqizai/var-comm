"""Once-only persistent parent and actual wait for preparation or clean owner."""
import argparse,json,os,pathlib,signal,subprocess,sys,time,traceback
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from owner import E,L,RUN,REG,CODE,desc,put,identity,gone,success
def stop_children():
    live=[]
    for name in ['tx','rx','audit']:
        p=RUN/f'{name}.launch.json'
        if p.exists():
            d=json.loads(p.read_text())['process']
            if not gone(d):
                assert os.getpgid(d['pid'])==d['pid'];os.killpg(d['pid'],signal.SIGTERM);live.append(d)
    return live
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','dispatch']);p.add_argument('--expected-registration');a=p.parse_args()
    control=L/('a2_clean_prepare_control_v1' if a.role=='prepare' else 'a2_clean_dispatch_v1')
    assert not control.exists(),'Existing attempt must be inspected, never overwritten/restarted'
    if a.role=='prepare':
        assert not RUN.exists() and not REG.exists();argv=[sys.executable,'-B',str(CODE/'owner.py'),'prepare'];timeout=600
    else:
        prior=L/'a2_clean_prepare_control_v1';success(prior/'actual_wait.json')
        assert gone(json.loads((prior/'supervisor.json').read_text())['process'])
        assert (prior/'completion.json').exists() and not (prior/'failure.json').exists()
        assert desc(REG)['sha256']==a.expected_registration and not (RUN/'owner.launch.json').exists()
        argv=[sys.executable,'-B',str(CODE/'owner.py'),'run','--expected-registration',a.expected_registration];timeout=5700
    control.mkdir();put(control/'supervisor.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'source':desc(__file__),'started_unix':time.time()})
    start=time.time();timed=terminated=killed=False;metadata_error=None
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    try:
        with (control/'stdout.log').open('xb') as o,(control/'stderr.log').open('xb') as e:
            child=subprocess.Popen(argv,env=env,cwd=E,stdout=o,stderr=e,start_new_session=True)
            try:ident=identity(child.pid)
            except BaseException:
                metadata_error=traceback.format_exc();child.terminate();child.wait();raise
            put(control/'launch.json',{'process':ident,'parent_process':identity(os.getpid()),'argv':argv,'started_unix':start,'timeout_seconds':timeout})
            try:rc=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed=terminated=True;children=stop_children() if a.role=='dispatch' else []
                os.killpg(child.pid,signal.SIGTERM)
                try:rc=child.wait(timeout=30)
                except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
                for d in children:
                    if not gone(d):killed=True;os.killpg(d['pid'],signal.SIGKILL)
            put(control/'actual_wait.json',{'process':ident,'returncode':rc,'seconds':time.time()-start,'timed_out':timed,'terminated':terminated,'killed':killed,'metadata_error':metadata_error})
        assert type(rc) is int and rc==0 and not timed and gone(ident)
        put(control/'completion.json',{'status':'PASS','actual_wait':desc(control/'actual_wait.json'),'ended_unix':time.time()})
    except BaseException:
        put(control/'failure.json',{'error':traceback.format_exc(),'ended_unix':time.time(),'retry_authorized':False});raise
if __name__=='__main__':main()
