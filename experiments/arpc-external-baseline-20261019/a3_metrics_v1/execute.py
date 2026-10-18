"""One durable supervisor per role, with real wait and no restart path."""
import argparse,json,os,signal,sys,time,traceback
import control as c
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','dispatch']);p.add_argument('--expected-registration');a=p.parse_args()
    root=c.L/('a3_metrics_'+a.role+'_control_v1');assert not root.exists()
    if a.role!='qualify':
        prior=c.L/('a3_metrics_'+('qualify' if a.role=='prepare' else 'prepare')+'_control_v1')
        c.success(prior/'child.actual_wait.json');assert c.gone(json.loads((prior/'supervisor.json').read_text())['process'])
    argv=[str(c.PY),'-B',str(c.CODE/'control.py'),'run' if a.role=='dispatch' else a.role]
    if a.role=='dispatch':assert c.desc(c.REG)['sha256']==a.expected_registration;argv+=['--expected-registration',a.expected_registration]
    root.mkdir();c.put(root/'supervisor.json',{'process':c.identity(os.getpid()),'parent_process':c.identity(os.getppid()),'argv':sys.argv,'source':c.desc(__file__),'started_unix':time.time()})
    c.old.RUN=root;os.chdir(c.CODE)
    try:
        c.old.run_stage('child',argv,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'),3000 if a.role=='dispatch' else 600)
        c.put(root/'completion.json',{'status':'PASS','actual_wait':c.desc(root/'child.actual_wait.json')})
    except BaseException:
        if a.role=='dispatch':
            pending=[]
            for name in ['score','statistics']:
                path=c.RUN/f'{name}.launch.json'
                if path.exists():
                    ident=json.loads(path.read_text())['process']
                    if not c.gone(ident):assert os.getpgid(ident['pid'])==ident['pid'];os.killpg(ident['pid'],signal.SIGTERM);pending.append(ident)
            if pending:
                time.sleep(5)
                for ident in pending:
                    if not c.gone(ident):os.killpg(ident['pid'],signal.SIGKILL)
        c.put(root/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
if __name__=='__main__':main()
