"""Finite CPU qualification and, after all metrics exit, one screening pass."""
import argparse,json,os,pathlib,subprocess,sys,time
E=pathlib.Path(__file__).resolve().parent.parent;CODE=E/'a2_screen_v1';sys.path.insert(0,str(E))
from a1_owner import L,desc,put,identity,gone
def closed(path):
    d=json.loads(path.read_text());assert type(d['returncode']) is int and d['returncode']==0
    assert not any(d[k] for k in ['timed_out','terminated','killed']) and d['metadata_error'] is None and gone(d['process'])
    return d
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','screen']);a=p.parse_args()
    root=L/('a2_screen_qualification_v1' if a.role=='qualify' else 'a2_screen_execution_v1')
    assert not root.exists()
    if a.role=='qualify':
        script="""import sys,unittest,json
blocked=[]
class Guard:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ['torch','tensorflow','transformers','compression','infinity']:
   blocked.append(fullname);raise AssertionError('No scientific imports in screening qualification')
sys.meta_path.insert(0,Guard())
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_screen'))
print(json.dumps({'tests':r.testsRun,'passed':r.wasSuccessful(),'blocked':blocked,'scientific_calls':0}));raise SystemExit(0 if r.wasSuccessful() and not blocked else 1)
"""
        argv=[sys.executable,'-B','-c',script];inputs=[]
    else:
        q=L/'a2_screen_qualification_v1';closed(q/'actual_wait.json')
        assert json.loads((q/'completion.json').read_text())['status']=='PASS'
        for d in json.loads((q/'launch.json').read_text())['sources']:assert desc(d['path'])==d
        D=L/'a2_metrics_dispatch_v1';M=L/'a2_metrics_v1';inputs=[]
        for f in [D/'actual_wait.json',M/'score.actual_wait.json']:closed(f);inputs.append(desc(f))
        for f in [D/'supervisor.json',D/'launch.json',M/'score.launch.json']:
            assert gone(json.loads(f.read_text())['process']);inputs.append(desc(f))
        assert not (D/'failure.json').exists() and not (M/'failure.json').exists()
        assert not list((M/'calls').glob('*.failed.json'))
        inputs.append(desc(q/'completion.json'));argv=[sys.executable,'-B',str(CODE/'screen.py')]
    root.mkdir();t=time.time();timed=terminated=killed=False
    with (root/'stdout.log').open('xb') as o,(root/'stderr.log').open('xb') as e:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(CODE))
        child=subprocess.Popen(argv,env=env,stdout=o,stderr=e,cwd=CODE);ident=identity(child.pid)
        put(root/'launch.json',{'process':ident,'parent_process':identity(os.getpid()),'argv':argv,'sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()],'inputs':inputs,'started_unix':t})
        try:rc=child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed=terminated=True;child.terminate()
            try:rc=child.wait(timeout=10)
            except subprocess.TimeoutExpired:killed=True;child.kill();rc=child.wait()
        put(root/'actual_wait.json',{'process':ident,'returncode':rc,'seconds':time.time()-t,'timed_out':timed,'terminated':terminated,'killed':killed,'metadata_error':None})
    assert type(rc) is int and rc==0 and not timed and gone(ident)
    put(root/'completion.json',{'status':'PASS','actual_wait':desc(root/'actual_wait.json'),'stdout':desc(root/'stdout.log'),'stderr':desc(root/'stderr.log'),'scientific_calls':0})
    print(json.dumps({'completion':desc(root/'completion.json'),'actual_wait':json.loads((root/'actual_wait.json').read_text())}))
if __name__=='__main__':main()
