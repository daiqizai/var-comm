"""New adapter tests only, with a real finite wait and no model imports."""
import json,os,pathlib,subprocess,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from owner import E,L,CODE,desc,put,identity,gone
def main():
    c=L/'a2_clean_qualification_v1';assert not c.exists();c.mkdir();(c/'tmp').mkdir()
    script="""import sys,unittest,json
blocked=[]
class Guard:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ['torch','tensorflow','transformers','compression','infinity']:
   blocked.append(fullname);raise AssertionError('Synthetic qualification forbids model imports')
sys.meta_path.insert(0,Guard())
suite=unittest.defaultTestLoader.loadTestsFromName('test_clean')
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'status':'PASS' if result.wasSuccessful() and not blocked else 'FAIL','tests':result.testsRun,'blocked_model_imports':blocked,'scientific_calls':0}))
raise SystemExit(0 if result.wasSuccessful() and not blocked else 1)
"""
    argv=[sys.executable,'-B','-c',script];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(CODE)+':'+str(E),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(c/'tmp'))
    start=time.time();timed=terminated=killed=False
    with (c/'stdout.log').open('xb') as o,(c/'stderr.log').open('xb') as e:
        p=subprocess.Popen(argv,cwd=CODE,env=env,stdout=o,stderr=e);ident=identity(p.pid)
        put(c/'launch.json',{'process':ident,'parent_process':identity(os.getpid()),'argv':argv,'started_unix':start,'sources':[desc(q) for q in sorted(CODE.iterdir()) if q.is_file()]})
        try:rc=p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed=terminated=True;p.terminate()
            try:rc=p.wait(timeout=10)
            except subprocess.TimeoutExpired:killed=True;p.kill();rc=p.wait()
        put(c/'actual_wait.json',{'process':ident,'returncode':rc,'seconds':time.time()-start,'timed_out':timed,'terminated':terminated,'killed':killed,'metadata_error':None})
    assert type(rc) is int and rc==0 and not timed and gone(ident)
    result=json.loads((c/'stdout.log').read_text().strip().splitlines()[-1]);assert result['status']=='PASS' and not result['blocked_model_imports']
    put(c/'completion.json',{'status':'PASS','scientific_calls':0,'tests':result['tests'],'wait':desc(c/'actual_wait.json'),'stdout':desc(c/'stdout.log'),'stderr':desc(c/'stderr.log')})
if __name__=='__main__':main()
