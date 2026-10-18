"""Finite CPU test/extraction owner; never runs historical experiment entries."""
import json,os,pathlib,subprocess,sys,time,traceback
E=pathlib.Path(__file__).parent.parent;sys.path.insert(0,str(E))
from a1_owner import L,desc,put,identity,gone
def main():
    c=L/'a2_reliability_extraction_v1';assert not c.exists();c.mkdir()
    put(c/'supervisor.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'source_files':[desc(p) for p in sorted(pathlib.Path(__file__).parent.glob('*.py'))],'started_unix':time.time()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    try:
        for name,argv in [('tests',[sys.executable,'-B','-m','unittest','test_extract','-v']),('extract',[sys.executable,'-B',str(pathlib.Path(__file__).parent/'extract.py')])]:
            start=time.time();timed=killed=False
            with (c/(name+'.stdout')).open('xb') as o,(c/(name+'.stderr')).open('xb') as e:
                child=subprocess.Popen(argv,cwd=pathlib.Path(__file__).parent,env=env,stdout=o,stderr=e);ident=identity(child.pid)
                put(c/(name+'.launch.json'),{'process':ident,'parent_process':identity(os.getpid()),'argv':argv,'started_unix':start})
                try:rc=child.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    timed=True;child.terminate()
                    try:rc=child.wait(timeout=10)
                    except subprocess.TimeoutExpired:killed=True;child.kill();rc=child.wait()
                put(c/(name+'.actual_wait.json'),{'process':ident,'returncode':rc,'seconds':time.time()-start,'timed_out':timed,'terminated':timed,'killed':killed,'metadata_error':None})
                assert type(rc) is int and rc==0 and not timed and gone(ident),name+' failed'
        put(c/'completion.json',{'status':'PASS','new_packets':0,'model_calls':0,'ended_unix':time.time()})
    except BaseException:
        put(c/'failure.json',{'error':traceback.format_exc(),'ended_unix':time.time(),'new_packets':0,'model_calls':0});raise
if __name__=='__main__':main()
