"""CPU-only source/owner tests and separate definition-import qualification."""
import argparse,json,os,pathlib,subprocess,sys,time,traceback
from a1_owner import L,E,desc,put,identity,gone
def main():
    a=argparse.ArgumentParser();a.add_argument('role',choices=['cpu','imports']);x=a.parse_args()
    c=L/('cpu_qualification_v1' if x.role=='cpu' else 'runtime_import_v1')
    assert not c.exists(),'Already attempted, inspect instead of retry'
    if x.role=='imports':
        d=json.loads((L/'environment_preparation_v1/completion.json').read_text());assert d['status']=='ENVIRONMENT_ENGINEERING_READY'
    c.mkdir();tmp=c/'tmp';tmp.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(L/'environment_v1')+':'+str(E)+':'+str(E/'upstream'),
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(tmp),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if x.role=='cpu':argv=[sys.executable,'-B','-m','unittest','test_wire','test_native_contract','test_owner','-v']
    else:
        code="""import sys,os,json,zlib
import torch
assert not torch.cuda.is_initialized()
loads=[]
def reject(*a,**k):loads.append(str(a[:1]));raise AssertionError('No weight/model loads in definition-import qualification')
torch.load=reject
import transformers
transformers.PreTrainedModel.from_pretrained=reject
import compression.sender,compression.reciever,tools.run_infinity
from a1_native import prefix_decoder
fn,ast_sha=prefix_decoder(os.environ['ARPC_UPSTREAM'])
assert not loads and not torch.cuda.is_initialized()
print(json.dumps({'status':'PASS','weight_loads':loads,'cuda_initialized':False,'torch':torch.__version__,
 'transformers':transformers.__version__,'zlib_compile':zlib.ZLIB_VERSION,'zlib_runtime':zlib.ZLIB_RUNTIME_VERSION,'prefix_ast_sha256':ast_sha}))
"""
        argv=[sys.executable,'-B','-c',code];env['ARPC_UPSTREAM']=str(E/'upstream')
    started=time.time()
    with (c/'stdout.log').open('xb') as o,(c/'stderr.log').open('xb') as e:
        p=subprocess.Popen(argv,env=env,cwd=E,stdout=o,stderr=e);ident=identity(p.pid)
        put(c/'launch.json',{'process':ident,'parent_process':identity(os.getpid()),'argv':argv,'source_files':[desc(q) for q in sorted(E.glob('*.py'))],'started_unix':started})
        timed=False;killed=False
        try:rc=p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed=True;p.terminate()
            try:rc=p.wait(timeout=10)
            except subprocess.TimeoutExpired:killed=True;p.kill();rc=p.wait()
        put(c/'actual_wait.json',{'process':ident,'returncode':rc,'timed_out':timed,'killed':killed,'seconds':time.time()-started})
    assert rc==0 and not timed and gone(ident)
    put(c/'completion.json',{'status':'PASS','science_calls':0,'role':x.role,'wait':desc(c/'actual_wait.json'),'stdout':desc(c/'stdout.log'),'stderr':desc(c/'stderr.log')})
if __name__=='__main__':main()
