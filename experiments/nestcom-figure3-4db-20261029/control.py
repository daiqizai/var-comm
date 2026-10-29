"""A single registered subprocess, no automatic retries or successor launches."""
from pathlib import Path
import sys,os,subprocess,time,signal,traceback
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
from common import *
L=B/'nestcom_fig3_4db_20261029_v1'
def main(path,expected):
    assert sha(path)==expected;r=read(path);role=r['role'];o=L/(role+'_control');o.mkdir()
    save(o/'started.json',dict(process=ident(os.getpid()),registration_sha256=expected))
    try:
        for d in r['code']:checked(d)
        for d in r.get('dependencies',[]):
            v=cr(d);assert v['returncode']==0 and gone(v['process'])
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',CUDA_VISIBLE_DEVICES='1'if r['gpu']else'',CUDA_DEVICE_ORDER='PCI_BUS_ID')
        env.pop('PYTHONPATH',None)
        if r['gpu']:
            a=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
            assert a[0]=='GPU-f5047136-08e5-9416-3c86-87786d0b355e'and int(a[1])>=45000 and int(a[2])<=80,a
        argv=r['argv']+[str(path),expected];beg=time.monotonic()
        with(o/'stdout.log').open('xb')as so,(o/'stderr.log').open('xb')as se:
            p=subprocess.Popen(argv,stdout=so,stderr=se,env=env,start_new_session=True);v=ident(p.pid);save(o/'child.json',dict(process=v,argv=argv));timed_out=False
            try:rc=p.wait(timeout=r['max_seconds'])
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:rc=p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=30)
        save(o/'actual_wait.json',dict(returncode=rc,process=v,parent_process=ident(os.getpid()),child_gone=gone(v),timed_out=timed_out,seconds=time.monotonic()-beg))
        assert rc==0 and not timed_out
    except BaseException:save(o/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
