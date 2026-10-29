"""Exclusive bounded sequential owner; actual waits and historical hash audit."""
import os,sys,subprocess,time,signal,traceback
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/five-method-timing-20261020'))
from contract import *
E=R/'experiments/nestcom-figure3-4db-20261029';D=R/'results/nestcom_fig3_4db_20261029_v1';L=B/'nestcom_fig3_4db_20261029_v1'

def main(expected):
    path=D/'recovery_registration.json';assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    out=L/'recovery_owner';out.mkdir()
    save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),registration=expected))
    try:
        for method in r['methods']:
            assert not(L/'STOP').exists()
            if method!='BPG':
                s=subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
                assert s[0]==r['gpu_uuid'] and int(s[1])>=r['minimum_free_mib'] and int(s[2])<=r['maximum_utilization'],s
            stage=out/method;stage.mkdir();python=CPY if method=='BPG'else'/usr/bin/python3'if method=='ARPC'else PY
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''if method=='BPG'else str(r['gpu_index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6')
            env.pop('PYTHONPATH',None)
            if method=='ARPC':env['PYTHONPATH']=r['arpc_environment']+':'+str(R/'experiments/five-method-timing-20261020')
            argv=[python,'-B',str(E/'recover_timing.py'),method,'--registration',str(path),'--expected',expected]
            with (stage/'stdout.log').open('xb')as stdout,(stage/'stderr.log').open('xb')as stderr:
                p=subprocess.Popen(argv,env=env,cwd=E,stdout=stdout,stderr=stderr,start_new_session=True);ident=process(p.pid);beg=time.monotonic()
                save(stage/'launch.json',dict(process=ident,argv=argv))
                timeout=False
                try:code=p.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    timeout=True;os.killpg(p.pid,signal.SIGTERM)
                    try:code=p.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=30)
                save(stage/'actual_wait.json',dict(returncode=code,child=ident,child_gone=process(p.pid)is None,timed_out=timeout,seconds=time.monotonic()-beg))
            assert code==0 and not timeout
            c=read(L/('recovery_'+method)/'completion.json')
            for v in c['results']:
                old=next(x for x in r['cases']if x['method']==method and x['key']==v['key'])
                assert v['image_sha256']==old['expected_image_sha256'],('HISTORICAL_OUTPUT_MISMATCH',method,v['key'],v['image_sha256'],old['expected_image_sha256'])
                assert v['status']==old['expected_status']
            save(stage/'hash_review.json',dict(status='PASS',exact_historical_hash_matches=2))
        save(out/'completion.json',dict(status='COMPLETE',reconstructions=10,exact_historical_hash_matches=10,channel_events=0,training=0))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':main(sys.argv[1])
