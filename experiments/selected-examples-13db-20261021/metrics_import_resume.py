"""Complete only the eight missing scores after a zero-model import failure."""
import sys,os,subprocess,time
from pathlib import Path
import execute as x
reg=sys.argv[1]
auth=x.read(x.O/'metrics_import_recovery_v1.json');assert auth['registration_sha256']==reg;x.checked(auth['source'])
if len(sys.argv)==3:
    x.bind(reg);x.native.paths()
    f=x.native.replace_function(x,'score',[("out=T/'metrics';out.mkdir()","out=T/'metrics_import_recovery_v1';out.mkdir()")])
    f(reg)
else:
    assert x.read(x.T/'metrics/failure.json')['completed']==0
    assert not (x.T/'metrics/model_reserved.json').exists() and not (x.T/'metrics/pairs').exists()
    d=x.T/'metrics_import_recovery_dispatch_v1';d.mkdir();start=time.monotonic()
    argv=[x.PY,'-B',str(Path(__file__)),reg,'score']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');env.pop('PYTHONPATH',None)
    with (d/'stdout.log').open('xb') as out,(d/'stderr.log').open('xb') as err:
        p=subprocess.Popen(argv,env=env,stdout=out,stderr=err,cwd=x.E,start_new_session=True);ident=x.process(p.pid)
        x.save(d/'launch.json',dict(process=ident,parent_process=x.process(os.getpid()),argv=argv));code=p.wait(timeout=600)
    x.save(d/'actual_wait.json',dict(returncode=code,child=ident,seconds=time.monotonic()-start,child_gone=x.process(p.pid) is None));assert code==0
