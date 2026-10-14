from common import *
import subprocess,time
C=T/'asset_audit_resume_control_v1';assert not C.exists();C.mkdir()
python=RT/'envs/frozen_ldpc_py311/bin/python';script=E/'asset_audit_resume.py'
argv=[str(python),'-B',str(script)];started=time.monotonic()
save(C/'prepared.json',dict(argv=argv,source=pin(script),old_failed_wait=pin(T/'asset_audit_control_v1/actual_wait.json'),model_calls=0))
with (C/'stdout.txt').open('xb') as out,(C/'stderr.txt').open('xb') as err:
 p=subprocess.Popen(argv,stdout=out,stderr=err,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
 identity=pid_identity(p.pid);save(C/'launch.json',dict(argv=argv,process=identity,parent_process=pid_identity(os.getpid())))
 rc=p.wait(timeout=900)
save(C/'actual_wait.json',dict(actual_wait=True,returncode=rc,process=identity,parent_process=pid_identity(os.getpid()),timed_out=False,terminated=False,killed=False,metadata_error=None,elapsed_seconds=time.monotonic()-started,stdout=pin(C/'stdout.txt'),stderr=pin(C/'stderr.txt')))
if rc:save(C/'failure.json',dict(status='FAILED_PRESERVE_DIAGNOSE_NO_RETRY',returncode=rc))
print(canonical(dict(returncode=rc,actual_wait=pin(C/'actual_wait.json'))),flush=True)
raise SystemExit(rc)
