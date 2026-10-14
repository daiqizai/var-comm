"""One owned actual child and durable raw logs; no automatic stage/retry/LE2."""
from common import *
import argparse,subprocess,traceback
def run(role,python,expected,max_seconds):
 assert role in ('asset_audit','tests','analyze','pilot_driver','candidate_plan','finalize')
 target=E/(role+'.py');assert sha(target)==expected
 out=T/(role+'_control_v1');assert not out.exists();out.mkdir(parents=True)
 save(out/'prepared.json',dict(role=role,source=pin(target),python=python,max_seconds=max_seconds,automatic_retry=False))
 argv=[python,'-B',str(target)];started=time.monotonic();timed_out=terminated=killed=False
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6')
 if role!='pilot_driver':env['CUDA_VISIBLE_DEVICES']=''
 else:
  r=read(O/'registration.json');assert r['authorized_stages']==['LE0','LE1'];env.update(CUDA_VISIBLE_DEVICES=str(r['gpu_index']),CUDA_DEVICE_ORDER='PCI_BUS_ID')
  for key,sub in (('TMPDIR','tmp'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')):
   p=T/'science/runtime_cache'/sub;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
 try:
  with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
   child=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,cwd=R)
   process=pid_identity(child.pid);assert process is not None
   save(out/'launch.json',dict(process=process,parent_process=pid_identity(os.getpid()),argv=argv,started_unix=time.time(),max_seconds=max_seconds))
   try:rc=child.wait(timeout=max_seconds)
   except subprocess.TimeoutExpired:
    timed_out=terminated=True;child.terminate()
    try:rc=child.wait(timeout=30)
    except subprocess.TimeoutExpired:killed=True;child.kill();rc=child.wait(timeout=30)
  save(out/'actual_wait.json',dict(actual_wait=True,returncode=rc,process=process,parent_process=pid_identity(os.getpid()),terminated=terminated,killed=killed,timed_out=timed_out,metadata_error=None,elapsed_seconds=time.monotonic()-started,stdout=pin(out/'stdout.txt'),stderr=pin(out/'stderr.txt')))
  if rc!=0 or timed_out:
   save(out/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',returncode=rc));raise SystemExit(rc or 1)
  print(canonical(dict(status='ACTUAL_CHILD_EXIT_ZERO',role=role,actual_wait=pin(out/'actual_wait.json'))),flush=True)
 except BaseException:
  if not (out/'failure.json').exists():save(out/'failure.json',dict(status='FAILED_CONTROL',traceback=traceback.format_exc()))
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--python',required=True);p.add_argument('--sha256',required=True);p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args();run(a.role,a.python,a.sha256,a.max_seconds)
