"""Exclusive detached owner, real child wait, finite wall time; no auto retry."""
from block_common import *
import subprocess,signal,argparse,traceback
def owner(request,expected):
 assert sha(request)==expected;r=read(request);c=Path(r['control']);c.mkdir(parents=True,exist_ok=False)
 save(c/'owner.json',dict(process=ident(os.getpid()),parent=ident(os.getppid()),request=pin(request)))
 for p in r['code']:checked(p)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8');env.update(r.get('environment',{}))
 start=time.monotonic()
 with (c/'stdout.log').open('xb') as so,(c/'stderr.log').open('xb') as se:
  child=subprocess.Popen(r['argv'],cwd=R,env=env,stdout=so,stderr=se,start_new_session=True)
  process=ident(child.pid);save(c/'launch.json',dict(process=process,owner=ident(os.getpid()),argv=r['argv'],request=pin(request)))
  timed_out=False
  try:rc=child.wait(timeout=r['seconds'])
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=30)
 save(c/'actual_wait.json',dict(returncode=rc,actual_wait=True,timed_out=timed_out,seconds=time.monotonic()-start,process=process,child_gone=gone(process),owner=ident(os.getpid()),request=pin(request),stdout=pin(c/'stdout.log'),stderr=pin(c/'stderr.log')))
 if rc:save(c/'failure.json',dict(returncode=rc,timed_out=timed_out,automatic_retry=False))
 return rc
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('request');p.add_argument('expected');a=p.parse_args();sys.exit(owner(a.request,a.expected))
