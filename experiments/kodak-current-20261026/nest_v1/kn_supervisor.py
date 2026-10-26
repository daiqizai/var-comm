"""Outer actual wait for the one-stage owner; no retries or successors."""
from kn_common import *
import subprocess,traceback,signal
def main(stage,expected):
 r=registration(expected);d=L/(stage+'_initial_control');assert d.is_dir()and not(d/'launch.json').exists()
 argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'kn_owner.py'),stage,expected]
 start=time.monotonic();terminated=killed=False
 with(d/'stdout.log').open('xb')as so,(d/'stderr.log').open('xb')as se:
  c=subprocess.Popen(argv,cwd=R,stdout=so,stderr=se,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  identity=ident(c.pid);assert identity;save(d/'launch.json',dict(process=identity,parent_process=ident(os.getpid()),argv=argv,registration_sha256=expected))
  try:rc=c.wait(timeout=r['max_seconds'][stage]+360)
  except subprocess.TimeoutExpired:
   assert ident(c.pid)['start_ticks']==identity['start_ticks'];os.killpg(c.pid,signal.SIGTERM);terminated=True
   try:rc=c.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);killed=True;rc=c.wait()
 save(d/'actual_wait.json',dict(actual_wait=True,returncode=rc,process=identity,parent_process=ident(os.getpid()),elapsed_seconds=time.monotonic()-start,terminated=terminated,killed=killed,timed_out=terminated,metadata_error=None,stdout=pin(d/'stdout.log'),stderr=pin(d/'stderr.log')))
 if rc:raise RuntimeError(f'Owner actual exit {rc}; no retry')
if __name__=='__main__':
 stage,expected=sys.argv[1:]
 try:main(stage,expected)
 except BaseException:
  d=L/(stage+'_initial_control')
  if not(d/'failure.json').exists():save(d/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
  raise
