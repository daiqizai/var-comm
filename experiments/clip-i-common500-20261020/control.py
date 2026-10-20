"""Persistent bounded actual child wait; never starts an existing role twice."""
from common import *
import subprocess,sys,fcntl,signal,traceback
def main(expected):
 reg=read(O/'registration.json');assert sha(O/'registration.json')==expected
 for d in reg['code']:checked(d)
 c=T/'score_control';c.mkdir()
 lock=(B/'five_method_timing_20261020_v1/timing.lock').open('r+')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 argv=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'score.py'),'--registration',expected]
 start=time.monotonic();p=None;launch={};rc=None;timed=False;terminated=False;killed=False
 try:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(reg['resource']['visible_index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
      PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
  env.pop('PYTHONPATH',None)
  with(c/'stdout.txt').open('xb')as so,(c/'stderr.txt').open('xb')as se:
   p=subprocess.Popen(argv,stdout=so,stderr=se,env=env,start_new_session=True)
   launch=dict(argv=argv,process=process(p.pid),parent_process=process(os.getpid()),started_unix=time.time())
   save(c/'launch.json',launch)
   try:rc=p.wait(timeout=reg['max_seconds']+120)
   except subprocess.TimeoutExpired:
    timed=terminated=True;os.killpg(p.pid,signal.SIGTERM)
    try:rc=p.wait(timeout=30)
    except subprocess.TimeoutExpired:killed=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=30)
  save(c/'actual_wait.json',dict(**launch,actual_wait=True,returncode=rc,timed_out=timed,terminated=terminated,killed=killed,
      metadata_error=None,seconds=time.monotonic()-start,stdout=desc(c/'stdout.txt'),stderr=desc(c/'stderr.txt')))
  assert type(rc)is int and rc==0 and not timed
 except BaseException:
  save(c/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main(sys.argv[1])
