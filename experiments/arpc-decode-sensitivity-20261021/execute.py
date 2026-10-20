"""One bounded child per explicit stage; durable real wait and failure records."""
from common import *
import argparse,subprocess,fcntl,sys,traceback,signal

def main(role,expected):
 regpath=O/'registration.json';assert sha(regpath)==expected
 reg=read(regpath)
 for d in reg['source_files']:checked(d)
 ctrl=L/(role+'_control');ctrl.mkdir()
 save(ctrl/'supervisor.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv,registration_sha256=expected))
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='1',
   PYTHONNOUSERSITE='1',PYTHONPATH=str(OLDL/'environment_v1'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 lock=(OLDL/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 glock=(OLDL/'gpu_GPU-f5047136-08e5-9416-3c86-87786d0b355e.lock').open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 command=[sys.executable,'-B',str(E/'rx.py'),'rx','--public',str(L/'public.json')]
 assert role=='rx' and os.uname().nodename==reg['host']
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 row=next(r.split(', ') for r in gpu.splitlines() if r.startswith('1,'))
 assert row[1]==reg['resource']['gpu']['uuid'] and int(row[2])>=65536 and int(row[3])<=25
 save(ctrl/'admission.json',dict(gpu=row,unix=time.time(),host=os.uname().nodename))
 start=time.monotonic();timed=False;killed=False;terminated=False
 with (ctrl/'stdout.txt').open('xb') as out,(ctrl/'stderr.txt').open('xb') as err:
  p=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
  ident=process(p.pid)
  save(ctrl/'launch.json',dict(process=ident,parent_process=process(os.getpid()),argv=command))
  try:rc=p.wait(timeout=reg['max_rx_seconds'])
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(p.pid,signal.SIGTERM)
   try:rc=p.wait(timeout=30)
   except subprocess.TimeoutExpired:killed=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
 save(ctrl/'actual_wait.json',dict(returncode=int(rc),process=ident,seconds=time.monotonic()-start,
     timed_out=timed,terminated=terminated,killed=killed,metadata_error=None))
 if rc or timed:save(ctrl/'failure.json',dict(returncode=rc,automatic_retry=False));raise RuntimeError('RX failed; preserve all calls')
 assert gone(ident);done=read(L/'rx/completion.json');assert done['counts']==dict(model_load=1,rx_render=592)
 save(ctrl/'completion.json',dict(status='COMPLETE',rx_completion=desc(L/'rx/completion.json'),wait=desc(ctrl/'actual_wait.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role',choices=['rx']);p.add_argument('--registration',required=True)
 a=p.parse_args();main(a.role,a.registration)
