"""One detached bounded child, real wait and durable process receipts."""
from common import *
import argparse, subprocess, signal, traceback
def run(regsha):
 assert sha(O/'stage0_registration.json')==regsha
 reg=read(O/'stage0_registration.json')
 for d in reg['new_sources']:checked(d)
 d=L/'benchmark_dispatch_v1';d.mkdir(parents=True,exist_ok=False)
 assert not (L/'benchmark_v1').exists()
 gpu=subprocess.run(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().split(',')
 assert gpu[0].strip()==reg['gpu_nvml_uuid'] and int(gpu[1])>=60000 and int(gpu[2])<=10,gpu
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
 argv=[str(PY),'-B',str(E/'benchmark.py'),'--registration-sha256',regsha]
 started=time.time();mono=time.perf_counter();terminated=False;killed=False
 with (d/'stdout.log').open('xb') as stdout,(d/'stderr.log').open('xb') as stderr:
  child=subprocess.Popen(argv,cwd=R,env=env,stdout=stdout,stderr=stderr,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(range(6))))
  identity=process(child.pid);assert identity
  save(d/'launch.json',dict(process=identity,parent_process=process(os.getpid()),argv=argv,registration_sha256=regsha,started_unix=started,startup_gpu=gpu))
  try:rc=child.wait(timeout=reg['walltime_seconds']+180)
  except subprocess.TimeoutExpired:
   assert process(child.pid)['start_ticks']==identity['start_ticks'];os.killpg(child.pid,signal.SIGTERM);terminated=True
   try:rc=child.wait(timeout=90)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);killed=True;rc=child.wait()
 save(d/'actual_wait.json',dict(returncode=rc,process=identity,parent_process=process(os.getpid()),elapsed_seconds=time.perf_counter()-mono,terminated=terminated,killed=killed,timed_out=terminated,ended_unix=time.time(),stdout=pin(d/'stdout.log'),stderr=pin(d/'stderr.log')))
 if rc!=0:raise RuntimeError(f'Benchmark actual exit {rc}; no automatic retry')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);a=p.parse_args()
 try:run(a.registration_sha256)
 except BaseException as e:
  d=L/'benchmark_dispatch_v1';d.mkdir(parents=True,exist_ok=True)
  if not (d/'failure.json').exists():save(d/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
  raise
