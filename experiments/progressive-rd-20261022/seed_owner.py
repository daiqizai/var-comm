from common import *
import argparse,subprocess,signal,concurrent.futures,traceback
def run(regsha):
 assert sha(O/'seed_registration.json')==regsha;reg=read(O/'seed_registration.json')
 for d in reg['sources']:checked(d)
 assert read(O/'seed_qualification.json')['returncode']==0
 out=L/'seed_dispatch_v1';out.mkdir(exist_ok=False);jobs=[]
 assert not(L/'seed_cache_v1').exists()
 for i,row in enumerate(reg['resources']):
  gpu=subprocess.check_output(['nvidia-smi','-i',str(i),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
  assert gpu[0].strip()==row['uuid'] and int(gpu[1])>=60000 and int(gpu[2])<=10,gpu
 for i,row in enumerate(reg['resources']):
  d=out/f'shard_{i:02d}';d.mkdir();so=(d/'stdout.log').open('xb');se=(d/'stderr.log').open('xb')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
  argv=[str(PY),'-B',str(E/'seed_worker.py'),'--registration-sha256',regsha,'--shard',str(i)]
  cpus=row['cpu_affinity'];p=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=lambda cpus=cpus:os.sched_setaffinity(0,set(cpus)))
  identity=process(p.pid);assert identity
  save(d/'launch.json',dict(argv=argv,process=identity,parent_process=process(os.getpid()),started_unix=time.time(),resource=row))
  jobs.append((i,p,identity,d,so,se,time.perf_counter()))
 def wait(job):
  i,p,identity,d,so,se,t=job;terminated=False;killed=False
  try:rc=p.wait(timeout=reg['shard_walltime_seconds']+180)
  except subprocess.TimeoutExpired:
   assert process(p.pid)['start_ticks']==identity['start_ticks'];os.killpg(p.pid,signal.SIGTERM);terminated=True
   try:rc=p.wait(timeout=90)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);killed=True;rc=p.wait()
  so.close();se.close();v=dict(returncode=rc,process=identity,elapsed_seconds=time.perf_counter()-t,terminated=terminated,killed=killed,timed_out=terminated,stdout=pin(d/'stdout.log'),stderr=pin(d/'stderr.log'))
  save(d/'actual_wait.json',v);return v
 with concurrent.futures.ThreadPoolExecutor(8)as pool:results=list(pool.map(wait,jobs))
 save(out/'completion.json',dict(status='ALL_SHARDS_EXITED',actual_waits=results,all_exit0=all(x['returncode']==0 for x in results),parent_process=process(os.getpid())))
 assert all(x['returncode']==0 for x in results),'A shard failed; retain all evidence and do not retry automatically'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);a=p.parse_args()
 try:run(a.registration_sha256)
 except BaseException as e:
  d=L/'seed_dispatch_v1';d.mkdir(parents=True,exist_ok=True)
  if not(d/'failure.json').exists():save(d/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
  raise
