"""One finite registered stage with durable real wait, private group and no retry."""
from nc_common import *
import subprocess,signal,traceback,fcntl
def main(role,expected):
 assert role in ('phy','receiver','metrics','delivery')
 r=registration('link',expected)
 C=T/(role+'_control');C.mkdir();parent=pid_identity(os.getpid());timeout=r.get('max_seconds',7200)+60
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=r['cuda_visible_devices'] if role in ('receiver','metrics') else '')
 py=RT/'envs'/('frozen_visual_py310' if role in ('source','receiver','metrics') else 'frozen_ldpc_py311')/'bin/python'
 argv=[str(py),'-B','-u',str(E/('nc_'+role+'.py')),expected];save(C/'prepared.json',dict(parent_process=parent,argv=argv,timeout=timeout,automatic_retry=False))
 locks=[];start=time.monotonic();ident=None;timed=terminated=killed=False
 try:
  for p in [T/'execution.lock',B/'t1b_m10_calibration_20261016_v1/execution.lock']:
   f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  with(C/'stdout.txt').open('xb') as so,(C/'stderr.txt').open('xb') as se:
   child=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True);ident=pid_identity(child.pid)
   save(C/'launch.json',dict(process=ident,parent_process=parent,argv=argv,started_unix=time.time()))
   try:rc=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
    try:rc=child.wait(timeout=30)
    except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=30)
  save(C/'actual_wait.json',dict(actual_wait=True,returncode=rc,process=ident,parent_process=parent,argv=argv,elapsed_seconds=time.monotonic()-start,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(C/'stdout.txt'),stderr=pin(C/'stderr.txt')))
  assert rc==0 and not timed and gone(ident)
 except BaseException:save(C/'failure.json',dict(traceback=traceback.format_exc(),process=ident,automatic_retry=False));raise
 finally:
  for f in locks:f.close()
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
