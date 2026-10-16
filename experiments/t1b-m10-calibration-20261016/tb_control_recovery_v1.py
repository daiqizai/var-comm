"""One owner, four finite stages, durable actual waits; stops on any failure."""
from tb_common import *
import subprocess,traceback,signal,fcntl
def run(expected,recovery_expected):
 from recovery_binding import verify_recovery
 verify_recovery(recovery_expected)
 r=registration(expected);out=T/'control_recovery_v1';out.mkdir(exist_ok=False)
 lock=(out/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(out/'supervisor.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,registration=pin(O/'registration.json')))
 child=None;ident=None
 try:
  for stage in ('receiver_recovery_v1','metrics','selection'):
   registration(expected);gpu=stage in('receiver_recovery_v1','metrics');p=out/stage;p.mkdir()
   assert not(T/'STOP').exists()
   if gpu:
    val=subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert val[0].strip()==r['gpu_uuid'] and int(val[1])>=15360 and int(val[2])<=25,('GPU_ADMISSION_FAILED_NO_RETRY',val)
    save(p/'admission.json',dict(gpu_uuid=val[0].strip(),free_MiB=int(val[1]),utilization=int(val[2]),minimum_free_MiB=15360,private_allocator_GiB=12,running_minimum_free_MiB=2048,exclusive_device_required=False))
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NVIDIA_TF32_OVERRIDE='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(r['gpu_index'])if gpu else'')
   for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='6'if gpu else'2'
   python=RT/'envs'/('frozen_visual_py310'if gpu else'frozen_ldpc_py311')/'bin/python'
   filename={'receiver_recovery_v1':'tb_receiver_recovery_v1.py','metrics':'tb_metrics_recovery_v1.py','selection':'tb_select.py'}[stage]
   verify_recovery(recovery_expected)
   argv=[str(python),'-B','-u',str(E/filename),expected,recovery_expected];start=time.monotonic();timed=terminated=killed=False
   save(p/'prepared.json',dict(argv=argv,target=pin(E/filename),max_seconds=r['seconds']['receiver'if stage=='receiver_recovery_v1'else stage]+120,automatic_retry=False))
   with(p/'stdout.txt').open('xb')as so,(p/'stderr.txt').open('xb')as se:
    child=subprocess.Popen(argv,env=env,cwd=R,stdout=so,stderr=se,stdin=subprocess.DEVNULL,start_new_session=True);ident=pid_identity(child.pid);assert ident
    save(p/'launch.json',dict(process=ident,parent_process=pid_identity(os.getpid()),argv=argv,started_unix=time.time()))
    try:rc=child.wait(timeout=r['seconds']['receiver'if stage=='receiver_recovery_v1'else stage]+120)
    except subprocess.TimeoutExpired:
     timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
     try:rc=child.wait(timeout=30)
     except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=30)
   save(p/'actual_wait.json',dict(actual_wait=True,returncode=rc,process=ident,parent_process=pid_identity(os.getpid()),argv=argv,elapsed_seconds=time.monotonic()-start,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(p/'stdout.txt'),stderr=pin(p/'stderr.txt')))
   assert rc==0 and not timed and gone(ident) and not(T/stage/'failure.json').exists()
   done=read(T/stage/'completion.json');assert done['registration_sha256']==expected;child=None
  save(out/'completion.json',dict(status='T1B_SCIENCE_CLOSED_DELIVERY_PENDING',registration_sha256=expected,actual_stage_returns=[0,0,0],inherited_PHY_exit=0,recovery_registration_sha256=recovery_expected,T1C=False,automatic_followup=False))
 except BaseException:
  if child is not None and child.poll()is None:
   child.terminate()
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
  save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,child_process=ident,child_returncode=None if child is None else child.returncode));raise
 finally:lock.close()
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
