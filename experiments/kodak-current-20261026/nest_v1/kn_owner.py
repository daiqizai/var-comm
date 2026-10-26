"""One explicitly requested stage, bounded real child.wait, no automatic retry."""
from kn_common import *
import fcntl,signal,subprocess,traceback
def run(stage,expected):
 assert stage in ('source','phy','receiver','delivery');r=registration(expected)
 if stage=='phy':closed('source',expected)
 if stage=='receiver':closed('phy',expected)
 if stage=='delivery':closed('receiver',expected)
 assert not (L/stage).exists()if stage!='delivery'else not(O/'delivery').exists()
 d=L/(stage+'_control');d.mkdir(parents=True,exist_ok=False)
 lock=(L/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6')
 gpu=None
 if stage in ('source','receiver'):
  gpu=subprocess.check_output(['nvidia-smi','-i',r['gpu']['uuid'],'--query-gpu=uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
  assert gpu[0].strip()==r['gpu']['uuid']and gpu[1].strip()==r['gpu']['pci_bus_id']and int(gpu[2])>=30000 and int(gpu[3])<=80,gpu
  env.update(CUDA_VISIBLE_DEVICES=r['gpu']['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 py=r['python']['gpu'if stage in ('source','receiver')else'cpu'];argv=[py,'-B',str(E/f'kn_{stage}.py'),expected]
 beg=time.monotonic();started=time.time();terminated=killed=False;identity=None
 with (d/'stdout.log').open('xb')as stdout,(d/'stderr.log').open('xb')as stderr:
  child=subprocess.Popen(argv,cwd=R,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  identity=ident(child.pid);assert identity
  save(d/'launch.json',dict(process=identity,parent_process=ident(os.getpid()),argv=argv,started_unix=started,registration_sha256=expected,startup_gpu=gpu))
  try:rc=child.wait(timeout=r['max_seconds'][stage]+180)
  except subprocess.TimeoutExpired:
   assert ident(child.pid)['start_ticks']==identity['start_ticks'];os.killpg(child.pid,signal.SIGTERM);terminated=True
   try:rc=child.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);killed=True;rc=child.wait()
 save(d/'actual_wait.json',dict(actual_wait=True,returncode=rc,process=identity,parent_process=ident(os.getpid()),elapsed_seconds=time.monotonic()-beg,terminated=terminated,killed=killed,timed_out=terminated,metadata_error=None,stdout=pin(d/'stdout.log'),stderr=pin(d/'stderr.log'),registration_sha256=expected))
 if rc:raise RuntimeError(f'{stage} actual exit {rc}; preserve failure, no automatic retry')
if __name__=='__main__':
 stage,expected=sys.argv[1:]
 try:run(stage,expected)
 except BaseException:
  d=L/(stage+'_control');d.mkdir(parents=True,exist_ok=True)
  if not(d/'failure.json').exists():save(d/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
  raise
