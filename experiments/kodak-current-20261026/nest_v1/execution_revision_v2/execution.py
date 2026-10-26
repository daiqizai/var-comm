"""Execution-only PCI mapping correction; frozen scientific modules are unmodified."""
from pathlib import Path
import importlib,importlib.util,json,os,signal,sqlite3,subprocess,sys,time,traceback
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
import kn_common as c
OLD_L=c.L
NEW_L=OLD_L/'recovery_v1'
REV=c.O/'execution_revision_v2/registration.json'
SOURCE_REVISION='291a4b884d2ff38c4a2ef31940773df4443dd20b9a1bc7afb798a5852f6f42d3'
PARENT='8c66f5261d2b37d1d14b5273923d37ceebdd6e93d801e1da93b3c7c802e805b2'
STAGES=('source','phy','receiver','delivery')
def assert_mapping(m,gpu):
 assert m['nvml_uuid']==gpu['uuid'] and m['nvml_index']==0
 assert m['pci'].lower()==gpu['pci_bus_id'][-12:].lower()
 assert m['cuda_visible_devices']=='0' and m['cuda_ordinal']==0 and m['cuda_device_order']=='PCI_BUS_ID'
 assert m['model_imports']==0 and m['context_creation_calls']==0
def load(expected):
 assert c.sha(REV)==expected
 rev=c.read(REV);assert rev['parent_registration_sha256']==PARENT
 r=c.registration(PARENT)
 assert c.checked(rev['source_execution_revision'])==c.O/'execution_revision_v1/registration.json'
 assert rev['source_execution_revision']['sha256']==SOURCE_REVISION
 for d in rev['code']:c.checked(d)
 audit=c.cr(rev['failed_initialization_audit'])
 for d in audit['files']:c.checked(d)
 assert audit['successful_scientific_calls']==0 and audit['failed_model_initializations']==1
 assert all(c.gone(p) for p in audit['processes'])
 assert_mapping(rev['gpu_mapping'],r['gpu'])
 c.L=NEW_L
 return rev,r
def current_mapping(rev,r):
 p=c.checked(rev['mapping_source']);spec=importlib.util.spec_from_file_location('_kn_gpu_mapping',p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 records=m.query_clean()
 raw=subprocess.check_output(['nvidia-smi','-i',r['gpu']['uuid'],'--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 nv=dict(index=int(raw[0]),uuid=raw[1].strip(),pci=raw[2].strip())
 mapping=m.select_mapping(records,nv);assert_mapping(mapping,r['gpu'])
 assert mapping==rev['gpu_mapping']
 assert int(raw[3])>=30000 and int(raw[4])<=80,raw
 return dict(nvml=raw,mapping=mapping)
def closure(stage,expected):
 load(expected)
 for suffix in ('_control','_initial_control'):
  w=c.read(NEW_L/(stage+suffix)/'actual_wait.json')
  assert w['actual_wait'] is True and type(w['returncode']) is int and w['returncode']==0
  assert not any(w[k] for k in ('terminated','killed','timed_out')) and w['metadata_error'] is None
  assert c.gone(w['process']) and c.gone(w['parent_process'])
  assert w['execution_revision_sha256']==(SOURCE_REVISION if stage=='source' else expected)
 c.closed(stage,PARENT)
def wait_child(argv,d,env,limit,expected,extra=None):
 start=time.monotonic();terminated=killed=False
 with (d/'stdout.log').open('xb')as so,(d/'stderr.log').open('xb')as se:
  p=subprocess.Popen(argv,cwd=c.R,env=env,stdout=so,stderr=se,start_new_session=True)
  identity=c.ident(p.pid);assert identity
  launch=dict(process=identity,parent_process=c.ident(os.getpid()),argv=argv,started_unix=time.time(),registration_sha256=PARENT,execution_revision_sha256=expected)
  if extra:launch.update(extra)
  c.save(d/'launch.json',launch)
  try:rc=p.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   assert c.ident(p.pid)['start_ticks']==identity['start_ticks'];os.killpg(p.pid,signal.SIGTERM);terminated=True
   try:rc=p.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);killed=True;rc=p.wait()
 c.save(d/'actual_wait.json',dict(actual_wait=True,returncode=rc,process=identity,parent_process=c.ident(os.getpid()),elapsed_seconds=time.monotonic()-start,terminated=terminated,killed=killed,timed_out=terminated,metadata_error=None,stdout=c.pin(d/'stdout.log'),stderr=c.pin(d/'stderr.log'),registration_sha256=PARENT,execution_revision_sha256=expected))
 if rc:raise RuntimeError(f'Actual child exit {rc}; no automatic retry')
def worker(stage,expected):
 load(expected)
 assert stage in STAGES
 mod=importlib.import_module('kn_'+stage)
 assert mod.L==NEW_L
 c.save(NEW_L/(stage+'_control')/'worker_revision.json',dict(registration_sha256=PARENT,execution_revision_sha256=expected,argv=sys.argv,scientific_entry=c.pin(BASE/('kn_'+stage+'.py')),scientific_output_root=str(NEW_L)))
 assert stage in ('phy','receiver','delivery')
 mod.run(PARENT)
def owner(stage,expected):
 import fcntl
 rev,r=load(expected);assert stage in ('phy','receiver','delivery')
 if stage!='source':closure(STAGES[STAGES.index(stage)-1],expected)
 assert not (NEW_L/stage).exists() if stage!='delivery' else not(c.O/'delivery').exists()
 d=NEW_L/(stage+'_control');d.mkdir(parents=True,exist_ok=False)
 lock=(OLD_L/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6')
 gpu=current_mapping(rev,r) if stage in ('source','receiver') else None
 if gpu:env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 py=r['python']['gpu' if gpu else 'cpu']
 argv=[py,'-B',str(Path(__file__).resolve()),'worker',stage,expected]
 wait_child(argv,d,env,r['max_seconds'][stage]+180,expected,dict(startup_gpu=gpu))
def supervisor(stage,expected):
 rev,r=load(expected);d=NEW_L/(stage+'_initial_control')
 assert d.is_dir() and not(d/'launch.json').exists()
 argv=[r['python']['cpu'],'-B',str(Path(__file__).resolve()),'owner',stage,expected]
 wait_child(argv,d,dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),r['max_seconds'][stage]+360,expected)
def dispatch(stage,expected):
 rev,r=load(expected);assert stage in ('phy','receiver','delivery')
 if stage!='source':closure(STAGES[STAGES.index(stage)-1],expected)
 for p in (NEW_L/(stage+'_initial_control'),NEW_L/(stage+'_control'),NEW_L/stage if stage!='delivery'else c.O/'delivery'):assert not p.exists(),p
 if stage in ('source','receiver'):current_mapping(rev,r)
 d=NEW_L/(stage+'_initial_control');d.mkdir(parents=True,exist_ok=False)
 argv=[r['python']['cpu'],'-B',str(Path(__file__).resolve()),'supervisor',stage,expected]
 with(d/'supervisor_stdout.log').open('xb')as so,(d/'supervisor_stderr.log').open('xb')as se:
  p=subprocess.Popen(argv,cwd=c.R,stdout=so,stderr=se,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
  identity=c.ident(p.pid);assert identity
 c.save(d/'dispatch.json',dict(process=identity,parent_process=c.ident(os.getpid()),argv=argv,registration_sha256=PARENT,execution_revision_sha256=expected))
 print(json.dumps(dict(stage=stage,supervisor=identity,execution_revision_sha256=expected)))
if __name__=='__main__':
 role,stage,expected=sys.argv[1:]
 assert role in ('worker','owner','supervisor','dispatch')
 try:globals()[role](stage,expected)
 except BaseException:
  d=NEW_L/(stage+('_initial_control' if role in ('supervisor','dispatch')else'_control'))
  if d.exists() and not(d/(role+'_failure.json')).exists():c.save(d/(role+'_failure.json'),dict(traceback=traceback.format_exc(),automatic_retry=False))
  raise
