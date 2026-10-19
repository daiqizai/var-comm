"""Immutable paths, receipts, bounded call ledger and actual child waits."""
from pathlib import Path
import hashlib,json,os,sys,time,sqlite3,traceback,subprocess,signal
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
RT=R.parent.parent/'var_comm_runtime_20261010'
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
E=R/'experiments/arpc-n2048-adaptive-20261019'
O=R/'results/arpc_n2048_adaptive_20261019_v1'
L=BASE/'arpc_n2048_adaptive_20261019_v1'
OLD_E=R/'experiments/arpc-external-baseline-20261019'
OLD_O=R/'results/arpc_external_baseline_20261019_v1'
OLD_L=BASE/'arpc_external_20261019_v1'
REG=O/'b1_registration.json'
CPY=RT/'envs/frozen_ldpc_py311/bin/python'
PY=RT/'envs/frozen_visual_py310/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pin(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(d):
 p=Path(d['path']);assert p.is_absolute() and '..' not in p.parts and p.is_file()
 for x in [p,*p.parents]:assert not x.is_symlink()
 a=p.stat();assert ('bytes' not in d or a.st_size==d['bytes']) and sha(p)==d['sha256'],p
 b=p.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 return p
def read(p):return json.loads(Path(p).read_text())
def put(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=int(pid),start_ticks=int(s[19]),state=s[0],host=os.uname().nodename)
 except FileNotFoundError:return None
def gone(i):
 assert i['host']==os.uname().nodename
 a=identity(i['pid']);return a is None or a['start_ticks']!=i['start_ticks'] or a['state']=='Z'
def cache(out):
 import tempfile
 for n in ('tmp','inductor','triton'):(out/'runtime_cache'/n).mkdir(parents=True,exist_ok=False)
 os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 tempfile.tempdir=os.environ['TMPDIR']
class Ledger:
 def __init__(self,p,caps):
  assert not p.exists();self.caps=caps;self.db=sqlite3.connect(p);self.db.execute('PRAGMA synchronous=FULL')
  self.db.execute('CREATE TABLE calls(kind TEXT,key TEXT,status TEXT,started REAL,finished REAL,error TEXT,PRIMARY KEY(kind,key))');self.db.commit()
 def call(self,kind,key,fn):
  n=self.db.execute('SELECT count(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0];assert n<self.caps[kind]
  self.db.execute('INSERT INTO calls VALUES(?,?,?, ?,NULL,NULL)',(kind,str(key),'RESERVED',time.time()));self.db.commit()
  try:v=fn()
  except BaseException:
   self.db.execute('UPDATE calls SET status=?,finished=?,error=? WHERE kind=? AND key=?',('FAILED',time.time(),traceback.format_exc(),kind,str(key)));self.db.commit();raise
  self.db.execute('UPDATE calls SET status=?,finished=? WHERE kind=? AND key=?',('COMPLETE',time.time(),kind,str(key)));self.db.commit();return v
 def summary(self):return dict(caps=self.caps,counts=[dict(kind=k,status=s,count=n) for k,s,n in self.db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status')],unresolved=self.db.execute("SELECT count(*) FROM calls WHERE status!='COMPLETE'").fetchone()[0])
def registration(expected):
 assert sha(REG)==expected;r=read(REG)
 for d in r['code']:verify(d)
 return r
def wait_child(out,argv,env,timeout):
 start=time.time();timed=terminated=killed=False;metadata_error=None
 with (out/'stdout').open('xb') as so,(out/'stderr').open('xb') as se:
  p=subprocess.Popen(argv,env=env,stdout=so,stderr=se,start_new_session=True)
  i=identity(p.pid)
  if i is None:p.terminate();p.wait();raise RuntimeError('child identity unavailable')
  put(out/'launch.json',dict(process=i,parent_process=identity(os.getpid()),argv=argv,started_unix=start,timeout_seconds=timeout))
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(p.pid,signal.SIGTERM)
   try:rc=p.wait(timeout=30)
   except subprocess.TimeoutExpired:killed=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
  put(out/'actual_wait.json',dict(process=i,returncode=rc,seconds=time.time()-start,timed_out=timed,terminated=terminated,killed=killed,metadata_error=metadata_error))
  assert type(rc) is int and rc==0 and not timed and gone(i)
