"""New LE0/LE1 scope, immutable metadata, and once-only scientific ledger."""
from pathlib import Path
import csv,hashlib,json,os,sqlite3,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/submission-closure-20261016'
O=R/'results/submission_closure_20261016_v1'
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
T=B/'submission_closure_20261016_v1'
OLD=R/'results/layered_mismatch_raw_20261013'
L=B.parent/'layered_mismatch_raw_20261013'
MODELS=dict(vae='6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b',var='b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5',decoder='bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1')
SIZES=(1,2,3,4,5,6,8,10,13,16)
OFFSETS=tuple(sum(x*x for x in SIZES[:i]) for i in range(11))
ENDS=tuple((m,K) for m in range(4,10) for K in (0,*(SIZES[m]**2*i//4 for i in (1,2,3))))
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pin(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def checked(v):
 p=Path(v['path']);assert p.is_absolute() and '..' not in p.parts and p.is_file()
 for q in [p,*p.parents]:assert not q.is_symlink()
 before=p.stat();assert ('bytes' not in v or before.st_size==v['bytes']) and sha(p)==v['sha256']
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 return p
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def csvsave(p,rows):
 assert rows
 with Path(p).open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def asha(a):
 import numpy as np
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def pid_identity(pid):
 try:
  p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=int(pid),start_ticks=int(v[19]),state=v[0])
 except FileNotFoundError:return None
def gone(v):
 p=pid_identity(v['pid']);return p is None or p['start_ticks']!=v['start_ticks'] or p['state']=='Z'
class Ledger:
 def __init__(self,path,caps):
  self.caps=caps;self.db=sqlite3.connect(path);self.db.execute('PRAGMA synchronous=FULL')
  self.db.execute('CREATE TABLE IF NOT EXISTS calls(kind TEXT,key TEXT,status TEXT,started REAL,finished REAL,error TEXT,PRIMARY KEY(kind,key))');self.db.commit()
 def call(self,kind,key,fn):
  assert kind in self.caps
  n=self.db.execute('SELECT COUNT(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0];assert n<self.caps[kind],(kind,n,self.caps[kind])
  self.db.execute('INSERT INTO calls VALUES(?,?,?, ?,NULL,NULL)',(kind,str(key),'RESERVED',time.time()));self.db.commit()
  try:value=fn()
  except BaseException as e:
   self.db.execute('UPDATE calls SET status=?,finished=?,error=? WHERE kind=? AND key=?',('FAILED',time.time(),repr(e),kind,str(key)));self.db.commit();raise
  self.db.execute('UPDATE calls SET status=?,finished=? WHERE kind=? AND key=?',('COMPLETE',time.time(),kind,str(key)));self.db.commit();return value
 def summary(self):return dict(caps=self.caps,counts=[dict(kind=k,status=s,count=n) for k,s,n in self.db.execute('SELECT kind,status,COUNT(*) FROM calls GROUP BY kind,status')],unresolved=self.db.execute("SELECT COUNT(*) FROM calls WHERE status!='COMPLETE'").fetchone()[0])
