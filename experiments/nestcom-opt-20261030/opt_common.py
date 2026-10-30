"""New bounded optimization namespace; never imports an old task controller."""
from pathlib import Path
import os,sys,json,hashlib,time,sqlite3,ast
import numpy as np
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/nestcom-opt-20261030';D=R/'results/nestcom_opt_20261030_v1';L=B/'nestcom_opt_20261030_v1'
SIZES=(1,2,3,4,5,6,8,10,13,16)
OFFSETS=np.cumsum([0]+[s*s for s in SIZES]).tolist()
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(8<<20),b''):h.update(a)
 return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(v):
 p=Path(v['path']);assert p.is_absolute()and not p.is_symlink()and sha(p)==v['sha256'],p
 assert 'bytes'not in v or p.stat().st_size==v['bytes'];return p
def cr(v):return read(checked(v))
def asha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def definitions(p,names,ns):
 nodes=[n for n in ast.parse(Path(p).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns
def ident(pid):
 try:
  a=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),start_ticks=int(a[19]),state=a[0])
 except FileNotFoundError:return None
class BudgetStop(Exception):pass
class Ledger:
 def __init__(self,path,caps):
  assert not Path(path).exists();self.caps=caps;self.db=sqlite3.connect(path)
  self.db.execute('PRAGMA synchronous=FULL');self.db.execute('CREATE TABLE calls(kind TEXT,key TEXT,status TEXT,started REAL,finished REAL,error TEXT,PRIMARY KEY(kind,key))');self.db.commit()
 def call(self,kind,key,fn):
  assert kind in self.caps and self.db.execute('SELECT COUNT(*)FROM calls WHERE kind=?',(kind,)).fetchone()[0]<self.caps[kind]
  self.db.execute('INSERT INTO calls VALUES(?,?,?,?,NULL,NULL)',(kind,str(key),'RESERVED',time.time()));self.db.commit()
  try:v=fn()
  except BaseException as e:
   self.db.execute('UPDATE calls SET status=?,finished=?,error=? WHERE kind=?AND key=?',('FAILED',time.time(),repr(e),kind,str(key)));self.db.commit();raise
  self.db.execute('UPDATE calls SET status=?,finished=? WHERE kind=?AND key=?',('COMPLETE',time.time(),kind,str(key)));self.db.commit();return v
 def summary(self):return dict(caps=self.caps,counts=[dict(kind=k,status=s,count=n)for k,s,n in self.db.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status')],unresolved=self.db.execute("SELECT COUNT(*)FROM calls WHERE status!='COMPLETE'").fetchone()[0])
