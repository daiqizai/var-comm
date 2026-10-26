"""Immutable finite-stage helpers for the A/B/C response study."""
from pathlib import Path
import ast, csv, hashlib, json, os, sqlite3, sys, time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/adatoken-response-20261026'
O=R/'results/adatoken_response_20261026_v1'
L=B/'adatoken_response_20261026_v1'
SIZES=(1,2,3,4,5,6,8,10,13,16)
OFFSETS=tuple(sum(x*x for x in SIZES[:s])for s in range(11))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p);a=p.stat();h=sha(p);b=p.stat();assert(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 return dict(path=str(p),bytes=a.st_size,sha256=h)
def checked(d):
 p=Path(d['path']);assert p.is_absolute()and '..'not in p.parts
 for q in(p,*p.parents):assert not q.is_symlink(),q
 v=pin(p);assert v==d,(v,d);return p
def read(p):return json.loads(Path(p).read_bytes())
def cr(d):return read(checked(d))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8',newline='\n')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def csvsave(p,rows):
 with Path(p).open('x',encoding='utf8',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def asha(a):
 import numpy as np
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def ident(pid):
 try:
  v=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),start_ticks=int(v[19]),state=v[0])
 except FileNotFoundError:return None
def gone(v):
 p=ident(v['pid']);return p is None or p['start_ticks']!=v['start_ticks']or p['state']=='Z'
def definitions(p,names,ns):
 a=ast.parse(Path(p).read_text(encoding='utf-8-sig'));nodes=[x for x in a.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name in names]
 assert {x.name for x in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns
def grids():
 return {str(d):tuple((m,k)for m in range(4,10)for k in(0,*(SIZES[m]**2*j//d for j in range(1,d))))+((10,0),)for d in(1,2,4,8)}
def runtime_cache(out):
 import tempfile
 for n in('tmp','inductor','triton'):(out/'runtime_cache'/n).mkdir(parents=True,exist_ok=False)
 os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TEMP=str(out/'runtime_cache/tmp'),TMP=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 tempfile.tempdir=os.environ['TMPDIR']
class Ledger:
 def __init__(self,p,caps):
  assert not Path(p).exists();self.caps=caps;self.db=sqlite3.connect(p);self.db.execute('PRAGMA synchronous=FULL');self.db.execute('CREATE TABLE calls(kind TEXT,key TEXT,status TEXT,started REAL,finished REAL,error TEXT,PRIMARY KEY(kind,key))');self.db.commit()
 def call(self,kind,key,fn):
  assert kind in self.caps
  assert self.db.execute('SELECT COUNT(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0]<self.caps[kind]
  self.db.execute('INSERT INTO calls VALUES(?,?,?,?,NULL,NULL)',(kind,str(key),'RESERVED',time.time()));self.db.commit()
  try:v=fn()
  except BaseException as e:
   self.db.execute('UPDATE calls SET status=?,finished=?,error=? WHERE kind=? AND key=?',('FAILED',time.time(),repr(e),kind,str(key)));self.db.commit();raise
  self.db.execute('UPDATE calls SET status=?,finished=? WHERE kind=? AND key=?',('COMPLETE',time.time(),kind,str(key)));self.db.commit();return v
 def summary(self):return dict(caps=self.caps,counts=[dict(kind=k,status=s,count=n)for k,s,n in self.db.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status')],unresolved=self.db.execute("SELECT COUNT(*)FROM calls WHERE status!='COMPLETE'").fetchone()[0])
