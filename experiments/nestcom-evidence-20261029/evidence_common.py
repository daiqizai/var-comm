"""Exclusive finite-stage records for the 2026-10-29 E1/E2 study."""
from pathlib import Path
import hashlib,json,os,sqlite3,sys,time
import numpy as np
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/nestcom-evidence-20261029'
O=R/'results/nestcom_evidence_20261029_v1'
L=B/'nestcom_evidence_20261029_v1'
BASE='8bab318fbc8f8f115e96c9aeffeef10d3c2bb4b6'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(d):
 p=Path(d['path']);assert p.is_absolute() and not p.is_symlink() and sha(p)==d['sha256'],p
 assert 'bytes' not in d or p.stat().st_size==d['bytes'];return p
def read(p):return json.loads(Path(p).read_bytes())
def cr(d):return read(checked(d))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def asha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),start_ticks=int(s[19]),state=s[0])
 except FileNotFoundError:return None
def gone(v):
 z=ident(v['pid']);return z is None or z['start_ticks']!=v['start_ticks'] or z['state']=='Z'
class Ledger:
 def __init__(self,path,caps):
  assert not Path(path).exists();self.caps=caps;self.db=sqlite3.connect(path)
  self.db.execute('PRAGMA synchronous=FULL');self.db.execute('CREATE TABLE calls(kind TEXT,key TEXT,status TEXT,started REAL,finished REAL,error TEXT,PRIMARY KEY(kind,key))');self.db.commit()
 def call(self,kind,key,fn):
  assert kind in self.caps
  assert self.db.execute('SELECT COUNT(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0]<self.caps[kind]
  self.db.execute('INSERT INTO calls VALUES(?,?,?,?,NULL,NULL)',(kind,str(key),'RESERVED',time.time()));self.db.commit()
  try:v=fn()
  except BaseException as e:
   self.db.execute('UPDATE calls SET status=?,finished=?,error=? WHERE kind=? AND key=?',('FAILED',time.time(),repr(e),kind,str(key)));self.db.commit();raise
  self.db.execute('UPDATE calls SET status=?,finished=? WHERE kind=? AND key=?',('COMPLETE',time.time(),kind,str(key)));self.db.commit();return v
 def summary(self):return dict(caps=self.caps,counts=[dict(kind=k,status=s,count=n) for k,s,n in self.db.execute('SELECT kind,status,COUNT(*) FROM calls GROUP BY kind,status')],unresolved=self.db.execute("SELECT COUNT(*) FROM calls WHERE status!='COMPLETE'").fetchone()[0])

