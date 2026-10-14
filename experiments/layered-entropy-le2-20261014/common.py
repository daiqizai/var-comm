from pathlib import Path
import csv,json,hashlib,os,time,sqlite3
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/layered-entropy-le2-20261014'
O=R/'results/layered_entropy_le2_20261014_v1';T=B/'layered_entropy_le2_20261014_v1'
E1=R/'experiments/layered-entropy-feasibility-20261014';O1=R/'results/layered_entropy_feasibility_20261014_v1';T1=B/'layered_entropy_feasibility_20261014_v1'
L=B.parent/'layered_mismatch_raw_20261013'
SIZES=(1,2,3,4,5,6,8,10,13,16);OFFSETS=tuple(sum(x*x for x in SIZES[:i]) for i in range(11))
ENDS=tuple((m,K) for m in range(4,10) for K in (0,*(SIZES[m]**2*i//4 for i in (1,2,3))))
RATES=('1/3','1/2','2/3','3/4','5/6');SNRS=(7,10);CAP=21504
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def pin(p):p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def checked(v):
 p=Path(v['path']);assert p.is_absolute() and '..' not in p.parts and not any(q.is_symlink() for q in (p,*p.parents))
 b=p.stat();assert b.st_size==v.get('bytes',b.st_size) and sha(p)==v['sha256'];a=p.stat();assert (b.st_size,b.st_mtime_ns)==(a.st_size,a.st_mtime_ns);return p
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def csvsave(p,rows):
 assert rows
 with Path(p).open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def asha(a):
 import numpy as np
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def process(pid):
 try:v=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=int(pid),start_ticks=int(v[19]),state=v[0])
 except FileNotFoundError:return None
def gone(p):a=process(p['pid']);return a is None or a['start_ticks']!=p['start_ticks'] or a['state']=='Z'
def verify_registration():
 p=O/'registration.json';reg=read(p)
 assert reg['authorized_stages']==['LE2'] and reg['packet_cap']==CAP and reg['image_wireless_frames']==0
 for v in reg['bindings']:checked(v)
 return reg
def wilson(s,n):
 import math
 z=1.959963984540054;p=s/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0,c-r),min(1,c+r)]
class Ledger:
 def __init__(self,p):
  self.db=sqlite3.connect(p);self.db.execute('PRAGMA synchronous=FULL');self.db.execute('CREATE TABLE calls(kind TEXT,key TEXT,status TEXT,started REAL,finished REAL,error TEXT,PRIMARY KEY(kind,key))');self.db.commit()
 def reserve(self,kind,keys,cap):
  assert self.db.execute('SELECT COUNT(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0]+len(keys)<=cap
  self.db.executemany('INSERT INTO calls VALUES(?,?,?, ?,NULL,NULL)',[(kind,str(k),'RESERVED',time.time()) for k in keys]);self.db.commit()
 def finish(self,kind,keys,error=None):
  self.db.executemany('UPDATE calls SET status=?,finished=?,error=? WHERE kind=? AND key=?',[('FAILED' if error else 'COMPLETE',time.time(),error,kind,str(k)) for k in keys]);self.db.commit()
 def call(self,kind,key,cap,fn):
  self.reserve(kind,[key],cap)
  try:r=fn()
  except BaseException as e:self.finish(kind,[key],repr(e));raise
  self.finish(kind,[key]);return r
 def summary(self):return [dict(kind=k,status=s,count=n) for k,s,n in self.db.execute('SELECT kind,status,COUNT(*) FROM calls GROUP BY kind,status')]
