"""Kodak-only execution metadata; original scientific functions remain immutable."""
from pathlib import Path
import ast,csv,hashlib,json,os,sqlite3,sys,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
RT=R.parent.parent/'var_comm_runtime_20261010'
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
E=R/'experiments/kodak-current-20261026/nest_v1'
O=R/'results/kodak_current_20261026_v1/nest_v1'
L=B/'kodak_current_20261026_v1/nest_v1'
SIZES=(1,2,3,4,5,6,8,10,13,16)
OFFSETS=tuple(sum(s*s for s in SIZES[:i])for i in range(11))
ENDS=tuple((m,k)for m in range(4,10)for k in (0,*(SIZES[m]**2*j//4 for j in (1,2,3))))
MODELS=dict(vae='6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b',var='b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5',decoder='bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1')
sys.path[:0]=[str(R/'experiments/entropy-partial-mismatch-20261010/scripts'),str(R/'experiments/wcl-evidence-closure-20261009/scripts'),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'src')]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p);a=p.stat();h=sha(p);z=p.stat();assert(a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
 return dict(path=str(p),bytes=a.st_size,sha256=h)
def checked(d):
 p=Path(d['path']);assert p.is_absolute()and '..'not in p.parts
 for q in (p,*p.parents):assert not q.is_symlink(),q
 a=pin(p);assert a['sha256']==d['sha256']and a['bytes']==d.get('bytes',a['bytes']),p
 return p
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def cr(d):return read(checked(d))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8',newline='\n')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def csvsave(p,rows):
 assert rows
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
def definitions(path,names,ns):
 nodes=[n for n in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def registration(expected):
 assert sha(O/'registration.json')==expected;r=read(O/'registration.json')
 for d in r['code']:checked(d)
 assert r['scope']=='KODAK24_CURRENT_N1024_NEST_WHOLE_ONLY'and r['logical_frames']==432
 return r
def runtime_cache(out):
 import tempfile
 for n in ('tmp','inductor','triton'):(out/'runtime_cache'/n).mkdir(parents=True,exist_ok=False)
 os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TMP=str(out/'runtime_cache/tmp'),TEMP=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'))
 tempfile.tempdir=os.environ['TMPDIR']
def source_key(rx):
 b=rx['body'];p=rx['rx_profile']
 if not rx['header']['header_ok']or b is None or not b['crc_accepted']or not b['parser_accepted']:return digest({'kind':'FIXED_GRAY'})
 return digest(dict(kind='ACTUAL_SOURCE_INPUT',family=p['family'],m=p['m'],K=p['K'],payload=b['payload']))
def counter(i,snr,seed):
 assert 0<=i<24 and snr in (4,10,19)and seed in (2001,2002,2003)
 return ((4,10,19).index(snr)*24+i)*3+seed-2001
def start(stage,expected):
 r=registration(expected);out=L/stage;out.mkdir(parents=True,exist_ok=False)
 save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 return r,out
def closed(stage,expected):
 w=read(L/(stage+'_control')/'actual_wait.json')
 assert w['actual_wait']is True and type(w['returncode'])is int and w['returncode']==0
 assert all(w[k]is False for k in ('timed_out','terminated','killed'))and w['metadata_error']is None
 assert gone(w['process'])and gone(w['parent_process'])
 c=read(L/stage/'completion.json');assert c['registration_sha256']==expected;return c
class Ledger:
 def __init__(self,p,caps):
  assert not Path(p).exists();self.caps=caps;self.db=sqlite3.connect(p);self.db.execute('PRAGMA synchronous=FULL');self.db.execute('CREATE TABLE calls(kind TEXT,key TEXT,status TEXT,started REAL,finished REAL,error TEXT,PRIMARY KEY(kind,key))');self.db.commit()
 def call(self,kind,key,fn):
  assert kind in self.caps
  n=self.db.execute('SELECT COUNT(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0];assert n<self.caps[kind],(kind,n)
  self.db.execute('INSERT INTO calls VALUES(?,?,?, ?,NULL,NULL)',(kind,str(key),'RESERVED',time.time()));self.db.commit()
  try:value=fn()
  except BaseException as e:
   self.db.execute('UPDATE calls SET status=?,finished=?,error=? WHERE kind=? AND key=?',('FAILED',time.time(),repr(e),kind,str(key)));self.db.commit();raise
  self.db.execute('UPDATE calls SET status=?,finished=? WHERE kind=? AND key=?',('COMPLETE',time.time(),kind,str(key)));self.db.commit();return value
 def summary(self):return dict(caps=self.caps,counts=[dict(kind=k,status=s,count=n)for k,s,n in self.db.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status')],unresolved=self.db.execute("SELECT COUNT(*)FROM calls WHERE status!='COMPLETE'").fetchone()[0])
