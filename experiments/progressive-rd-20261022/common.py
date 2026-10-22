"""Independent stage-0 utilities. No imports of model libraries or launches."""
import ast, hashlib, json, os, pathlib, time
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
E=R/'experiments/progressive-rd-20261022'
O=R/'results/progressive_rd_20261022_v1'
L=B/'progressive_rd_20261022_v1'
PY=RT/'envs/frozen_visual_py310/bin/python'
SIZES=(1,2,3,4,5,6,8,10,13,16)
OFFSETS=tuple(sum(s*s for s in SIZES[:i]) for i in range(11))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 p=pathlib.Path(p);assert p.is_file() and not p.is_symlink(),p
 return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(d):
 p=pathlib.Path(d['path']);assert p.is_file() and not p.is_symlink(),p
 assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256'],p
 return p
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def definitions(path,names,ns):
 nodes=[n for n in ast.parse(pathlib.Path(path).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns
def process(pid):
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return None
 s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(s[19]),state=s[0])
def write_torch(path,v,torch):
 path=pathlib.Path(path);assert not path.exists()
 with path.open('xb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
 return pin(path)
class Ledger:
 def __init__(self,path):self.path=path;self.calls=[]
 def event(self,v):
  with self.path.open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 def call(self,kind,key,fn):
  assert not any(x['kind']==kind and x['key']==key for x in self.calls),(kind,key)
  item=dict(kind=kind,key=key,status='STARTED',started_unix=time.time());self.calls.append(item);self.event(item)
  start=time.perf_counter()
  try:value=fn()
  except BaseException as e:
   self.event(dict(item,status='FAILED',error=repr(e),elapsed_seconds=time.perf_counter()-start));raise
  item.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-start);self.event(item);return value
