"""Immutable identities and durable single-call receipts for Kodak scoring."""
from pathlib import Path
import hashlib,json,os,csv,time,sqlite3
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
O=R/'results/kodak_current_20261026_v1/baselines_metrics_v1'
L=B/'kodak_current_20261026_v1/baselines_metrics_v1'
E=R/'experiments/kodak-current-20261026/baselines_metrics_v1'
METHOD_MAP={'EC_VAR_PARTIAL':'NEST_PARTIAL','PARTIAL':'NEST_PARTIAL','NEST_PARTIAL':'NEST_PARTIAL','EC_VAR_WHOLE':'NEST_WHOLE','WHOLE':'NEST_WHOLE','NEST_WHOLE':'NEST_WHOLE','P100K':'P100000','P100000':'P100000','SWIN200K':'Swin200000','Swin200000':'Swin200000','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG_ADAPTIVE','BPG_ADAPTIVE':'BPG_ADAPTIVE','ARPC_ADAPTED':'ARPC_ADAPTED'}
METHODS=('NEST_PARTIAL','NEST_WHOLE','P100000','Swin200000','BPG_ADAPTIVE','ARPC_ADAPTED')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def desc(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(d):
 p=Path(d['path']);assert p.is_file() and not p.is_symlink(),p
 assert sha(p)==d['sha256'] and ('bytes' not in d or p.stat().st_size==d['bytes']),p
 return p
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p)
 with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def csvsave(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def ah(x):return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes(order='C')).hexdigest()
def rawah(x):return hashlib.sha256(x.tobytes(order='C')).hexdigest()
def assert_array(x,expected,convention):
 import numpy as np
 assert x.dtype==np.float32 and x.shape==(3,256,256) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
 if convention in ('dtype_shape_cbytes','dtype_shape_bytes','canonical','dtype_shape_c_contiguous_bytes','sha256(str(dtype)+str(shape)+contiguous_bytes)'):actual=ah(x)
 elif convention in ('raw_bytes','raw_cbytes','raw_float32_cbytes','sha256_raw_contiguous_bytes','raw_c_contiguous_bytes','sha256(array.tobytes(C)); float32 CHW [3,256,256]'):actual=rawah(x)
 else:raise AssertionError('Unknown array hash convention: '+str(convention))
 assert actual==expected,(actual,expected,convention)
 return ah(x)
def framekey(r):return (METHOD_MAP[r['method']],int(r['source_index']),int(r['snr_db']),int(r['noise_seed']))
def assert_grid(rows):
 exp={(m,i,s,n) for m in METHODS for i in range(24) for s in (4,10,19) for n in (2001,2002,2003)}
 keys=[framekey(r) for r in rows];assert len(keys)==len(exp) and set(keys)==exp
def process(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=int(pid),start_ticks=int(s[19]),state=s[0],cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def normalized_pci_bus(value):
 domain,bus,dev=value.strip().lower().split(':');slot,fn=dev.split('.')
 return f'{int(domain,16):04x}:{int(bus,16):02x}:{int(slot,16):02x}.{int(fn,16):x}'
def gone(p):
 f=Path('/proc')/str(p['pid'])/'stat'
 if not f.exists():return True
 s=f.read_text().rsplit(')',1)[1].split();return int(s[19])!=int(p['start_ticks']) or s[0]=='Z'
def verify_host_closure(proof,hostname,identities,registration,role):
 assert proof['hostname']==hostname and proof['registration_sha256']==registration and proof['role']==role
 assert isinstance(proof['observed_unix'],(int,float)) and proof['observed_unix']>0
 expected={(int(p['pid']),int(p['start_ticks'])) for p in identities}
 actual={(int(p['identity']['pid']),int(p['identity']['start_ticks'])) for p in proof['processes']}
 assert len(proof['processes'])==len(expected) and actual==expected
 assert all(p['gone'] is True for p in proof['processes'])
class Ledger:
 def __init__(self,out,limits):
  self.out=Path(out);self.limits=limits;self.db=sqlite3.connect(self.out/'ledger.sqlite')
  self.db.execute('CREATE TABLE IF NOT EXISTS calls(kind TEXT,id TEXT,status TEXT,receipt TEXT,PRIMARY KEY(kind,id))');self.db.commit()
  (self.out/'call_results').mkdir(exist_ok=True)
 def call(self,kind,key,fn,feature=False):
  row=self.db.execute('SELECT status,receipt FROM calls WHERE kind=? AND id=?',(kind,key)).fetchone()
  if row:
   assert row[0]=='COMPLETE','Incomplete call preserved; explicit diagnosis needed'
   p=checked(json.loads(row[1]))
   if feature:
    import numpy as np
    with p.open('rb') as f:return np.load(f,allow_pickle=False)
   return read(p)['value']
  assert kind in self.limits and self.db.execute('SELECT count(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0]<self.limits[kind]
  self.db.execute('INSERT INTO calls VALUES(?,?,?,?)',(kind,key,'RESERVED',''));self.db.commit()
  try:
   val=fn();name=hashlib.sha256((kind+'|'+key).encode()).hexdigest();p=self.out/'call_results'/(name+('.npy' if feature else '.json'))
   if feature:
    import numpy as np
    with p.open('xb') as f:np.save(f,val,allow_pickle=False);f.flush();os.fsync(f.fileno())
   else:save(p,dict(kind=kind,id=key,value=val))
   self.db.execute('UPDATE calls SET status=?,receipt=? WHERE kind=? AND id=?',('COMPLETE',json.dumps(desc(p),sort_keys=True),kind,key));self.db.commit();return val
  except BaseException:
   self.db.execute('UPDATE calls SET status=? WHERE kind=? AND id=?',('FAILED',kind,key));self.db.commit();raise
 def counts(self):return self.db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status').fetchall()
