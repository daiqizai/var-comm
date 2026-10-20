"""Frozen-output CLIP-I supplement; no communication or reconstruction imports."""
from pathlib import Path
import json,hashlib,os,csv,io,time
import numpy as np
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
T=B/'clip_i_common500_20261020_v1'
E=R/'experiments/clip-i-common500-20261020'
O=R/'results/clip_i_common500_20261020_v1'
METHODS=('EC_VAR_PARTIAL','P100000','Swin200000','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024','ARPC_ADAPTED')
NAMES=dict(zip(METHODS,('NeST-Com','Latent JSCC','SwinJSCC','BPG + LDPC','ARPC + LDPC')))
SEEDS={m:(6201,6202,6203)if m in(METHODS[0],METHODS[4])else(2001,2002,2003)for m in METHODS}
SNRS=(4,10,19)
def read(p):return json.loads(Path(p).read_bytes())
def save(p,v):
 with Path(p).open('x',encoding='utf8')as f:
  json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def desc(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(d):
 p=Path(d['path']);assert p.is_absolute()and '..'not in p.parts and not p.is_symlink()
 a=p.stat();assert p.is_file()and ('bytes'not in d or a.st_size==d['bytes'])and sha(p)==d['sha256'],str(p)
 b=p.stat();assert(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 return p
def ash(a):
 a=np.ascontiguousarray(a)
 return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def rawsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def pixels(a):
 assert a.dtype==np.float32 and a.shape==(3,256,256)and np.isfinite(a).all()and a.min()>=0 and a.max()<=1
 return np.ascontiguousarray(a)
def load_image(v):
 with np.load(checked(v['archive']),allow_pickle=False)as z:a=z[v['image_key']]
 if v.get('image_slot')is not None:a=a[int(v['image_slot'])]
 if v.get('source_uint8'):
  assert a.dtype==np.uint8 and rawsha(a)==v['preprocessing_id']
  a=a.astype(np.float32)/np.float32(255)
 a=pixels(a);assert ash(a)==v['pixel_sha256']
 return a
def key(r):return r['method'],int(r['source_index']),int(r['snr_db']),int(r['noise_seed'])
def grid(rows,sources):
 assert len(sources)==500 and [s['source_index']for s in sources]==list(range(500))
 assert len({s['source_id']for s in sources})==500
 keys=[key(r)for r in rows]
 assert len(keys)==22500 and len(set(keys))==22500
 assert set(keys)=={(m,i,s,n)for m in METHODS for i in range(500)for s in SNRS for n in SEEDS[m]}
 for r in rows:
  assert r['source_id']==sources[r['source_index']]['source_id']
  assert r['reference_sha256']==sources[r['source_index']]['pixel_sha256']
  assert r['N']==1024
def csvsave(p,rows):
 assert rows
 with Path(p).open('x',newline='',encoding='utf8')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def process(pid):
 p=Path('/proc')/str(pid)/'stat';v=p.read_text().rsplit(')',1)[1].split()
 return dict(pid=int(pid),start_ticks=int(v[19]),state=v[0])
def gone(v):
 try:return process(v['pid'])['start_ticks']!=v['start_ticks']
 except FileNotFoundError:return True
def interval(a,bank):
 assert a.shape==(500,)and bank.shape==(10000,500)
 d=a[bank].mean(axis=1,dtype=np.float64)
 lo,hi=np.quantile(d,[.025,.975],method='linear')
 return dict(mean=float(a.mean(dtype=np.float64)),ci_low=float(lo),ci_high=float(hi))
