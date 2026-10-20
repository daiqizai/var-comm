from pathlib import Path
import hashlib,json,os,time,csv
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
E=R/'experiments/arpc-decode-sensitivity-20261021'
O=R/'results/arpc_decode_sensitivity_20261021_v1'
L=B/'arpc_decode_sensitivity_20261021_v1'
OLD=R/'results/arpc_external_baseline_20261019_v1'
OLDL=B/'arpc_external_20261019_v1'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def desc(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(d):
 p=Path(d['path']);assert p.is_file() and not p.is_symlink()
 assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256'],p
 return p
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:
  json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def process(pid):
 a=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(a[19]),state=a[0])
def gone(v):
 try:return process(v['pid'])['start_ticks']!=v['start_ticks']
 except FileNotFoundError:return True
def csvsave(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
