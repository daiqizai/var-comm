"""Finite TX-policy mismatch experiment; immutable pins and exclusive receipts."""
from pathlib import Path
import hashlib,json,os,sys,time,csv
import numpy as np
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/wcl-revision-20261027';O=R/'results/wcl_revision_20261027_v1';L=B/'wcl_revision_20261027_v1'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(d):
 p=Path(d['path']);assert p.is_absolute() and not p.is_symlink() and sha(p)==d['sha256'];assert 'bytes'not in d or p.stat().st_size==d['bytes'];return p
def cr(d):return read(checked(d))
def save(p,v):
 with Path(p).open('x',encoding='utf8')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def ash(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def ident(pid):
 a=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=int(a[19]),state=a[0])
def gone(v):
 try:return ident(v['pid'])['start_ticks']!=v['start_ticks']
 except FileNotFoundError:return True
def csvsave(p,rows):
 with Path(p).open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
SIZES=(1,2,3,4,5,6,8,10,13,16)
TARGETS={'WHOLE':(9,0),'PARTIAL':(8,126)}
def nt(e):return sum(x*x for x in SIZES[:e[0]])+e[1]
def chain(family):
 eps=[(m,k)for m in range(4,10)for k in (0,*(SIZES[m]**2*j//4 for j in (1,2,3)))]+[(10,0)]
 return sorted([e for e in eps if nt(e)<=nt(TARGETS[family])and(family=='PARTIAL'or e[1]==0)],key=nt,reverse=True)
def choose(streams,family):
 for m,k in chain(family):
  a=streams[f'm{m}_K{k}']
  if 2<=len(a)<=2520:return m,k,a
 raise RuntimeError('NO_FEASIBLE_ENDPOINT: stop, no invented null transmission')
def imports():
 sys.path[:0]=[str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/entropy-partial-mismatch-20261010/scripts'),str(R/'experiments/wcl-evidence-closure-20261009/scripts')]
def registration(expected):
 p=O/'p1_registration.json';assert sha(p)==expected;r=read(p)
 for d in r['code']:checked(d)
 assert r['logical']==1800 and r['true_SNRs']==[8,10,12]and r['TX_policy_SNR']==10;return r
