"""Finite 7 dB development study; exclusive records and exact input hashes."""
from pathlib import Path
import sys,os,time,json,hashlib,subprocess,traceback
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'))
from n512_common import sha,pin,checked,read,cr,save,digest,asha,Ledger,ident,gone,np
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/nestcom-7db-20261030';D=R/'results/nestcom_7db_20261030_v1';L=B/'nestcom_7db_20261030_v1'
MCS=((2,'2/3'),(2,'3/4'),(2,'5/6'),(4,'1/2'),(4,'2/3'))
def initialize(request,expected):
 assert sha(request)==expected;r=read(request)
 for d in r['code']:checked(d)
 assert sha(D/'protocol.json')==r['protocol_sha256']
 o=Path(r['output']);o.mkdir(parents=True,exist_ok=False);(o/'tmp').mkdir()
 import tempfile
 tempfile.tempdir=str(o/'tmp');os.environ.update(TMPDIR=str(o/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(o/'inductor'),TRITON_CACHE_DIR=str(o/'triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 os.sched_setaffinity(0,set(range(6)))
 save(o/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),request=pin(request)))
 return r,o
