"""Pinned, private runtime; no global environment or upstream source edits."""
from pathlib import Path
import os, sys, json, hashlib
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
L=B/'multirate_reconstruction_20261105_v1'
OLD=B/'rdvq_prefix_20261104_v1'
UP=OLD/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184'
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
CHECKPOINT_ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/multirate_reconstruction_20261105_v1')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def setup():
    for p in [UP,OLD/'environment/site',R/'experiments/rdvq-prefix-20261104']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(OLD/'extensions_headers_v2'))
    os.environ.setdefault('CPATH',str(OLD/'dependencies/python310_headers/Include')+':'+str(OLD/'dependencies/python310_headers'))
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
def env():
    e=os.environ.copy();e.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1')
    e['PYTHONPATH']=':'.join([str(UP),str(OLD/'environment/site'),str(R/'experiments/rdvq-prefix-20261104')])
    e['TORCH_EXTENSIONS_DIR']=str(OLD/'extensions_headers_v2')
    e['CPATH']=str(OLD/'dependencies/python310_headers/Include')+':'+str(OLD/'dependencies/python310_headers')
    return e
