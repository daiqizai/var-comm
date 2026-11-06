"""Independent VAR RD-transfer runtime. Historical models and sources are read-only."""
from pathlib import Path
import ast,hashlib,json,os,sys
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
L=B/'nestcom_rd_transfer_20261106_v1'
S=R/'experiments/nestcom-rd-transfer-20261106'
OLD=B/'rdvq_prefix_20261104_v1'
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
CK=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/nestcom_rd_transfer_20261106_v1')
SIZES=(1,2,3,4,5,6,8,10,13,16)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def definitions(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def setup():
    for p in [OLD/'environment/site',R/'experiments/rdvq-prefix-20261104',R/'experiments/wcl-evidence-closure-20261009/scripts']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
def env():
    e=os.environ.copy();e.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    e['PYTHONPATH']=':'.join([str(S),str(OLD/'environment/site'),str(R/'experiments/rdvq-prefix-20261104')])
    return e
def numerical():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
def old_paused():
    assert (OLD/'STOP_P2').exists()
    assert (B/'multirate_reconstruction_20261105_v1/STOP_MAIN').exists()
