"""Real import readiness, without constructing evaluators or loading weights."""
from pathlib import Path
import hashlib,importlib.metadata as md,importlib.util,json,os,socket,sys,tempfile
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
D=T/'dependency_import_qualification_v1';assert not D.exists() and os.environ['CUDA_VISIBLE_DEVICES']=='';D.mkdir()
for n in ('tmp','inductor','triton'):(D/n).mkdir()
os.environ.update(TMPDIR=str(D/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(D/'inductor'),TRITON_CACHE_DIR=str(D/'triton'));tempfile.tempdir=str(D/'tmp')
S=T/'dependency_overlay_v1/sitepackages';sys.path.insert(0,str(S))
def forbid(*a,**k):raise RuntimeError('No checkpoint loads or network allowed during import readiness')
socket.socket.connect=forbid;socket.create_connection=forbid
import torch
torch.load=forbid;torch.jit.load=forbid;torch.hub.load=forbid
assert not torch.cuda.is_initialized()
m=json.loads((T/'metric_assets_posix_manifest.json').read_text())
root=Path(m['models']['dreamsim']['implementation']['path']);sys.path.insert(0,str(root))
import dreamsim,peft,timm,open_clip,transformers,markupsafe,einops,sniffio
assert not torch.cuda.is_initialized() and Path(dreamsim.__file__).is_relative_to(root)
for module in (markupsafe,einops,sniffio):assert Path(module.__file__).is_relative_to(S)
v=dict(status='PASS_REAL_DEPENDENCY_IMPORT_ONLY',cuda_initialized=False,checkpoint_loads=0,model_constructions=0,model_forwards=0,scientific_scores=0,
    modules={x.__name__:dict(origin=x.__file__,version=md.version({'open_clip':'open-clip-torch','markupsafe':'MarkupSafe'}.get(x.__name__,x.__name__))) for x in (peft,timm,open_clip,transformers,markupsafe,einops,sniffio)},
    dreamsim_origin=dreamsim.__file__,overlay_manifest_sha256=hashlib.sha256((T/'dependency_overlay_v1/manifest.json').read_bytes()).hexdigest())
with (D/'completion.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(v,indent=2))
