from pathlib import Path
import os,sys,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');sys.path.insert(0,str(R/'experiments/submission-closure-20261016'));from common import *
routes=read(R/'results/layered_entropy_feasibility_20261014_v1/assets/model_routes.json');print('FILES',routes['files'])
import torch
torch.set_num_threads(1)
vp=next(Path(v['path'])for v in routes['files'].values()if 'vae_ch160' in v['path']) if isinstance(routes['files'],dict)else next(Path(v['path'])for v in routes['files']if 'vae_ch160' in v['path'])
x=torch.load(vp,map_location='cpu',weights_only=True)
print('VAEKEYS',list(x)[:6]);state=x.get('state_dict',x)
groups={}
for k,v in state.items():
 if isinstance(v,torch.Tensor):groups[k.split('.')[0]]=groups.get(k.split('.')[0],0)+v.numel()
print('VAEGROUP',groups);print('BUFFER',[(k,v.numel())for k,v in state.items()if 'ema' in k]);
save(O/'facts/vae_weight_tensor_inventory.json',dict(source_weight=str(vp),existing_weight_identity=routes['files'],top_level_tensor_elements=groups,nonparameter_buffer_keys=[k for k in state if 'ema' in k],scope='CPU state-dict metadata only; no model constructor/forward/tokenizer/channel; original runtime full parameter counts kept separately',CUDA_initialized=torch.cuda.is_initialized()))
