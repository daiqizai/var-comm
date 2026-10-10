"""Portable I/O and original mathematical definitions for baseline continuation.

The original source files are read and bound by SHA. Definitions are compiled
unchanged, avoiding legacy module-level launch paths. No historical file is edited.
"""
from __future__ import annotations
import ast, copy, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
MODELTEAM=Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
RT=MODELTEAM/'code/liulu/var_comm_runtime_20261010'
CP=MODELTEAM/'liberai-checkpoint/liulu/var_comm/baseline_strength_20261011'
TMP=MODELTEAM/'liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011'
DATA=TMP/'inputs'
RESULT=ROOT/'results/baseline_strength_20261011'
P_SHA='12b979260ebf46dc3ec6a453ef7218d734f9b6aed6c3157c3676a833c3bb7df9'
S_SHA='8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(8<<20),b''):h.update(v)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    return dict(path=str(p),sha256=sha(p))

def definitions(path,names,ns,bindings):
    p=Path(path);text=p.read_text();tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    bindings[str(p)]=sha(p)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)

def p_math():
    ns=dict(torch=torch,nn=torch.nn,np=np,copy=copy,math=math,hashlib=hashlib)
    bindings={};base=ROOT/'experiments/var-latent-enhancement-20260917'
    for p,names in [
        (base/'src/latent_enhancement/latent.py',{'normalize_enhancement','ContinuousDecoder'}),
        (base/'phase_b/src/latent_enhancement_b/model.py',{'ResidualBlock'}),
        (base/'research/src/latent_research/models.py',{'PureContinuous'}),
        (ROOT/'experiments/extreme-bandwidth-20261001-N1024/budget_model.py',{'BudgetContinuous'}),
        (base/'src/latent_enhancement/runtime.py',{'image_losses'}),
        (ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/train.py',{'forward','losses','update','same_tree'}),
        (base/'src/latent_enhancement/training.py',{'PairedOrder'}),
        (ROOT/'src/var_comm/study.py',{'seeded_noise'}),
        (ROOT/'src/var_comm/next_scale_prior.py',{'state_sha256'}),
    ]:definitions(p,names,ns,bindings)
    return ns,bindings

def configure():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.set_num_threads(6);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    assert torch.cuda.device_count()==1
    prop=torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction((16<<30)/prop.total_memory)
    return dict(torch=torch.__version__,cuda=torch.version.cuda,device=prop.name,
                device_uuid=str(prop.uuid),FP32=True,TF32=False,deterministic=True,
                allocator_limit_bytes=16<<30,threads=6,interop_threads=2,
                affinity=sorted(os.sched_getaffinity(0)),old_device_bitwise_equivalence_claimed=False)

def global_rng():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(v):torch.set_rng_state(v['cpu']);torch.cuda.set_rng_state_all(v['cuda'])

def p_models(ns):
    parent=next((CP/'parents/p').glob('*.pt'));assert sha(parent)==P_SHA
    payload=torch.load(parent,map_location='cpu',weights_only=True)
    original=read(CP/'parents/p/registration.json')
    assert payload['registration_sha256']==sha(CP/'parents/p/registration.json')
    assert payload['state']['step']==40000
    scale=payload['models']['P1024.scale'].flatten()
    model=ns['BudgetContinuous'](scale,1024)
    model.load_state_dict({k.removeprefix('P1024.'):v for k,v in payload['models'].items()},strict=True)
    assert sum(p.numel() for p in model.parameters())==495208
    sys.path.insert(0,str(RT/'assets/var_upstream_candidate_v1'))
    sys.path.insert(0,str(RT/'assets/var_native_source_seed_v1/VAR'))
    from models.vqvae import VQVAE
    vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,test_mode=True,share_quant_resi=4)
    weight=RT/'assets/model_seed_v1/external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth'
    assert sha(weight)=='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c'
    vae.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    assert ns['state_sha256'](vae)==original['encoder_state_sha256']
    dc=ns['ContinuousDecoder'](vae);del vae
    weight=RT/'assets/model_seed_v1/VAR_COMM/outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_A_v1/checkpoints/update_0038000_1789635001035591298.pt'
    assert sha(weight)=='e3d757a5a16206f5f9be345d0e6330b0f1fd6c78bf167670e70378b429d931c4'
    dc.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['decoder'],strict=True)
    assert ns['state_sha256'](dc)==original['decoder_state_sha256']
    import lpips
    lp=lpips.LPIPS(net='alex',pnet_rand=True,model_path=str(Path(lpips.__file__).parent/'weights/v0.1/alex.pth'),verbose=False)
    weight=RT/'assets/pilot_metric_weights_seed_v1/external/home/liulu/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
    weights=torch.load(weight,map_location='cpu',weights_only=True)
    lp.net.load_state_dict({n:weights['features.'+n.split('.',1)[1]] for n in lp.net.state_dict()},strict=True)
    assert ns['state_sha256'](lp)==original['lpips_state_sha256']
    return model.cuda(),dc.cuda().eval().requires_grad_(False),lp.cuda().eval().requires_grad_(False),scale.cuda(),payload

class Population:
    def __init__(self,role,latent=False,limit=None):
        assert role in ('train','calibration')
        manifest=read(TMP/'manifest.json');receipt=read(DATA/'RESTORED.json')
        assert receipt['manifest_sha256']==sha(TMP/'manifest.json')
        rows=[r for r in manifest['files'] if r['destination'].startswith('rgb/'+role+'/')]
        rows.sort(key=lambda r:r['destination']);self.ids=[];images=[];features=[];self.bindings={}
        reg=read(CP/'parents/p/registration.json');expected=reg['train_ids' if role=='train' else 'calibration_ids']
        for row in rows:
            path=DATA/row['destination'];assert sha(path)==row['sha256'];self.bindings[str(path)]=row['sha256']
            v=torch.load(path,map_location='cpu',weights_only=True);ids=v['image_ids']
            images.append(v['targets_u8']);self.ids.extend(ids)
            if latent:
                path=DATA/'latent'/role/path.name
                old=f'/home/liulu/projects/VAR_COMM/outputs/VAR-LATENT-ENHANCEMENT-20260917/cache_v1/{role}/{path.name}'
                digest=reg['latent_cache_bindings'][role][old];assert sha(path)==digest
                f=torch.load(path,map_location='cpu',weights_only=True);assert f['image_ids']==ids
                features.append(f['F']);self.bindings[str(path)]=digest
            if limit and len(self.ids)>=limit:break
        count=(20000 if role=='train' else 1000) if limit is None else limit
        self.ids=self.ids[:count];assert self.ids==expected[:count] and len(self.ids)==count
        self.images=torch.cat(images)[:count].contiguous();assert self.images.dtype==torch.uint8
        self.F=torch.cat(features)[:count].float() if latent else None
    def __len__(self):return len(self.ids)
    def batch(self,ids,device='cuda:0'):return self.images[ids].to(device=device,dtype=torch.float32).div(255)
    def p_batch(self,ids,snrs,seeds,noise_fn):
        noise=np.stack([noise_fn('VAR-CONTINUOUS-1024|'+self.ids[int(i)],int(s),(1024,2)) for i,s in zip(ids,seeds)])
        return dict(F=self.F[ids].cuda(),target=self.batch(ids),snr=torch.as_tensor(snrs,device='cuda:0'),noise=torch.as_tensor(noise,device='cuda:0',dtype=torch.float32))

def swin_modules():
    sys.path.insert(0,str(ROOT/'experiments/external-comparison-20261004'))
    import swin_model,swin_protocol,swin_replay,swin_state
    swin_protocol.configure_phy(ROOT)
    # Only the new process's build destination changes; the PHY source/math do not.
    swin_protocol._PHY.ROOT=TMP/'native_build'
    return swin_model,swin_protocol,swin_replay,swin_state
