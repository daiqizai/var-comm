"""Common frozen visual assets for the authorized N512 experiment."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'experiments/rx-posterior-step1-20260929'
os.environ['VAR_COMM_DECODER_GATE']=str(ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v2_repaired_20260921/decoder_gate.json')
sys.path.insert(0,str(OLD))
import run_preflight as environment
import rx_v3_common as old
import torch
from token_efficiency.digital_grid import population

OUT=ROOT/'outputs/EXTREME-BW-20260930-R1'
RESULT=ROOT/'results/extreme_bandwidth_20260930_R1'
SNRS=[1,4,7,13,19]
CAL_SEEDS=[4101,4102,4103]
DEV_SEEDS=[2001,2002,2003]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def setup():
    vae,var,stats,static=old.start()
    device=torch.device('cuda:0')
    decoder=old.b.load_decoder(vae,device)
    lp,dino,qmeta=old.load_quality()
    models=dict(vae=vae,var=var,decoder=decoder,lpips=lp,dino=dino)
    for model in models.values():model.eval().requires_grad_(False)
    identity=dict(models={name:old.b.state_sha256(model) for name,model in models.items()},quality=qmeta,
        decoder_gate=str(os.environ['VAR_COMM_DECODER_GATE']),
        decoder_gate_sha256=sha(os.environ['VAR_COMM_DECODER_GATE']),
        calibration_statistics_sha256=sha(old.b.OUT/'calibration_statistics.pt'))
    return dict(**models,stats=stats,static=static,device=device,identity=identity)


def load_calibration():
    F,Fq,T,ids,bindings=old.b.load_calibration()
    records,pixel_bindings=population('calibration')
    if len(records)!=1000 or ids!=[r['image_id'] for r in records]:
        raise RuntimeError('complete calibration source/latent identities differ')
    return dict(F=F,Fq=Fq,T=T,records=records,bindings=dict(latents=bindings,pixels=pixel_bindings))


def check():old.b.boundary()


def assert_frozen(loaded):
    for name,expected in loaded['identity']['models'].items():
        model=loaded[name]
        if old.b.state_sha256(model)!=expected:raise RuntimeError('frozen asset changed '+name)
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('frozen asset acquired a parameter gradient '+name)


def mismatch_permutation():
    import numpy as np
    rng=np.random.default_rng(20260930)
    while True:
        order=rng.permutation(100)
        if (order!=np.arange(100)).all():return order.tolist()
