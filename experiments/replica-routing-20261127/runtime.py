"""Shared frozen experiment runtime, source identity checks, and channel replay."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "protocol.json").read_text())
GPU = int(os.environ["CUDA_VISIBLE_DEVICES"])
SCRATCH = Path(os.environ["STUDY_SCRATCH"])
PARENT = Path(os.environ["STUDY_PARENT"])
SNAP = SCRATCH / "snapshot"
sys.path.insert(0,str(PARENT))
from evaluate_replicas import BACKBONE, WEIGHTS, ReceiverTrace, PACKAGES, digest, record
assert digest(PARENT / "evaluate_replicas.py") == CFG["parent_evaluator_sha256"]
for package in PACKAGES:
    sys.path.insert(0,str(SNAP / "experiments" / package / "src"))
from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.metrics import OfflineVisualEvaluator, compute_psnr_batch
from dir4_rn.noma_semsic_blind import sample_multiuser_uplink_channels
from dir6_mb.overloaded_noma_4x2 import (OverloadedNOMA4x2Transceiver, build_walsh_16_basis,
    vec_to_real_ring, physical_grantfree_4x2_mac)
import torch

PARENT_STATS = json.loads((PARENT / "events/compute.json").read_text())

def init(stage):
    start = time.time()
    out = HERE / stage
    out.mkdir(exist_ok=False)
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topo/"physical_package_id").read_text(),(topo/"core_id").read_text())
        cores.setdefault(key,cpu)
    chosen = list(cores.values())[:4]
    assert len(chosen) == 4
    os.sched_setaffinity(0,chosen)
    configure_runtime_env(SCRATCH / ("runtime_" + stage),num_threads=4)
    gpu = admit_gpu(physical_gpu=GPU,min_free_mib=25000)
    configure_torch_determinism(4)
    assert digest(BACKBONE) == "53fad7c13546b3e60c582da37e7f7bf5d687c3dc4722691ae5880af31cf7bec2"
    record(out/"identity.json",dict(stage=stage,gpu=gpu,physical_cpus=chosen,torch=torch.__version__,
           cuda=torch.version.cuda,start_unix=start,registration=digest(HERE/"protocol.json"),
           snapshot=digest(SNAP/"manifest.json"),backbone=digest(BACKBONE),
           scripts={f.name:digest(f) for f in sorted(HERE.glob("*.py"))}))
    bb = load_swinjscc_backbone(BACKBONE,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device=torch.device("cuda:0")).eval()
    return out,bb,start

def weight(bb,seed):
    model = OverloadedNOMA4x2Transceiver(bb,method="semsic_3round").cuda().eval()
    p = WEIGHTS / f"seed_{seed}" / "best_indist.pt"
    assert digest(p) == PARENT_STATS["checkpoint_sha256"][str(seed)]
    params = torch.load(p,map_location="cpu",weights_only=True)["model_state"]
    assert set(params) == {k for k in model.state_dict() if not k.startswith("backbone.")}
    model.load_state_dict(params,strict=False)
    return model

def source_set(bb,split,out):
    data = load_split_u8(split,allow_holdout=split=="common500")
    images,ids = data.images_u8,data.source_ids
    n = {"val256":256,"common500":500}[split]
    assert len(set(ids)) == len(images) == n
    pixel = [hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in images]
    if split=="common500":
        assert hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()==PARENT_STATS["image_tensor_hashes"][split]
    record(out/"sources.json",{"ids":ids,"pixels":pixel})
    stems = []
    for batch in images.split(32):
        x = bb.encoder.patch_embed(batch.cuda().float()/255.)
        for layer in bb.encoder.layers:
            x = layer(x)
        stems.append(bb.encoder.norm(x).cpu())
    return images,ids,torch.cat(stems)

def make_bank(n,seed,levels,layout):
    offset = {"original":0,"development":1911013,"rotated":91000003}[layout]
    gen = torch.Generator(device="cpu").manual_seed(730000+41*seed+int(levels[-1]*10)+offset)
    hr,hi = sample_multiuser_uplink_channels(n,4,2,32,.85,generator=gen)
    eps = torch.randn((n,2,32,384,2),generator=gen)
    base = torch.tensor(levels,dtype=torch.float32)
    powers = torch.stack([torch.roll(base,(g+CFG["seeds"].index(seed))%4) if offset else base for g in range(n)])
    return hr,hi,eps,powers

def transmit(model,stems,bank,st,en):
    hr,hi,noise,powers = bank
    parts = [model.encode_single_user_stem(stems[4*st+u:4*en:4].cuda(),u) for u in range(4)]
    tx = torch.stack([p[0] for p in parts],dim=1)
    y,truth = physical_grantfree_4x2_mac(tx,powers[st:en].cuda(),n_rx=2,
                         h_mu_override=(hr[st:en],hi[st:en]),standard_noise_iq=noise[st:en])
    return y,tx,truth,[p[1] for p in parts]

def rows(path, records):
    if not records:
        raise ValueError(f"No rows for {path}")
    tmp = path.with_suffix(".partial")
    with tmp.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    tmp.rename(path)

def finish(out,start,**kw):
    record(out/"closed.json",dict(elapsed_s=time.time()-start,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                                 optimization_steps=0,**kw))
