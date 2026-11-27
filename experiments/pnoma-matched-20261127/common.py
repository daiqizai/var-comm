"""Bind reads and training to the frozen four-user reference implementation and normal PFS files."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
B=Path("/mnt/pfs/pfs-yc2F4O/modelTeam")
SCRATCH=B/"liberai-tmp/liulu/liulu-var-comm/pnoma_matched_20261127_v1"
CHECK=B/"liberai-checkpoint/liulu/var_comm/pnoma_matched_20261127_v1"
PRIOR=B/"code/liulu/analysis/var_comm_replica_routing_20261127"
OLD=B/"code/liulu/analysis/var_comm_overloaded_semsic_20261124"
SNAP=SCRATCH.parent/"replica_routing_20261127_v1/snapshot"
UP=SCRATCH/"upstream"
PROTOCOL=json.loads((HERE/"protocol.json").read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    tmp=Path(str(path)+".partial")
    with tmp.open("x") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    tmp.replace(path)

def write_rows(path,data):
    assert data
    with path.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)

def append_row(path,row):
    fresh=not path.exists()
    with path.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if fresh:
            w.writeheader()
        w.writerow(row)

def verify_inputs():
    assert sha(SNAP/"manifest.json")==PROTOCOL["snapshot_sha256"]
    for rel,expected in json.loads((SNAP/"manifest.json").read_text())["files"].items():
        assert sha(SNAP/rel)==expected,rel
    vendor=json.loads((UP/"vendor_identity.json").read_text())
    for rel,entry in vendor["upstream"].items():
        assert sha(UP/rel)==entry["sha256"],rel
    for rel,content in vendor["namespace_shims"].items():
        assert (UP/rel).read_text()==content
    return sha(UP/"vendor_identity.json")

sys.path.insert(0,str(UP))
for package in sorted((SNAP/"experiments").iterdir()):
    sys.path.append(str(package/"src"))
from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, seed_everything
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.metrics import OfflineVisualEvaluator, compute_psnr_batch
from dir4_rn.noma_semsic_blind import sample_multiuser_uplink_channels
from dir6_mb.overloaded_noma_4x2 import physical_grantfree_4x2_mac
import torch
from adapter import AdaptedPNOMA, ReferenceReceiver

SEEDS=PROTOCOL["evaluation"]["channel_seeds"]
CONDITIONS=PROTOCOL["evaluation"]["conditions"]
PINS=json.loads((OLD/"events/compute.json").read_text())

def init(name):
    vendor=verify_inputs()
    out=HERE/name
    out.mkdir(exist_ok=False)
    keys={}
    for cpu in sorted(os.sched_getaffinity(0)):
        topo=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key=((topo/"physical_package_id").read_text(),(topo/"core_id").read_text())
        keys.setdefault(key,cpu)
    cpus=list(keys.values())[:4]
    os.sched_setaffinity(0,cpus)
    configure_runtime_env(SCRATCH/("env_"+name),4)
    gpu=admit_gpu(physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),min_free_mib=22000)
    configure_torch_determinism(4)
    seed_everything(PROTOCOL["training"]["seed"])
    write_json(out/"identity.json",dict(gpu=gpu,cpus=cpus,upstream=vendor,snapshot=sha(SNAP/"manifest.json"),
            torch=torch.__version__,cuda=torch.version.cuda,protocol=sha(HERE/"protocol.json"),time=time.time()))
    return out

def modem_for(seed):
    p=B/"liberai-checkpoint/liulu/var_comm/dir6_matched_breakthroughs_v1/semsic_3round"/f"seed_{seed}/best_indist.pt"
    assert sha(p)==PINS["checkpoint_sha256"][str(seed)]
    state=torch.load(p,map_location="cpu",weights_only=True)["model_state"]
    modem=ReferenceReceiver(mode="linear_1stage").cuda().eval().requires_grad_(False)
    modem.load_state_dict({k:state["modem."+k] for k in modem.state_dict()}, strict=True)
    return modem

def bank(n,seed,levels,layout):
    offset={"rotated":91000003,"development":1911013}[layout]
    gen=torch.Generator().manual_seed(730000+41*seed+int(levels[-1]*10)+offset)
    hr,hi=sample_multiuser_uplink_channels(n,4,2,32,.85,generator=gen)
    eps=torch.randn(n,2,32,384,2,generator=gen)
    powers=torch.stack([torch.roll(torch.tensor(levels,dtype=torch.float32),(g+SEEDS.index(seed))%4) for g in range(n)])
    return hr,hi,eps,powers

def wave_from_bank(tx,registered,st,en):
    hr,hi,eps,power=registered
    wave,_=physical_grantfree_4x2_mac(tx,power[st:en].cuda(),n_rx=2,
            h_mu_override=(hr[st:en],hi[st:en]),standard_noise_iq=eps[st:en])
    return wave

def source(split):
    data=load_split_u8(split,allow_holdout=split=="common500")
    images=data.images_u8
    pixels=[hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in images]
    if split in ("val256","common500"):
        ref=json.loads((PRIOR/("formal" if split=="common500" else "development")/"sources.json").read_text())
        assert ref["ids"]==data.source_ids and ref["pixels"]==pixels
    return images,data.source_ids,pixels

def chosen_model():
    record=json.loads((HERE/"train/freeze.json").read_text())
    p=CHECK/"frozen_four.pt"
    assert record["weight_sha256"]==sha(p)
    model=AdaptedPNOMA().cuda()
    model.projections.expand()
    model.projections.expand()
    model.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
    return model.eval()
