"""Shared resources and deterministic input, scoring and storage identities."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = Path("/mnt/pfs/pfs-yc2F4O/modelTeam")
ANALYSIS = BASE / "code/liulu/analysis"
TMP = BASE / "liberai-tmp/liulu/liulu-var-comm"
RUNTIME = TMP / "inband_image_frontiers_20261128_v1"
SNAP = TMP / "replica_routing_20261127_v1/snapshot"
PRIOR = ANALYSIS / "var_comm_pnoma_matched_20261127"
CHECK = BASE / "liberai-checkpoint/liulu/var_comm"
BACKBONE = CHECK / "dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
CFG = json.loads((HERE / "protocol.json").read_text())
for package in sorted((SNAP / "experiments").iterdir()):
    sys.path.append(str(package / "src"))

from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import configure_runtime_env, configure_torch_determinism, admit_gpu, seed_everything
from dir2_zpac.metrics import OfflineVisualEvaluator, compute_psnr_batch


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as file:
        for block in iter(lambda: file.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(p, data):
    p = Path(p)
    with p.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def rows(p, values):
    assert values
    with Path(p).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def setup(role):
    out = HERE / role
    out.mkdir(exist_ok=False)
    assert digest(SNAP / "manifest.json") == CFG["snapshot"]
    for n, expected in json.loads((SNAP / "manifest.json").read_text())["files"].items():
        assert digest(SNAP / n) == expected, n
    assert digest(BACKBONE) == CFG["swin"]
    core_map = {}
    for i in sorted(os.sched_getaffinity(0)):
        cpu = Path(f"/sys/devices/system/cpu/cpu{i}/topology")
        key = tuple((cpu / n).read_text().strip() for n in ("physical_package_id","core_id"))
        core_map.setdefault(key, i)
    cores = list(core_map.values())
    offset = int(os.environ["CUDA_VISIBLE_DEVICES"]) * 4
    selected = [cores[(offset+j) % len(cores)] for j in range(4)]
    os.sched_setaffinity(0, selected)
    configure_runtime_env(RUNTIME / ("cache_" + role), num_threads=4)
    gpu = admit_gpu(int(os.environ["CUDA_VISIBLE_DEVICES"]), min_free_mib=20000)
    configure_torch_determinism(4)
    seed_everything(281128)
    torch.cuda.set_per_process_memory_fraction(.22)
    save(out / "identity.json", {"gpu":gpu,"cores":selected,"pytorch":torch.__version__,"cuda":torch.version.cuda,
         "protocol":digest(HERE/"protocol.json"),"snapshot":CFG["snapshot"],"swin":CFG["swin"],
         "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32,"start":time.time()})
    return out


def source(split, out):
    data = load_split_u8(split, allow_holdout=split == "common500")
    values = data.images_u8
    pixels = [hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in values]
    ref = ANALYSIS / "var_comm_replica_routing_20261127" / ("development" if split=="val256" else "formal") / "sources.json"
    original = json.loads(ref.read_text())
    assert original["ids"] == data.source_ids and original["pixels"] == pixels
    ident = {"ids":data.source_ids,"pixels":pixels,"parent_manifest":digest(ref)}
    if split == "common500" and (out / "val256.json").exists():
        other = json.loads((out / "val256.json").read_text())
        assert set(pixels).isdisjoint(other["pixels"]) and set(data.source_ids).isdisjoint(other["ids"])
    save(out / (split+".json"),ident)
    return values, data.source_ids


def scorer():
    return OfflineVisualEvaluator("cuda:0", include_lpips=False).dists_model


def quality(x, y, dists):
    y = y.clamp(0,1)
    assert torch.isfinite(y).all()
    p = compute_psnr_batch(x,y)
    d = dists(x,y,require_grad=False,batch_average=False).reshape(-1)
    return p.cpu().numpy(),d.cpu().numpy()


def finish(out, **record):
    ident = json.loads((out / "identity.json").read_text())
    save(out / "complete.json", {**record,"wall_seconds":time.time()-ident["start"],
        "peak_gpu_mib":torch.cuda.max_memory_reserved()/2**20})

