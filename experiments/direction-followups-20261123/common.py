"""Paths and shared invariants for the paired SemSIC and waveform experiments."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


BASE = Path("/mnt/pfs/pfs-yc2F4O/modelTeam")
USER = BASE / "code/liulu"
STUDY = USER / "analysis/var_comm_direction_followups_20261123"
PRIOR = USER / "analysis/var_comm_directions_20261123"
TEMP_BASE = BASE / "liberai-tmp/liulu/liulu-var-comm"
OLD = TEMP_BASE / "direction_controls_20261123_v1"
TEMP = TEMP_BASE / "direction_followups_20261123_v1"
CKPT = BASE / "liberai-checkpoint/liulu/var_comm/direction_followups_20261123_v1"
PY = USER / "var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python"
D2 = "swinjscc-mismatch-zpac-20261120"
D3 = "dir3-mimo-holorate-20261121"
D4 = "dir4-rf1bit-noma-20261122"
D5 = "dir5-otfs-genvar-20261122"
SNAP = OLD / "producer_snapshots/published_v2"
SHARED = TEMP_BASE / "dir5_otfs_genvar_v1/cache"
SWIN = BASE / "liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
SWIN_HASH = "53fad7c13546b3e60c582da37e7f7bf5d687c3dc4722691ae5880af31cf7bec2"
SEED = 2026112307
STEPS = 7000
WAVE_ARMS = {"spread": "symp_otfs", "no_spread": "symp_ofdm_2d"}
SIC_ARMS = ("neither", "cancel", "covariance", "both")
PRIMARY = ("psnr_db", "dists")


def path_check(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symbolic link in {path}")
    if path.is_file() and path.stat().st_nlink != 1:
        raise ValueError(f"Multiple hard links: {path}")
    return path


def digest(path):
    h = hashlib.sha256()
    with path_check(path).open("rb") as handle:
        for buf in iter(lambda: handle.read(4 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def save(path, data):
    path_check(path).parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def record(path, obj):
    save(path, (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def read(path):
    return json.loads(path_check(path).read_text())


def table(path, rows):
    if not rows:
        raise ValueError(f"Empty table: {path}")
    path_check(path).parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def include_sources():
    dirs = [TEMP / "source" / name / "src" for name in (D4, D5)]
    dirs.extend(OLD / "source/experiments" / name / "src" for name in (D2, D3))
    sys.path[:0] = [str(x) for x in dirs]
    return os.pathsep.join(str(x) for x in dirs)


def load_state(model, path):
    import torch
    state = torch.load(path_check(path), map_location="cpu", weights_only=True)
    allowed = {name for name, par in model.named_parameters() if par.requires_grad}
    if set(state["state_dict"]) != allowed:
        raise ValueError("Adapter checkpoint parameters differ from the model")
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if unexpected or allowed.intersection(missing):
        raise ValueError("Incomplete adapter checkpoint")
    return state


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def single_replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}")
    return text.replace(old, new)


def gpu_snapshot():
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
    result = {}
    for row in csv.reader(subprocess.check_output(cmd, text=True).splitlines()):
        ix, uuid, model, util, free, temp = [v.strip() for v in row]
        result[int(ix)] = dict(uuid=uuid, name=model, util=int(util), free_mib=int(free), temperature_c=int(temp))
    return result
