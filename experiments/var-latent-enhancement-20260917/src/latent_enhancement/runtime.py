from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import yaml


EXPERIMENT = Path(__file__).resolve().parents[2]
ROOT = EXPERIMENT.parents[1]
CONFIG = EXPERIMENT / "configs/experiment.json"
OUTPUT = ROOT / "outputs/VAR-LATENT-ENHANCEMENT-20260917"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_torch(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(record, temporary)
    temporary.replace(path)


def settings():
    return json.loads(CONFIG.read_text())


def model_paths():
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    for name in ("vae_checkpoint", "var_checkpoint"):
        if digest(paths[name]) != paths[name + "_sha256"]:
            raise RuntimeError(f"official asset hash changed: {name}")
    return paths


def configure():
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def foreign_gpu_processes():
    output = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits",
    ], text=True)
    processes = []
    for line in output.splitlines():
        fields = line.split(",", 2)
        if fields and fields[0].strip().isdigit() and int(fields[0]) != os.getpid():
            processes.append({"pid": int(fields[0]), "description": line.strip()})
    return processes


class ResourceBusy(RuntimeError):
    pass


def require_available():
    processes = foreign_gpu_processes()
    if processes:
        raise ResourceBusy(json.dumps(processes))


def wait_for_available(status_path):
    while True:
        processes = foreign_gpu_processes()
        if not processes:
            return
        write_json(status_path, {"status": "WAITING_FOR_AUTHORIZED_GPU0", "foreign_processes": processes,
                                 "pid": os.getpid(), "timestamp": time.time()})
        time.sleep(30)


def snapshot(extra=()):
    files = [CONFIG, *sorted((EXPERIMENT / "src").rglob("*.py")), *map(Path, extra)]
    return {str(path): digest(path) for path in files}


def verify_snapshot(receipt):
    for path, expected in receipt.items():
        if digest(path) != expected:
            raise RuntimeError(f"running implementation changed: {path}")


def perceptual_model(device):
    import lpips

    paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    if digest(paths["alexnet_checkpoint"]) != paths["alexnet_checkpoint_sha256"]:
        raise RuntimeError("LPIPS trunk checkpoint changed")
    linear = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
    model = lpips.LPIPS(net="alex", pnet_rand=True, model_path=str(linear), verbose=False)
    weights = torch.load(paths["alexnet_checkpoint"], map_location="cpu", weights_only=True)
    model.net.load_state_dict({name: weights["features." + name.split(".", 1)[1]]
                               for name in model.net.state_dict()}, strict=True)
    return model.to(device).eval().requires_grad_(False)


def image_losses(predicted, target, perceptual):
    mse = (predicted - target).square().flatten(1).mean(1)
    lpips = perceptual(predicted * 2 - 1, target * 2 - 1).reshape(-1)
    return mse, lpips
