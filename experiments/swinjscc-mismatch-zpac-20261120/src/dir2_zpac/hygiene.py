"""PFS link-hygiene verification, cache isolation, determinism, and A800 GPU admission."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import stat
import subprocess
import tempfile
import time
from typing import Iterable


PFS_CODE_ROOT = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1")
PFS_CKPT_ROOT = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1")
PFS_TMP_ROOT = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_swinjscc_mismatch_v1")


def configure_runtime_env(cache_root: Path | str | None = None, num_threads: int = 6) -> dict[str, str]:
    """Redirect all temporary/model caches into isolated personal PFS tmp with zero symlinks."""
    root = Path(cache_root) if cache_root is not None else (PFS_TMP_ROOT / "cache")
    subdirs = {
        "TMPDIR": root / "tmp",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
        "TORCH_HOME": root / "torch",
        "HF_HOME": root / "hf",
        "MPLCONFIGDIR": root / "mpl",
        "XDG_CACHE_HOME": root / "xdg",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
        "TRITON_CACHE_DIR": root / "triton",
    }
    applied: dict[str, str] = {}
    for key, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
        applied[key] = str(path)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    tempfile.tempdir = str(subdirs["TMPDIR"])
    return applied


def configure_torch_determinism(num_threads: int = 6) -> None:
    """Enforce FP32 precision (TF32 disabled) and deterministic kernels."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(num_threads)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_no_links(roots: Iterable[Path | str]) -> dict:
    """Scan given PFS roots with lstat and assert 0 symlinks and 0 multi-hardlink files."""
    symlinks: list[str] = []
    hardlinks: list[dict] = []
    scanned_files = 0
    scanned_dirs = 0

    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists() and not root.is_symlink():
            continue
        # Check ancestors for symlinks too
        for anc in [root, *root.parents]:
            st_anc = os.lstat(anc)
            if stat.S_ISLNK(st_anc.st_mode):
                symlinks.append(str(anc))
        if root.is_symlink():
            symlinks.append(str(root))
            continue
        if root.is_file():
            st = os.lstat(root)
            scanned_files += 1
            if st.st_nlink > 1:
                hardlinks.append({"path": str(root), "nlink": int(st.st_nlink)})
            continue

        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            scanned_dirs += 1
            dpath = Path(dirpath)
            # Check directory entries that might be symlinks to directories
            keep_dirs: list[str] = []
            for dname in dirnames:
                p = dpath / dname
                st = os.lstat(p)
                if stat.S_ISLNK(st.st_mode):
                    symlinks.append(str(p))
                else:
                    keep_dirs.append(dname)
            dirnames[:] = keep_dirs

            for fname in filenames:
                p = dpath / fname
                st = os.lstat(p)
                if stat.S_ISLNK(st.st_mode):
                    symlinks.append(str(p))
                elif stat.S_ISREG(st.st_mode):
                    scanned_files += 1
                    if st.st_nlink > 1:
                        hardlinks.append({"path": str(p), "nlink": int(st.st_nlink)})

    report = {
        "status": "PASS" if (not symlinks and not hardlinks) else "FAIL",
        "timestamp_unix": time.time(),
        "roots": [str(r) for r in roots],
        "scanned_dirs": scanned_dirs,
        "scanned_files": scanned_files,
        "symlink_count": len(symlinks),
        "hardlink_count": len(hardlinks),
        "symlinks": symlinks[:50],
        "hardlinks": hardlinks[:50],
    }
    if symlinks or hardlinks:
        raise RuntimeError(f"Link hygiene check failed: {json.dumps(report, indent=2)}")
    return report


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def admit_gpu(
    physical_gpu: int | None = None,
    min_free_mib: int = 32000,
    max_temp_c: int = 82,
) -> dict:
    """Inspect nvidia-smi and verify safe free VRAM & temperature on shared A800 server."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    gpus = []
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "temperature_c": int(parts[3]),
                "utilization_pct": int(parts[4]),
                "memory_used_mib": int(parts[5]),
                "memory_free_mib": int(parts[6]),
            }
        )

    if physical_gpu is not None:
        chosen = next((g for g in gpus if g["index"] == physical_gpu), None)
        if chosen is None:
            raise RuntimeError(f"Requested GPU {physical_gpu} not found in {gpus}")
    else:
        eligible = [
            g for g in gpus if g["memory_free_mib"] >= min_free_mib and g["temperature_c"] <= max_temp_c
        ]
        if not eligible:
            raise RuntimeError(f"No GPU satisfies free>={min_free_mib} MiB and temp<={max_temp_c}C: {gpus}")
        eligible.sort(key=lambda g: (-g["memory_free_mib"], g["utilization_pct"], g["temperature_c"]))
        chosen = eligible[0]

    if chosen["memory_free_mib"] < min_free_mib or chosen["temperature_c"] > max_temp_c:
        raise RuntimeError(f"GPU {chosen['index']} failed admission: {chosen}")
    return chosen


def save_json(path: Path | str, payload: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
