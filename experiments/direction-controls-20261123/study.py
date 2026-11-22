"""Materialize the exact research implementation and predeclare matched controls on PFS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

BASE = Path("/mnt/pfs/pfs-yc2F4O/modelTeam")
REPO = BASE / "code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1"
STUDY = BASE / "code/liulu/analysis/var_comm_directions_20261123"
TEMP = BASE / "liberai-tmp/liulu/liulu-var-comm/direction_controls_20261123_v1"
WEIGHTS = BASE / "liberai-checkpoint/liulu/var_comm/direction_controls_20261123_v1"
PYTHON = BASE / "code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python"
SWIN = BASE / "liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
SWIN_SHA = "53fad7c13546b3e60c582da37e7f7bf5d687c3dc4722691ae5880af31cf7bec2"
SHA = "ec608c7d89d78704b4efa7d32004fafec85f6766"
D2 = "swinjscc-mismatch-zpac-20261120"
D3 = "dir3-mimo-holorate-20261121"
D4 = "dir4-rf1bit-noma-20261122"
SPEC = {
    "noma": {"seed": 2026112301, "steps": 4500, "arms": ["own_denoise", "full"], "gpu": 1,
             "project": D4, "train": "train_dir4_rf1bit_noma.py", "method": "semsic_noma",
             "primary": "weak-user PSNR 3 users / 2 RX antennas, 13/8/4 dB", "seeds": [3001, 3002, 3003],
             "conditions": ["over3x2_13_8_4", "balanced_10_10", "nearfar_12_3"]},
    "holo": {"seed": 2026112302, "steps": 5000, "arms": ["identity", "full"], "gpu": 0,
             "project": D3, "train": "train_dir3_mimo_holorate.py", "method": "holorate_blind",
             "primary": "first 8/32 resource blocks, 10 dB PSNR", "seeds": [2001, 2002, 2003],
             "conditions": ["first8", "first16", "full", "contiguous16"]},
}
LOCAL_CONTROL = """            if getattr(self, "own_denoise_only", False):
                # Each user receives only its own two gated latent updates after joint LMMSE.
                # Keep the learned gate, auxiliary target and the original per-RB reliability.
                from dataclasses import replace
                own_outputs = []
                for u, (obs, cr, alpha, sigma) in enumerate(stage_outs):
                    gate_real = omega_list[u].unsqueeze(-1).expand(B, M, N_b).reshape(B, -1)
                    gate = torch.cat([gate_real, gate_real], dim=1).reshape_as(obs.u_restored)
                    own_latent = torch.lerp(obs.u_restored, z_clean_round[u], gate)
                    own_outputs.append((replace(obs, u_restored=own_latent), cr, alpha, sigma))
                stage_outs = own_outputs
                continue

"""


def regular_path(p):
    for parent in (p, *p.parents):
        if parent.is_symlink():
            raise RuntimeError(f"Symbolic link in path: {parent}")
        if parent.exists():
            mode = parent.stat()
            if stat.S_ISREG(mode.st_mode) and mode.st_nlink != 1:
                raise RuntimeError(f"File has multiple hard links: {parent}")


def save_once(p, data):
    regular_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_bytes() != data:
            raise RuntimeError(f"Existing identity differs: {p}")
        return
    with p.open("xb") as handle:
        handle.write(data)


def dump_once(p, obj):
    save_once(p, (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def edit(s, old, new):
    if s.count(old) != 1:
        raise RuntimeError(f"Source context must appear exactly once: {old}")
    return s.replace(old, new)


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def public_source(name, s):
    if name.endswith("/noma_semsic_blind.py"):
        anchor = "            denoised_latents = z_clean_round\n"
        s = edit(s, anchor, anchor + LOCAL_CONTROL)
    if name.endswith("/holorate_blind.py"):
        # Preserve the sign dither, all data/noise power rules, packing and reliability inputs.
        s = edit(s, "z_disp = torch.matmul(h256, z_unit * dither)", "z_disp = z_unit * dither if getattr(self, 'identity_spatial', False) else torch.matmul(h256, z_unit * dither)")
        s = edit(s, "return torch.matmul(h256, z_disp) * dither", "return z_disp * dither if getattr(self, 'identity_spatial', False) else torch.matmul(h256, z_disp) * dither")
    for track, q in SPEC.items():
        if name.endswith("/" + q["train"]):
            s = edit(s, "    trainable_params = ", "    model.modem." + ("own_denoise_only" if track == "noma" else "identity_spatial") + " = args.run_tag == '" + q["arms"][0] + "'\n    trainable_params = ")
            import re
            s, n = re.subn(r'cache_root=Path\("[^"]+/cache"\) / args.run_tag', "cache_root=Path(" + repr(str(TEMP / "cache" / track)) + ") / args.run_tag", s)
            if n != 1:
                raise RuntimeError("Train cache path identity")
    return s


def main():
    regular_path(SWIN)
    if hashlib.sha256(SWIN.read_bytes()).hexdigest() != SWIN_SHA:
        raise RuntimeError("Unexpected backbone weight")
    for p in (STUDY, TEMP, WEIGHTS):
        regular_path(p)
        p.mkdir(parents=True, exist_ok=True)
    names = git("ls-tree", "-r", "--name-only", SHA, *(f"experiments/{d}" for d in (D2, D3, D4))).decode().splitlines()
    manifest = {}
    for name in names:
        if not name.endswith(".py") or not ("/src/" in name or any(name.endswith("/"+q["train"]) for q in SPEC.values())):
            continue
        original = git("show", f"{SHA}:{name}")
        copied = public_source(name, original.decode()).encode()
        compile(copied, name, "exec")
        save_once(TEMP / "source" / name, copied)
        manifest[name] = {"git": hashlib.sha256(original).hexdigest(), "run": hashlib.sha256(copied).hexdigest()}
    dump_once(STUDY / "source_identity.json", manifest)
    dump_once(STUDY / "protocol.json", {
        "tracks": SPEC, "source_commit": SHA, "backbone_sha256": SWIN_SHA, "schema": 1,
        "training": "Common fresh seed per mechanism; identical init, sampling, AdamW 0.0003, batch 16, cosine, clipping and original val256 selector. Historical control run arguments are not fully archived. Old successful runs are retained.",
        "noma": "Both arms have two rounds, same prior and same active gate, same .35 latent MSE. The control gates local denoising without writing another user's waveform or lowering interference variance.",
        "holo": "Both arms keep triplet packing, fixed sign dither, original reference projections and per-group reliability; only the spatial Hadamard and its inverse differ.",
        "evaluation": "Test once per selected model; use all common500 and Kodak24 sources and three registered noise seeds, matching original seeds and tensors, plus historical stage1/raster as reference. Export per-source PSNR & DISTS and all pairs, 5000 paired source bootstrap draws. NOMA fixed source tuple shifts are inherited, not new independent combinations.",
        "limits": {"total_gpu_device_hours": 12, "max_wall_hours_per_train": 3, "new_training_runs": 4, "sweeps": 0},
        "formal_input": "Preexisting test sets observed in prior research; new mechanism controls are preregistered, not a fresh blinded holdout.",
    })
    print(f"{len(manifest)} verified source files, {STUDY}/protocol.json", flush=True)


if __name__ == "__main__":
    main()
