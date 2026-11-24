"""Paired four-user receiver intervention on one immutable received waveform."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from prepare_snapshot import PACKAGES


BASE = Path("/mnt/pfs/pfs-yc2F4O/modelTeam")
CHECKPOINTS = BASE / "liberai-checkpoint/liulu/var_comm"
BACKBONE = CHECKPOINTS / "dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
WEIGHTS = CHECKPOINTS / "dir6_matched_breakthroughs_v1/semsic_3round"
PROTOCOL = Path(__file__).with_name("protocol.json")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


class ReceiverTrace:
    """Accepts RX-side values only. No source pixels, true channel or transmit waveform."""
    def __init__(self, model, arm, reference_gates=None):
        self.model = model
        self.arm = arm
        self.reference_gates = reference_gates
        self.gates = {}
        self.subtractions = []

    def replica(self, step, user, ch, snr, clean):
        if self.arm == "latent":
            return clean
        if self.arm == "linear_fixed_gate":
            return ch.u_restored
        if self.arm != "rgb_reencode":
            raise ValueError(self.arm)
        # Preserve the receiver's measured amplitude and reliability. The local RGB
        # reconstruction supplies only the other user's waveform replica.
        norm = clean / ch.pwr.clamp_min(1.e-8).sqrt()[:, None, None]
        cleaned = replace(ch, u_restored=clean, u_mmse_restored=clean, u_norm=norm, z_norm=norm)
        rgb = self.model._decode_from_channel_out(cleaned, snr).clamp(0, 1)
        stem = self.model.extract_encoder_stem(rgb)
        _, encoded = self.model.encode_single_user_stem(stem, user)
        return encoded

    def gate(self, step, user, value):
        key = (step, user)
        if key in self.gates:
            raise ValueError("Duplicate receiver stage")
        applied = self.reference_gates[key] if self.arm == "linear_fixed_gate" else value
        self.gates[key] = applied.detach()
        return applied

    def capture(self, real, imag):
        self.subtractions.append((real.detach(), imag.detach()))


def score_interference(torch, build_basis, signals, gains, predictions):
    """Post-reception scorer. Strip the registered pilot/signature subspace from truth."""
    power = signals.square().sum(-1).mean(dim=(2, 3), keepdim=True).clamp_min(1.e-8)
    transmitted = signals[:, :3] / power[:, :3].sqrt().unsqueeze(-1)
    z = torch.view_as_complex(transmitted.contiguous())
    h = torch.complex(gains["g_true_r"][:, :3], gains["g_true_i"][:, :3])
    truth = torch.einsum("burm,bumn->brmn", h, z)
    w = build_basis(12, 384, device=z.device, dtype=z.real.dtype).to(z.dtype)
    truth = truth - torch.einsum("brmk,kn->brmn", torch.einsum("brmn,kn->brmk", truth, w), w)
    energy = truth.abs().square().mean(dim=(1, 2, 3)).clamp_min(1.e-12)
    scores = []
    for re, im in predictions:
        mse = (truth - torch.complex(re, im)).abs().square().mean(dim=(1, 2, 3))
        scores.append((10 * torch.log10((mse / energy).clamp_min(1.e-12))).cpu().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    for pkg in PACKAGES:
        sys.path.insert(0, str(args.snapshot / "experiments" / pkg / "src"))
    from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism
    configure_runtime_env(args.runtime, num_threads=4)
    gpu = admit_gpu(physical_gpu=args.gpu, min_free_mib=25000)
    import torch
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.metrics import OfflineVisualEvaluator, compute_psnr_batch
    from dir2_zpac.models import load_swinjscc_backbone
    from dir4_rn.noma_semsic_blind import sample_multiuser_uplink_channels
    from dir6_mb.overloaded_noma_4x2 import (
        OverloadedNOMA4x2Transceiver, build_walsh_16_basis, physical_grantfree_4x2_mac,
    )
    configure_torch_determinism(4)
    device = torch.device("cuda:0")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in (str(args.gpu), gpu["uuid"])
    cfg = json.loads(PROTOCOL.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    weight_hashes = {}
    image_hashes = {}
    row_count = 0
    admission = dict(gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda,
                     protocol_sha256=digest(PROTOCOL), snapshot_sha256=digest(args.snapshot / "snapshot.json"),
                     script_sha256=digest(Path(__file__)), backbone_sha256=digest(BACKBONE),
                     started_unix=started, wall_limit_seconds=7200)
    assert admission["backbone_sha256"] == "53fad7c13546b3e60c582da37e7f7bf5d687c3dc4722691ae5880af31cf7bec2"
    record(out / "admission.json", admission)
    with torch.inference_mode():
        backbone = load_swinjscc_backbone(BACKBONE, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device).eval()
        evaluator = OfflineVisualEvaluator(device, include_lpips=False)
        phases = ["dev64", *cfg["splits"]]
        for split in phases:
            data = load_split_u8(split, allow_holdout=split != "dev64")
            limit = 4 if split == "dev64" else cfg["splits"][split]
            imgs = data.images_u8[:limit]
            ids = data.source_ids[:limit]
            assert len(set(ids)) == len(ids) == len(imgs) == limit
            image_hashes[split] = hashlib.sha256(imgs.contiguous().numpy().tobytes()).hexdigest()
            stems = []
            for batch in imgs.split(32):
                x = backbone.encoder.patch_embed(batch.to(device=device, dtype=torch.float32) / 255.)
                for layer in backbone.encoder.layers:
                    x = layer(x)
                stems.append(backbone.encoder.norm(x).cpu())
            stems_all = torch.cat(stems)
            seeds = cfg["seeds"][:1] if split == "dev64" else cfg["seeds"]
            conditions = list(cfg["conditions"].items())[:1] if split == "dev64" else cfg["conditions"].items()
            for seed in seeds:
                model = OverloadedNOMA4x2Transceiver(backbone, method="semsic_3round").to(device).eval()
                ckpt = WEIGHTS / f"seed_{seed}" / "best_indist.pt"
                weight_hashes[str(seed)] = digest(ckpt)
                state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state"]
                expected = {k for k in model.state_dict() if not k.startswith("backbone.")}
                if set(state) != expected:
                    raise ValueError(f"Checkpoint mismatch: {sorted(set(state)^expected)[:10]}")
                missing, unexpected = model.load_state_dict(state, strict=False)
                assert all(n.startswith("backbone.") for n in missing) and not unexpected
                del state
                for condition, snrs in conditions:
                    n = limit // 4
                    gen = torch.Generator(device="cpu").manual_seed(730000 + seed * 41 + int(snrs[-1] * 10))
                    hr, hi = sample_multiuser_uplink_channels(n, 4, 2, 32, 0.85, generator=gen)
                    eps = torch.randn((n, 2, 32, 384, 2), generator=gen)
                    snr = torch.tensor(snrs, dtype=torch.float32, device=device)
                    path = out / f"{split}_{condition}_{seed}.csv"
                    fields = ["split", "condition", "seed", "group", "slot", "source", "arm", "psnr_db", "dists",
                              "interference_nmse_db_r1", "interference_nmse_db_r2", "interference_nmse_db_r3"]
                    with path.with_suffix(".partial").open("x", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        for st in range(0, n, 12):
                            en = min(st + 12, n)
                            bsz = en - st
                            in_stems = [stems_all[4*st+u:4*en:4].to(device) for u in range(4)]
                            tx = torch.stack([model.encode_single_user_stem(z, u)[0] for u, z in enumerate(in_stems)], dim=1)
                            observation, truth = physical_grantfree_4x2_mac(
                                tx, snr.expand(bsz, 4), n_rx=2,
                                h_mu_override=(hr[st:en], hi[st:en]), standard_noise_iq=eps[st:en])
                            def receive(trace):
                                model.modem.study_trace = trace
                                try:
                                    stages, _, aux = model.modem.blind_demix_users(observation, 4, model.latent_channels)
                                    return [model._decode_from_channel_out(ch, c).clamp(0, 1) for ch, c in stages], aux["active_hat"]
                                finally:
                                    model.modem.study_trace = None
                            control = receive(None)[0] if split == "dev64" else None
                            observed_before = observation.clone()
                            native_gates = None
                            native_activity = None
                            for arm in cfg["arms"]:
                                trace = ReceiverTrace(model, arm, native_gates if arm == "linear_fixed_gate" else None)
                                images, activity = receive(trace)
                                assert len(trace.gates) == 12 and len(trace.subtractions) == 3
                                assert torch.equal(observation, observed_before)
                                if arm == "latent":
                                    native_gates = trace.gates
                                    native_activity = activity
                                    if control is not None:
                                        assert all(torch.equal(x, y) for x, y in zip(images, control)), "Native path changed"
                                else:
                                    assert torch.equal(activity, native_activity)
                                if arm == "linear_fixed_gate":
                                    assert all(torch.equal(v, native_gates[k]) for k, v in trace.gates.items())
                                residual = score_interference(torch, build_walsh_16_basis, tx, truth, trace.subtractions)
                                for slot, recon in enumerate(images):
                                    target = imgs[4*st+slot:4*en:4].to(device=device, dtype=torch.float32) / 255.
                                    psnr = compute_psnr_batch(target, recon)
                                    dists = evaluator.dists_model(target, recon, require_grad=False, batch_average=False).reshape(-1)
                                    assert torch.isfinite(psnr).all() and torch.isfinite(dists).all()
                                    for j, (pval, dval) in enumerate(zip(psnr.tolist(), dists.tolist())):
                                        rec = dict(split=split, condition=condition, seed=seed, group=st+j, slot=slot,
                                                   source=ids[4*(st+j)+slot], arm=arm, psnr_db=pval, dists=dval)
                                        if slot == 3:
                                            for rnd in range(3):
                                                rec[f"interference_nmse_db_r{rnd+1}"] = residual[rnd][j]
                                        writer.writerow(rec)
                                        row_count += 1
                            stream.flush()
                    path.with_suffix(".partial").rename(path)
                    print(f"Closed {split}, {condition}, seed {seed}, {n} groups", flush=True)
                del model
            if split == "dev64":
                record(out / "qualification.json", {"native_equal": True, "gates_verified": 12,
                       "observation_immutable": True, "development_images": 4, "sha256": image_hashes[split]})
            del stems_all
    record(out / "compute.json", {"elapsed_seconds": time.time()-started, "start_unix": started,
           "end_unix": time.time(), "image_tensor_hashes": image_hashes, "checkpoint_sha256": weight_hashes,
           "peak_pytorch_reserved_mib": torch.cuda.max_memory_reserved()/2**20, "gpu_count": 1,
           "score_rows_including_qualification": row_count, "arms": cfg["arms"]})


if __name__ == "__main__":
    main()
