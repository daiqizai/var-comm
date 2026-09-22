from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from latent_enhancement.runtime import digest, model_paths, settings
from latent_enhancement_b.common import load_decoder, scale_statistics
from latent_enhancement_b.model import build_arms, render_received
from latent_enhancement_eval.runner import (
    arithmetic_receive_budget, arithmetic_transmit_budget, load_targets, raw_receive_budget,
    raw_transmit_budget, render_digital_candidate,
)
from latent_enhancement.latent import complete_latent, enhancement_noise, original_rgb
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.progressive import receive_whole, split_prefix, transmit_whole
from var_comm.study import seeded_noise
from var_comm.whole_entropy import encode_prefixes


PROJECT = Path(__file__).resolve().parents[5]
EXPERIMENT = PROJECT / "experiments/var-latent-enhancement-20260917"
CONFIG = EXPERIMENT / "timing/config.json"
OUTPUT_ROOT = PROJECT / "outputs/VAR-LATENT-ENHANCEMENT-20260917"


def read_json(path):
    return json.loads(Path(path).read_text())


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure():
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_arm(name, shape, scale, recipe, device):
    selected = read_json(OUTPUT_ROOT / f"stage_B_v1/training/selected_{name}.json")
    checkpoint = Path(selected["checkpoint"])
    if digest(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError(f"selected timing checkpoint changed: {name}")
    arms = build_arms(shape, scale, recipe).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arms.load_state_dict(state["arms"], strict=True)
    return arms[name].eval().requires_grad_(False), selected


def source_prepare(vae, image, device):
    with torch.no_grad():
        continuous = vae.quant_conv(vae.encoder(image))
        scales = vae.quantize.f_to_idxBl_or_fhat(continuous, to_fhat=False)
    source = [scale[0].cpu().numpy() for scale in scales]
    return continuous, source


def run(args):
    configure()
    config = read_json(CONFIG)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("timing requires authorized GPU0")
    targets = load_targets()
    paths = model_paths()
    vae, var = load_models(paths, device)
    decoder = load_decoder(vae, device)
    scale = scale_statistics(device)
    recipe = settings()["stage_B"]
    arms = {name: load_arm(name, (32, 16, 16), scale, recipe, device)[0] for name in (
        "enhancement512", "enhancement1024", "receiver_only_refiner")}
    source_indices = config["timing_sources"]
    snrs = config["snrs_db"]
    seeds = config["noise_seeds"]
    method_order = config["methods"]
    before = {"vae": state_sha256(vae), "var": state_sha256(var), "decoder": state_sha256(decoder),
              **{name: state_sha256(model) for name, model in arms.items()}}
    metadata_path = output / "metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps({"status": "TIMING_RUNNING", "config_sha256": digest(CONFIG),
            "sources": source_indices, "snrs_db": snrs, "seeds": seeds, "methods": method_order,
            "hardware": config["hardware"], "excluded": config["endpoints"]["excluded"], "new_holdout_used": False}, indent=2) + "\n")
    rows = []
    warmup = targets[source_indices[0]]
    warm_image = torch.from_numpy(warmup["pixels"][None].astype(np.float32) / 127.5 - 1).to(device)
    with torch.no_grad():
        warm_f, warm_source = source_prepare(vae, warm_image, device)
        warm_base = complete_latent(vae, var, warm_source[:8], int(warmup["target"]["class_index"]), device)
        warm_signal = transmit_whole(warm_source, int(warmup["target"]["class_index"]), 8)
        _ = receive_whole(warm_signal, 7)
        _ = decoder(warm_base)
    for source_index in source_indices:
        record = targets[source_index]
        label = int(record["target"]["class_index"])
        image = torch.from_numpy(record["pixels"][None].astype(np.float32) / 127.5 - 1).to(device)
        for snr in snrs:
            for seed in seeds:
                prepared = {}
                for method in method_order:
                    tx_start = time.perf_counter()
                    with torch.no_grad():
                        continuous, source = source_prepare(vae, image, device)
                        base_tx = complete_latent(vae, var, source[:8], label, device)
                        if method == "m8_D0" or method == "m8_Dc_only" or method == "m8_receiver_only_refiner":
                            signal = transmit_whole(source, label, 8)
                            tx_extra = "base"
                        elif method.startswith("m8_plus_latent_"):
                            uses = 512 if method.endswith("512") else 1024
                            arm_name = "enhancement512" if uses == 512 else "enhancement1024"
                            signal_base = transmit_whole(source, label, 8)
                            waveform = arms[arm_name].encoder(continuous - base_tx, base_tx)
                            waveform_np = waveform[0].cpu().numpy()
                            signal = np.concatenate((signal_base, waveform_np))
                            tx_extra = arm_name
                        else:
                            family = method.split("_N", 1)[0]
                            parts = method.split("_")
                            budget = int(parts[1][1:])
                            mode = int(parts[2][1:])
                            if family == "raw":
                                signal, _ = raw_transmit_budget(source, label, mode, budget)
                            else:
                                payloads = encode_prefixes(vae, var, source, label, device, modes=(7, 8, 9))
                                signal, _ = arithmetic_transmit_budget(payloads[mode], label, mode, budget)
                            tx_extra = family
                    sync(device)
                    tx_ms = (time.perf_counter() - tx_start) * 1000
                    rx_start = time.perf_counter()
                    with torch.no_grad():
                        if method in ("m8_D0", "m8_Dc_only", "m8_receiver_only_refiner") or method.startswith("m8_plus_latent_"):
                            base_signal = signal[:3060]
                            base_received = base_signal + seeded_noise(record["target"]["image_id"], seed, (3060, 2)) / np.sqrt(10 ** (snr / 10))
                            reception = receive_whole(base_received, snr)
                            if reception["label"] is None:
                                base_latent = None
                                status = torch.tensor([[0., 0., 0.]], device=device)
                            else:
                                base_latent = complete_latent(vae, var, reception["prefix"], reception["label"], device)
                                status = torch.tensor([[1., float(bool(reception["events"] and reception["events"][0]["accepted"])),
                                                        float(reception["header"]["mode"])]], device=device)
                            if method == "m8_D0":
                                final = torch.full((1, 3, 256, 256), .5, device=device) if base_latent is None else original_rgb(vae, base_latent)
                            elif method == "m8_Dc_only":
                                final = render_received(decoder, base_latent if base_latent is not None else torch.zeros_like(base_tx), status)
                            elif method == "m8_receiver_only_refiner":
                                final_latent = arms["receiver_only_refiner"].receiver(None, base_latent if base_latent is not None else torch.zeros_like(base_tx),
                                                                                       torch.tensor([snr], device=device), status)
                                final = render_received(decoder, final_latent, status)
                            else:
                                uses = 512 if method.endswith("512") else 1024
                                arm_name = "enhancement512" if uses == 512 else "enhancement1024"
                                added_signal = signal[3060:]
                                added_received = added_signal + enhancement_noise(record["target"]["image_id"], seed, uses) / np.sqrt(10 ** (snr / 10))
                                final_latent = arms[arm_name].receiver(torch.as_tensor(added_received[None], device=device, dtype=torch.float32),
                                                                       base_latent if base_latent is not None else torch.zeros_like(base_tx),
                                                                       torch.tensor([snr], device=device), status)
                                final = render_received(decoder, final_latent, status)
                            header_ok = int(status[0, 0].item())
                        else:
                            family = method.split("_N", 1)[0]
                            parts = method.split("_")
                            budget = int(parts[1][1:])
                            mode = int(parts[2][1:])
                            received = signal + seeded_noise(record["target"]["image_id"], seed, signal.shape) / np.sqrt(10 ** (snr / 10))
                            phy = raw_receive_budget(received, snr, budget) if family == "raw" else arithmetic_receive_budget(received, snr, budget)
                            renderer = "Dc"
                            final_np, candidate = render_digital_candidate(phy, family, renderer, vae, var, decoder, device, {})
                            final = torch.as_tensor(final_np[None], device=device)
                            header_ok = int(phy["header"]["accepted"])
                    sync(device)
                    rx_ms = (time.perf_counter() - rx_start) * 1000
                    rows.append({"source_index": source_index, "image_id": record["target"]["image_id"], "snr_db": snr,
                                 "noise_seed": seed, "method": method, "tx_ms": tx_ms, "rx_ms": rx_ms,
                                 "total_ms": tx_ms + rx_ms, "header_ok": header_ok, "tx_extra": tx_extra})
    with (output / "per_call.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    for method, values in grouped.items():
        summary.append({"method": method, "calls": len(values),
                        "tx_ms_mean": float(np.mean([float(v["tx_ms"]) for v in values])),
                        "rx_ms_mean": float(np.mean([float(v["rx_ms"]) for v in values])),
                        "total_ms_mean": float(np.mean([float(v["total_ms"]) for v in values])),
                        "tx_ms_p50": float(np.median([float(v["tx_ms"]) for v in values])),
                        "rx_ms_p50": float(np.median([float(v["rx_ms"]) for v in values])),
                        "total_ms_p50": float(np.median([float(v["total_ms"]) for v in values]))})
    summary.sort(key=lambda value: value["method"])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    after = {"vae": state_sha256(vae), "var": state_sha256(var), "decoder": state_sha256(decoder),
             **{name: state_sha256(model) for name, model in arms.items()}}
    if before != after:
        raise RuntimeError("frozen timing model state changed")
    (output / "completion.json").write_text(json.dumps({"status": "ONLINE_ENDPOINT_TIMING_COMPLETE", "calls": len(rows),
        "sources": source_indices, "snrs_db": snrs, "seeds": seeds, "new_holdout_used": False,
        "model_state_unchanged": True, "batch": 1, "metrics_excluded": True}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "online_timing_v1"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
