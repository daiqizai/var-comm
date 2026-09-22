from __future__ import annotations

import json
import time

import numpy as np
import torch

from latent_enhancement.runtime import digest, image_losses, require_available, write_json

from .data import evaluation_grid, metric_summary
from .model import latent_errors, render_received


@torch.no_grad()
def calibrate(arms, decoder, original_decoder, perceptual, population, scale, output, step, full, device, microbatch):
    directory = output / "calibration"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{'full' if full else 'subset'}_{step:07d}"
    report_path = directory / (stem + ".json")
    array_path = directory / (stem + ".npz")
    if report_path.exists():
        result = json.loads(report_path.read_text())
        if digest(array_path) != result["arrays_sha256"]:
            raise RuntimeError("B calibration arrays changed")
        return result
    sources = torch.arange(len(population)) if full else torch.arange(100) * len(population) // 100
    indices, snr_indices, noise_indices = evaluation_grid(sources, len(population.snrs), len(population.seeds))
    enhancement_seeds = torch.tensor(population.seeds)[noise_indices]
    records = {name: [] for name in arms}
    if step == 0 and full:
        records.update({"original_m8_D0": [], "m8_Dc_only": []})
    arms.eval()
    started = time.perf_counter()
    header_failures = 0
    body_failures = 0
    for start in range(0, len(indices), microbatch):
        require_available()
        selection = slice(start, start + microbatch)
        batch = population.batch(indices[selection], snr_indices[selection], noise_indices[selection], enhancement_seeds[selection], device)
        header_failures += int((batch["rx_status"][:, 0] == 0).sum())
        body_failures += int(((batch["rx_status"][:, 0] == 1) & (batch["rx_status"][:, 1] == 0)).sum())
        for name, model in arms.items():
            latent, _ = model.receive_training_sample(batch["F"], batch["Fb_TX"], batch["Fb_RX"], batch["snr_db"], batch["rx_status"], batch["standard_noise"])
            predicted = render_received(decoder, latent, batch["rx_status"])
            mse, lpips = image_losses(predicted, batch["target"], perceptual)
            auxiliary = latent_errors(latent, batch["F"], scale, batch["rx_status"])
            records[name].append(torch.stack((mse, lpips, auxiliary), dim=1).cpu().numpy())
        if step == 0 and full:
            for name, renderer in (("original_m8_D0", original_decoder), ("m8_Dc_only", decoder)):
                predicted = render_received(renderer, batch["Fb_RX"], batch["rx_status"])
                mse, lpips = image_losses(predicted, batch["target"], perceptual)
                auxiliary = latent_errors(batch["Fb_RX"], batch["F"], scale, batch["rx_status"])
                records[name].append(torch.stack((mse, lpips, auxiliary), dim=1).cpu().numpy())
        if start % (microbatch * 100) == 0:
            write_json(output / "calibration_status.json", {"status": "CALIBRATING_STAGE_B", "step": step,
                "full": full, "frame_rows_completed_per_method": min(start + microbatch, len(indices)),
                "frame_rows_total_per_method": len(indices), "timestamp": time.time()})
    arrays = {name: np.concatenate(values) for name, values in records.items()}
    temporary = array_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, source_indices=indices.numpy(), snr_indices=snr_indices.numpy(), noise_indices=noise_indices.numpy(),
                 source_image_ids=np.asarray(population.identifiers), snrs_db=population.snrs.numpy(),
                 noise_seeds=np.asarray(population.seeds), **arrays)
    temporary.replace(array_path)
    summary = {name: metric_summary(values, indices.numpy(), snr_indices.numpy()) for name, values in arrays.items()}
    result = {"step": step, "role": "full_calibration" if full else "fixed_calibration_subset", "sources": len(sources),
              "snrs_db": population.snrs.tolist(), "per_snr_keys_are_indices_into_snrs_db": True,
              "frame_rows_per_method": len(indices), "summary": summary, "header_failures_per_method": header_failures,
              "body_crc_failures_per_method": body_failures, "all_failures_retained": True,
              "arrays_sha256": digest(array_path), "elapsed_seconds": time.perf_counter() - started, "DINO_used": False}
    write_json(report_path, result)
    print(f"B calibration {stem}: " + json.dumps({name: entry['all'] for name, entry in summary.items()}), flush=True)
    arms.train()
    return result
