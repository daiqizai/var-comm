from __future__ import annotations

import json

import numpy as np
import torch

from latent_enhancement.latent import enhancement_noise
from latent_enhancement.runtime import digest
from latent_enhancement.training import LatentPopulation

from .common import CACHE, OUT_B, config_b


class MatchedPopulation:
    def __init__(self, population):
        self.source = LatentPopulation(CACHE, population)
        self.identifiers = self.source.identifiers
        self.shape = self.source.shape
        config = config_b()
        self.snrs = torch.tensor(config["snrs_db"], dtype=torch.float32)
        self.seeds = config["training_base_noise_seeds"] if population == "train" else config["calibration_noise_seeds"]
        latents, statuses, identifiers = [], [], []
        for path in sorted((OUT_B / "rx_cache" / population).glob("shard_*.pt")):
            receipt = json.loads(path.with_suffix(".json").read_text())
            if digest(path) != receipt["sha256"]:
                raise RuntimeError("actual-RX cache changed")
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["snrs_db"] != config["snrs_db"] or record["noise_seeds"] != self.seeds:
                raise RuntimeError("actual-RX SNR/noise population changed")
            latents.append(record["Fb_RX"])
            statuses.append(record["rx_status"])
            identifiers.extend(record["image_ids"])
        if identifiers != self.identifiers:
            raise RuntimeError("actual-RX source IDs not paired")
        self.latents = torch.cat(latents)
        self.statuses = torch.cat(statuses)
        if self.latents.shape != (len(self), len(self.snrs), len(self.seeds), *self.shape):
            raise RuntimeError("actual-RX grid shape mismatch")

    def __len__(self):
        return len(self.identifiers)

    def batch(self, indices, snr_indices, noise_indices, enhancement_seeds, device):
        values = self.source.batch(indices, device)
        values.pop("Fq")
        values["Fb_RX"] = self.latents[indices, snr_indices, noise_indices].to(device)
        values["rx_status"] = self.statuses[indices, snr_indices, noise_indices].to(device)
        values["snr_db"] = self.snrs[snr_indices].to(device)
        noise = np.stack([enhancement_noise(self.identifiers[int(index)], int(seed), 1024)
                          for index, seed in zip(indices, enhancement_seeds)])
        values["standard_noise"] = torch.as_tensor(noise, device=device, dtype=torch.float32)
        return values


def evaluation_grid(source_indices, snr_count, noise_count):
    sources = torch.as_tensor(source_indices, dtype=torch.long)
    indices = sources.repeat_interleave(snr_count * noise_count)
    snrs = torch.arange(snr_count).repeat_interleave(noise_count).repeat(len(sources))
    noises = torch.arange(noise_count).repeat(len(sources) * snr_count)
    return indices, snrs, noises


def metric_summary(values, source_indices, snr_indices):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("finite MSE/LPIPS/normalized-latent metrics required")
    transformed = np.stack((values[:, 0], values[:, 1], values[:, 2],
                            -10 * np.log10(np.maximum(values[:, 0], 1e-12)),
                            values[:, 0] + 0.1 * values[:, 1] + 0.01 * values[:, 2]), axis=1)
    names = ("mse", "lpips", "normalized_latent", "psnr_db", "utility")

    def average(mask):
        unique = np.unique(source_indices[mask])
        means = np.stack([transformed[mask & (source_indices == index)].mean(0) for index in unique])
        return {name: float(value) for name, value in zip(names, means.mean(0))}

    return {"all": average(np.ones(len(values), dtype=bool)),
            "per_snr": {str(index): average(snr_indices == index) for index in np.unique(snr_indices)}}


def update_monitor(summary, previous, tolerances):
    best = {}
    improved = not previous
    for scope, metrics in [("all", summary["all"]), *summary["per_snr"].items()]:
        for metric in ("utility", "psnr_db", "lpips"):
            key = scope + "/" + metric
            current = metrics[metric]
            if key not in previous:
                best[key] = current
                continue
            old = previous[key]
            if metric == "psnr_db":
                improved = improved or current - old > tolerances["PSNR_improvement_db"]
                best[key] = max(old, current)
            else:
                threshold = (abs(old) * tolerances["relative_utility_improvement"] if metric == "utility"
                             else tolerances["LPIPS_improvement"])
                improved = improved or old - current > threshold
                best[key] = min(old, current)
    return best, bool(improved)


def paired_stop(state, recipe, patience):
    return (state["step"] >= recipe["minimum_updates"] and
            all(value >= patience for value in state["plateau_checks"].values()))
