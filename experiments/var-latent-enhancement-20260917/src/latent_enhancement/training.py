from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from var_comm.prefix_training_data import read_image_population

from .runtime import digest


class LatentPopulation:
    def __init__(self, cache, population):
        self.images, self.labels, self.identifiers, self.bindings = read_image_population(population)
        parts = {name: [] for name in ("F", "Fq", "Fb_TX")}
        identifiers = []
        for path in sorted((Path(cache) / population).glob("shard_*.pt")):
            receipt = json.loads(path.with_suffix(".json").read_text())
            if digest(path) != receipt["sha256"]:
                raise RuntimeError(f"latent cache changed: {path}")
            record = torch.load(path, map_location="cpu", weights_only=True)
            identifiers.extend(record["image_ids"])
            for name in parts:
                parts[name].append(record[name])
        if identifiers != self.identifiers:
            raise RuntimeError("source/latent ordering mismatch")
        self.values = {name: torch.cat(values) for name, values in parts.items()}
        self.shape = tuple(self.values["F"].shape[1:])
        if any(value.dtype != torch.float32 or tuple(value.shape[1:]) != self.shape for value in self.values.values()):
            raise RuntimeError("latent shape/precision mismatch")

    def __len__(self):
        return len(self.images)

    def batch(self, indices, device):
        result = {name: values[indices].to(device) for name, values in self.values.items()}
        result["target"] = self.images[indices].to(device).float().div(255)
        return result


class PairedOrder:
    def __init__(self, count, seed):
        self.count = count
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(count, generator=self.generator)
        self.cursor = 0
        self.epochs = 0

    def next(self, batch_size):
        pieces = []
        remaining = batch_size
        while remaining:
            size = min(remaining, self.count - self.cursor)
            pieces.append(self.order[self.cursor:self.cursor + size])
            self.cursor += size
            remaining -= size
            if self.cursor == self.count:
                self.order = torch.randperm(self.count, generator=self.generator)
                self.cursor = 0
                self.epochs += 1
        return torch.cat(pieces)

    def state_dict(self):
        return {"count": self.count, "order": self.order.clone(), "cursor": self.cursor,
                "epochs": self.epochs, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["count"] != self.count:
            raise RuntimeError("training source count changed at resume")
        self.order = state["order"].clone()
        self.cursor = state["cursor"]
        self.epochs = state["epochs"]
        self.generator.set_state(state["generator"])


def draw_mixture(batch_size, generator):
    if batch_size % 4:
        raise ValueError("logical batch must support exact 50/25/25 mixture")
    branches = torch.tensor([0] * (batch_size // 2) + [1] * (batch_size // 4) + [2] * (batch_size // 4))
    branches = branches[torch.randperm(batch_size, generator=generator)]
    return branches, torch.rand(batch_size, generator=generator)


def mixed_input(batch, branches, alpha):
    shape = (-1, 1, 1, 1)
    interpolation = batch["Fb_TX"] + alpha.reshape(shape) * (batch["F"] - batch["Fb_TX"])
    return torch.where((branches == 0).reshape(shape), batch["F"],
                       torch.where((branches == 1).reshape(shape), batch["Fq"], interpolation))


def calibration_summary(metrics):
    summary = {}
    for branch, values in metrics.items():
        mse, lpips = values[:, 0], values[:, 1]
        summary[branch] = {"mse": float(mse.mean()), "psnr_db": float((-10 * np.log10(np.maximum(mse, 1e-12))).mean()),
                           "lpips": float(lpips.mean()), "utility": float((mse + 0.1 * lpips).mean())}
    summary["mixture_utility"] = (0.5 * summary["F"]["utility"] + 0.25 * summary["Fq"]["utility"]
                                   + 0.25 * summary["interpolation"]["utility"])
    return summary


def monitor(summary, previous, tolerances):
    current = {"mixture_utility": float(summary["mixture_utility"])}
    for branch in ("F", "Fq", "interpolation", "Fb_TX"):
        current[branch + "_psnr"] = summary[branch]["psnr_db"]
        current[branch + "_lpips"] = summary[branch]["lpips"]
    if not previous:
        return current, True
    improved = previous["mixture_utility"] - current["mixture_utility"] > abs(previous["mixture_utility"]) * tolerances["relative_utility_improvement"]
    best = {"mixture_utility": min(previous["mixture_utility"], current["mixture_utility"])}
    for branch in ("F", "Fq", "interpolation", "Fb_TX"):
        psnr_key, lpips_key = branch + "_psnr", branch + "_lpips"
        improved = improved or current[psnr_key] - previous[psnr_key] > tolerances["PSNR_improvement_db"]
        improved = improved or previous[lpips_key] - current[lpips_key] > tolerances["LPIPS_improvement"]
        best[psnr_key] = max(previous[psnr_key], current[psnr_key])
        best[lpips_key] = min(previous[lpips_key], current[lpips_key])
    return best, bool(improved)


def paired_interval(differences, seed=2026091704, resamples=10000):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("paired differences must be finite source-image values")
    generator = np.random.default_rng(seed)
    means = np.concatenate([values[generator.integers(0, len(values), size=(min(500, resamples - start), len(values)))].mean(1)
                            for start in range(0, resamples, 500)])
    return {"mean": float(values.mean()), "low95": float(np.quantile(means, 0.025)), "high95": float(np.quantile(means, 0.975)),
            "sources": len(values), "resamples": resamples, "seed": seed}
