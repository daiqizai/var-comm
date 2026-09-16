"""Paired data, actual digital reception, and calibration for the fixed hybrid."""

from collections import OrderedDict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .hybrid_correction import receive_digital, receiver_features
from .prefix_training_data import read_image_population
from .progressive import complete_image, prefix_key, split_prefix
from .study import seeded_noise, sha256


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915"


class CachedPopulation:
    def __init__(self, cache, population, verify=True):
        directory = Path(cache) / population
        receipt = json.loads((directory / "completion.json").read_text())
        if verify:
            for name, expected in receipt["hashes"].items():
                if sha256(directory / name) != expected:
                    raise RuntimeError(f"hybrid source cache changed: {name}")
        self.records = json.loads((directory / "population.json").read_text())
        images, labels, identifiers, _bindings = read_image_population(population, verify=verify)
        self.images = images[:len(self.records)]
        self.labels = labels[:len(self.records)].numpy()
        self.ids = identifiers[:len(self.records)]
        if self.ids != [row["image_id"] for row in self.records]:
            raise RuntimeError("hybrid cache and historical source identities differ")
        self.bases = np.load(directory / "bases.npy", mmap_mode="r")
        self.tokens = np.load(directory / "tokens.npy", mmap_mode="r")
        self.digital = np.load(directory / "digital.npy", mmap_mode="r")
        if not len(self.ids) == len(self.bases) == len(self.tokens) == len(self.digital) == receipt["sources"]:
            raise RuntimeError("incomplete source cache")

    def __len__(self):
        return len(self.ids)


class DigitalRenderer:
    def __init__(self, vae, var, device, populations=(), maximum_error_images=2048):
        self.vae, self.var, self.device = vae, var, device
        self.clean = {}
        self.errors = OrderedDict()
        self.maximum = maximum_error_images
        self.calls = 0
        self.cache_hits = 0
        for population in populations:
            for index, label in enumerate(population.labels):
                key = prefix_key(split_prefix(population.tokens[index], 7), int(label))
                self.clean[key] = (population, index)

    def render(self, received):
        if not received["header_usable"]:
            return np.full((3, 256, 256), 0.5, dtype=np.float32)
        key = prefix_key(received["prefix"], received["label"])
        if key in self.clean:
            self.cache_hits += 1
            population, index = self.clean[key]
            return np.asarray(population.bases[index])
        if key in self.errors:
            self.cache_hits += 1
            self.errors.move_to_end(key)
            return self.errors[key]
        value = complete_image(self.vae, self.var, received["prefix"], received["label"], self.device)
        self.calls += 1
        self.errors[key] = value
        if len(self.errors) > self.maximum:
            self.errors.popitem(last=False)
        return value


def make_batch(population, indices, snrs, noise, renderer, device):
    indices = np.asarray(indices, dtype=np.int64)
    snrs = np.asarray(snrs, dtype=np.float64)
    if noise.shape != (len(indices), 3060, 2):
        raise ValueError("each physical frame must have one complete standard-noise vector")
    images = population.images[torch.as_tensor(indices)].to(device).float().div(255)
    tx_base = torch.from_numpy(np.array(population.bases[indices], copy=True)).to(device)
    outputs, features, usable, events = [], [], [], []
    for local, (index, snr) in enumerate(zip(indices, snrs)):
        observed = population.digital[index].astype(np.float64) + noise[local, :1950] / np.sqrt(10 ** (snr / 10))
        result = receive_digital(observed, snr)
        outputs.append(renderer.render(result))
        features.append(receiver_features(result, snr))
        usable.append(result["header_usable"])
        correct = result["header_usable"] and result["label"] == int(population.labels[index])
        correct = bool(correct and np.array_equal(result["tokens"], population.tokens[index]))
        events.append({"image_id": population.ids[index], "image_index": int(index), "snr_db": float(snr),
                       "header_usable": bool(result["header_usable"]), "body_crc_accepted": bool(result["crc_accepted"]),
                       "source_correct": correct, "accepted_correct": bool(correct and result["crc_accepted"]),
                       "false_acceptance": bool(result["crc_accepted"] and not correct),
                       "received_distance": result["received_distance"],
                       "received_digital_sha256": hashlib.sha256(observed.tobytes()).hexdigest()})
    return {"images": images, "residual": images - tx_base,
            "base": torch.from_numpy(np.stack(outputs)).to(device),
            "snrs": torch.as_tensor(snrs, dtype=torch.float32, device=device),
            "noise": torch.from_numpy(noise[:, 1950:].reshape(len(indices), 2220).astype(np.float32)).to(device),
            "features": torch.from_numpy(np.stack(features)).to(device),
            "usable": torch.as_tensor(usable, device=device, dtype=torch.bool), "events": events}


def training_draw(step, population_size, config):
    settings = config["training"]
    batch_size = settings["batch_size"]
    absolute = (step - 1) * batch_size + np.arange(batch_size)
    indices = np.empty(batch_size, dtype=np.int64)
    for epoch in np.unique(absolute // population_size):
        permutation = np.random.default_rng(settings["data_seed"] + int(epoch)).permutation(population_size)
        selected = absolute // population_size == epoch
        indices[selected] = permutation[absolute[selected] % population_size]
    generator = np.random.default_rng(np.random.SeedSequence([settings["noise_seed"], step]))
    snrs = generator.choice(config["snrs_db"], batch_size)
    noise = generator.standard_normal((batch_size, 3060, 2))
    return indices, snrs, noise


def model_forward(model, batch):
    return model(batch["residual"], batch["base"], batch["snrs"], batch["noise"], batch["features"], batch["usable"])


@torch.no_grad()
def evaluate(models, population, indices, config, renderer, perceptual, device, dino=None, image_callback=None):
    from .quality import dino_features

    previous_modes = {arm: model.training for arm, model in models.items()}
    for model in models.values():
        model.eval()
    rows = []
    role = "development" if image_callback is not None else "calibration"
    seeds = config[role + "_seeds"]
    batch_size = config["training"]["batch_size"]
    items = [(int(index), float(snr), int(seed)) for index in indices for snr in config["snrs_db"] for seed in seeds]
    try:
        for offset in range(0, len(items), batch_size):
            entries = items[offset:offset + batch_size]
            chosen, snrs, repeated_seeds = zip(*entries)
            noise = np.stack([seeded_noise(population.ids[index], seed, (3060, 2)) for index, seed in zip(chosen, repeated_seeds)])
            batch = make_batch(population, chosen, snrs, noise, renderer, device)
            source_features = dino_features(dino, batch["images"]) if dino is not None else None
            for arm, model in models.items():
                image, transmitted, gain = model_forward(model, batch)
                if not torch.isfinite(image).all():
                    raise FloatingPointError("nonfinite evaluation output; do not drop the transmission")
                mse = (image - batch["images"]).square().flatten(1).mean(1)
                lpips = perceptual(image * 2 - 1, batch["images"] * 2 - 1).reshape(-1)
                similarity = None
                if dino is not None:
                    similarity = torch.nn.functional.cosine_similarity(dino_features(dino, image), source_features)
                energies = transmitted.square().sum(1)
                if torch.max(torch.abs(energies - 2220)) > 0.01:
                    raise RuntimeError("analog energy escaped the registered total")
                for local, event in enumerate(batch["events"]):
                    row = {**event, "seed": repeated_seeds[local], "arm": arm,
                           "mse": float(mse[local]), "psnr_db": float(-10 * mse[local].clamp_min(1e-12).log10()),
                           "lpips": float(lpips[local]), "gain_mean": float(gain[local].mean()),
                           "complex_uses": 3060, "total_energy": 3900 + float(energies[local])}
                    if similarity is not None:
                        row["dino"] = float(similarity[local])
                    if image_callback is not None:
                        image_callback(row, image[local].cpu().numpy(), batch["base"][local].cpu().numpy())
                    rows.append(row)
    finally:
        for arm, model in models.items():
            model.train(previous_modes[arm])
    return rows


def aggregate(rows, primary_snrs):
    result = {}
    for arm in sorted({row["arm"] for row in rows}):
        result[arm] = {}
        for region, snrs in [("primary", primary_snrs), ("high", [13., 19.])] + [
                (f"snr_{snr:g}", [snr]) for snr in sorted({float(row["snr_db"]) for row in rows})]:
            selected = [row for row in rows if row["arm"] == arm and float(row["snr_db"]) in snrs]
            if selected:
                result[arm][region] = {name: float(np.mean([float(row[name]) for row in selected]))
                                       for name in ("mse", "psnr_db", "lpips")}
                result[arm][region]["frames"] = len(selected)
    return result


def reference_calibration(identifiers, config):
    policies = json.loads((STUDY / "POLICIES_001/policies.json").read_text())["actions"]
    identifiers = set(identifiers)
    results = {"raw": [], "arithmetic": []}
    with (STUDY / "CALIBRATION_001/per_frame.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] not in identifiers or float(row["snr_db"]) not in config["primary_snrs_db"]:
                continue
            family = row["family"]
            if int(row["mode"]) == policies[family]["quality"][row["snr_db"]]:
                results[family].append(row)
    expected = len(identifiers) * len(config["primary_snrs_db"]) * len(config["calibration_seeds"])
    for family, rows in results.items():
        if len(rows) != expected:
            raise RuntimeError("frozen digital reference coverage differs from hybrid calibration")
        keys = {(row["image_id"], float(row["snr_db"]), int(row["seed"])) for row in rows}
        expected_keys = {(identifier, snr, seed) for identifier in identifiers for snr in config["primary_snrs_db"]
                         for seed in config["calibration_seeds"]}
        if keys != expected_keys:
            raise RuntimeError("calibration noise/source pairing differs")
    return {family: {name: float(np.mean([float(row[name]) for row in rows])) for name in ("psnr_db", "lpips")}
            for family, rows in results.items()}


def choose_checkpoint(summaries, references, config):
    margin = config["selection"]["lpips_noninferiority_margin_absolute"]
    selected = {}
    for arm in config["arms"]:
        candidates = []
        for step, summary in summaries.items():
            primary = summary[arm]["primary"]
            violation = max(primary["lpips"] - reference["lpips"] - margin for reference in references.values())
            candidates.append({"step": int(step), "primary": primary, "maximum_constraint_violation": violation,
                               "feasible_on_calibration": bool(violation <= 0)})
        feasible = [candidate for candidate in candidates if candidate["feasible_on_calibration"]]
        if feasible:
            chosen = min(feasible, key=lambda value: (-value["primary"]["psnr_db"], value["step"]))
        else:
            chosen = min(candidates, key=lambda value: (value["maximum_constraint_violation"], -value["primary"]["psnr_db"], value["step"]))
        selected[arm] = chosen
    return selected
