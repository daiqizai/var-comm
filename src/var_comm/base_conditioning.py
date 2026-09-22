"""Matched spatial side-information adapters inside the existing residual decoder."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .hybrid_correction import ResidualLink


class BaseConditionedLink(ResidualLink):
    def __init__(self, specification, variant, shuffle_seed=2026091524):
        if variant not in ("unconditioned", "conditioned"):
            raise ValueError("only the two registered receiver arms are defined")
        super().__init__(specification, "add")
        self.variant = variant
        self.context_mode = "actual"
        self.context_size = int(specification["context_size"])
        if self.context_size != 64 or specification["context_channels"] != 16 or specification["injection_stages"] != [0, 1]:
            raise ValueError("this experiment freezes context geometry and injection stages")
        self.context_extractor = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(),
                                                nn.Conv2d(16, 16, 3, padding=1), nn.SiLU())
        self.context_projections = nn.ModuleList((nn.Conv2d(16, 96, 1), nn.Conv2d(16, 64, 1)))
        for projection in self.context_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        permutation = np.random.default_rng(shuffle_seed).permutation(64)
        self.register_buffer("context_permutation", torch.from_numpy(permutation), persistent=True)

    def latent_layout(self, observed):
        if observed.ndim != 2 or observed.shape[1] != 2220:
            raise ValueError("receiver requires exactly the paid 1110 complex observations")
        flat = observed.new_zeros(len(observed), 2560)
        return flat.index_copy(1, self.active_indices, observed).reshape(-1, 10, 16, 16)

    def spatial_permutation(self, image):
        tiles = image.reshape(len(image), 3, 8, 8, 8, 8).permute(0, 2, 4, 1, 3, 5)
        tiles = tiles.reshape(len(image), 64, 3, 8, 8).index_select(1, self.context_permutation)
        return tiles.reshape(len(image), 8, 8, 3, 8, 8).permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 64, 64)

    def spatial_context(self, latent, receiver_base):
        channel = torch.stack((latent[:, :3].mean(1), latent[:, 3:6].mean(1), latent[:, 6:].mean(1)), dim=1)
        channel = functional.interpolate(channel, size=(64, 64), mode="bilinear", align_corners=False)
        if self.variant == "conditioned":
            if receiver_base is None or receiver_base.shape != (len(latent), 3, 256, 256):
                raise ValueError("conditioned decoder needs the actual received digital base, not a TX target")
            base = functional.interpolate(receiver_base * 2 - 1, size=(64, 64), mode="area")
            if self.context_mode == "spatial_shuffle":
                base = self.spatial_permutation(base)
            elif self.context_mode != "actual":
                raise ValueError("unregistered conditioning mode")
        else:
            base = latent.new_zeros(len(latent), 3, 64, 64)
        return (self.context_extractor(channel) + self.context_extractor(base)) * .5

    def decode(self, observed, snrs, receiver_base=None):
        latent = self.latent_layout(observed)
        context = self.spatial_context(latent, receiver_base)
        condition = self.condition(snrs)
        value = self.decoder.input(latent)
        for block in self.decoder.bottleneck:
            value = block(value, condition)
        for index, stage in enumerate(self.decoder.stages):
            value = stage(value, condition)
            if index < 2:
                resized = functional.interpolate(context, size=value.shape[-2:], mode="area")
                value = value + self.context_projections[index](resized)
        return 2 * torch.sigmoid(self.decoder.output(functional.silu(self.decoder.output_norm(value)))) - 1

    def forward(self, residual, base, snrs, standard_noise, features, header_usable):
        transmitted = self.encode(residual, snrs)
        if standard_noise.shape != transmitted.shape:
            raise ValueError("noise must cover only the registered analog coordinates")
        observed = transmitted + standard_noise / torch.pow(10., snrs[:, None] / 20)
        correction = self.decode(observed, snrs, base)
        image, gain = self.fuse(base, correction, features, header_usable)
        return image, transmitted, gain


def restore_parent(model, checkpoint, config):
    parent_state = checkpoint["models"][config["initializer_arm"]]
    missing, unexpected = model.load_state_dict(parent_state, strict=False)
    expected = {name for name in model.state_dict() if name.startswith(("context_extractor.", "context_projections."))}
    expected.add("context_permutation")
    if set(missing) != expected or unexpected:
        raise RuntimeError("warm start changed an original E/D parameter or buffer")
    parent_names = [name for name, _parameter in model.named_parameters() if name in parent_state]
    fresh_names = [name for name, _parameter in model.named_parameters() if name not in parent_state]
    parameters = dict(model.named_parameters())
    settings = config["training"]
    optimizer = torch.optim.AdamW([
        {"params": [parameters[name] for name in parent_names]},
        {"params": [parameters[name] for name in fresh_names]},
    ], lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    old = checkpoint["optimizers"][config["initializer_arm"]]
    if len(old["param_groups"]) != 1 or len(old["param_groups"][0]["params"]) != len(parent_names):
        raise RuntimeError("parent Adam parameter ordering no longer matches the frozen E/D")
    original_order = [name for name in parent_state if name in parameters]
    if original_order != parent_names:
        raise RuntimeError("cannot associate parent Adam with parameter names safely")
    state = optimizer.state_dict()
    target_ids = state["param_groups"][0]["params"]
    old_ids = old["param_groups"][0]["params"]
    state["state"] = {target: copy.deepcopy(old["state"][previous]) for previous, target in zip(old_ids, target_ids) if previous in old["state"]}
    state["param_groups"][0] = {**copy.deepcopy(old["param_groups"][0]), "params": target_ids}
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        if group["lr"] != settings["learning_rate"] or group["weight_decay"] != settings["weight_decay"]:
            raise RuntimeError("this experiment does not change the registered optimizer recipe")
    return optimizer, {"shared_names": parent_names, "new_names": fresh_names}


def summarize_regions(rows, config):
    result = {}
    regions = {"mechanism": config["mechanism_snrs_db"], "primary": config["primary_snrs_db"], "high": [13., 19.]}
    regions.update({f"snr_{snr:g}": [snr] for snr in config["snrs_db"]})
    for arm in sorted({row["arm"] for row in rows}):
        result[arm] = {}
        for region, snrs in regions.items():
            chosen = [row for row in rows if row["arm"] == arm and float(row["snr_db"]) in snrs]
            if chosen:
                fields = [name for name in ("psnr_db", "lpips", "mse", "dino") if name in chosen[0]]
                result[arm][region] = {name: float(np.mean([float(row[name]) for row in chosen])) for name in fields}
                result[arm][region]["frames"] = len(chosen)
    return result


def stopping_decision(summaries, step, config):
    ordered = sorted(int(value) for value in summaries if int(value) <= step)
    stalls, intervals = 0, []
    settings = config["stopping"]
    for previous, current in zip(ordered[:-1], ordered[1:]):
        improvements = {}
        for arm in config["arms"]:
            before, after = summaries[str(previous)][arm]["mechanism"], summaries[str(current)][arm]["mechanism"]
            lpips_gain = before["lpips"] - after["lpips"]
            psnr_gain = after["psnr_db"] - before["psnr_db"]
            progress = lpips_gain >= settings["minimum_LPIPS_improvement_per_full_interval"] or (
                psnr_gain >= settings["alternative_minimum_PSNR_improvement_db"] and
                -lpips_gain <= settings["alternative_maximum_LPIPS_worsening"])
            improvements[arm] = {"progress": bool(progress), "LPIPS_gain": lpips_gain, "PSNR_gain": psnr_gain}
        stalls = 0 if any(item["progress"] for item in improvements.values()) else stalls + 1
        intervals.append({"from": previous, "to": current, "arms": improvements, "consecutive_both_stalls": stalls})
    at_cap = step >= config["training"]["maximum_new_updates"]
    plateau = step >= config["training"]["minimum_new_updates"] and stalls >= settings["both_arms_stalled_intervals_required"]
    return {"stop": bool(at_cap or plateau), "reason": "paired_plateau" if plateau else "budget_cap" if at_cap else "continue_paired",
            "new_updates_per_arm": step, "consecutive_both_stalls": stalls, "intervals": intervals,
            "cap_is_not_convergence_proof": bool(at_cap and not plateau)}


def select_checkpoints(summaries, config):
    choices = {}
    for arm in config["arms"]:
        initial = summaries["0"][arm]["mechanism"]
        candidates = []
        for step, summary in summaries.items():
            if int(step) not in config["selection"]["eligible_steps"]:
                continue
            metrics = summary[arm]["mechanism"]
            violation = initial["psnr_db"] - config["selection"]["maximum_PSNR_drop_vs_common_initial_db"] - metrics["psnr_db"]
            candidates.append({"step": int(step), "mechanism_calibration": metrics, "PSNR_guard_violation": violation,
                               "selection_guard_passed": violation <= 0})
        if not candidates:
            raise RuntimeError("no completed new calibration checkpoint is eligible")
        valid = [point for point in candidates if point["selection_guard_passed"]]
        if valid:
            choice = min(valid, key=lambda value: (value["mechanism_calibration"]["lpips"], -value["mechanism_calibration"]["psnr_db"], value["step"]))
        else:
            choice = min(candidates, key=lambda value: (value["PSNR_guard_violation"], value["mechanism_calibration"]["lpips"], value["step"]))
        choice["training_reference_qualified"] = bool(choice["selection_guard_passed"] and
            choice["mechanism_calibration"]["lpips"] <= initial["lpips"] + config["selection"]["control_maximum_LPIPS_worsening_vs_initial"])
        choices[arm] = choice
    return choices
