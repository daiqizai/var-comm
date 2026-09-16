#!/usr/bin/env python3
"""Frozen selected receiver pair plus one preregistered spatial-context ablation."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available
from evaluate_hybrid_correction import Development, frozen_references
from var_comm.base_conditioning import BaseConditionedLink
from var_comm.hybrid_training import DigitalRenderer, evaluate
from var_comm.next_scale_prior import load_models
from var_comm.quality import load_quality_models
from var_comm.study import sha256, snapshot, verify_snapshot, write_csv, write_json


class ShuffledContext(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *arguments):
        previous = self.model.context_mode
        self.model.context_mode = "spatial_shuffle"
        try:
            return self.model(*arguments)
        finally:
            self.model.context_mode = previous


def load_selected(training, config, device, mode):
    metadata = json.loads((training / "metadata.json").read_text())
    verify_snapshot(metadata["bindings"])
    if not (training / "completion.json").exists():
        raise RuntimeError("selected-model evaluation must follow completed training and calibration")
    if mode == "full":
        selection = json.loads((training / "selection.json").read_text())
        selected = selection["selected"]
    else:
        checkpoint = torch.load(training / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        selected = {arm: {"step": checkpoint["new_step"], "training_reference_qualified": False, "selection_guard_passed": False}
                    for arm in config["arms"]}
        selection = {"selected": selected, "engineering_only": True}
        del checkpoint
    models, hashes = {}, {}
    for arm, choice in selected.items():
        path = training / "checkpoints" / f"step_{choice['step']:07d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = BaseConditionedLink(config["model"], arm, config["mechanism"]["context_ablation_seed"]).to(device).eval()
        model.load_state_dict(checkpoint["models"][arm], strict=True)
        models[arm], hashes[str(path.resolve())] = model, sha256(path)
        del checkpoint
    return models, selection, hashes, metadata


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "configs/hybrid_base_conditioning.json").read_text())
    device = torch.device("cuda:0")
    models, selection, hashes, metadata = load_selected(arguments.training, config, device, arguments.mode)
    write_json(output / "selection_frozen_before_development.json", {**selection, "mode": arguments.mode,
                "weights": hashes, "training_metadata_sha256": sha256(arguments.training / "metadata.json")})
    source_hashes = snapshot(output, [Path(__file__), ROOT / "src/var_comm/base_conditioning.py"])
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    quality_paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    vae, var = load_models(paths, device)
    perceptual, dino, _weights = load_quality_models(quality_paths, device)
    count = 100 if arguments.mode == "full" else 2
    population = Development(vae, var, device, count)
    references, reference_hashes = frozen_references(population, config)
    write_csv(output / "frozen_reference_rows.csv", references)
    write_json(output / "bindings.json", {"weights": hashes, "references": reference_hashes, "source_population": population.original_bindings})
    image_directory = output / "images"
    image_directory.mkdir()
    for index in range(count):
        np.save(image_directory / f"source_{index:04d}.npy", population.images[index].numpy())
    renderer = DigitalRenderer(vae, var, device, (population,))
    checks = {"maximum_actual_base_difference_from_frozen_old_receiver": 0.0}
    checked_bases = set()
    old = ROOT / "outputs/HYBRID-SOURCE-CORRECTION-20260915/development_001"

    def save(row, image, base):
        frame = f"{row['image_index']:04d}_{row['snr_db']:g}_{row['seed']}"
        path = image_directory / f"{row['arm']}_{frame}.npy"
        np.save(path, image)
        row["image_path"], row["image_sha256"] = str(path.relative_to(output)), sha256(path)
        row.pop("gain_mean")
        row["diagnostic_context_ablation"] = row["arm"] == "conditioned_spatial_shuffle"
        variant = "conditioned" if row["diagnostic_context_ablation"] else row["arm"]
        row["training_reference_qualified"] = selection["selected"][variant]["training_reference_qualified"]
        row["selected_new_updates"] = selection["selected"][variant]["step"]
        if frame not in checked_bases:
            expected = np.load(old / "images" / f"shortened_base_{frame}.npy", allow_pickle=False)
            difference = float(np.max(np.abs(base - expected)))
            checks["maximum_actual_base_difference_from_frozen_old_receiver"] = max(checks["maximum_actual_base_difference_from_frozen_old_receiver"], difference)
            if difference > 1e-6:
                raise RuntimeError("actual digital RX base changed; do not credit a different PHY/base to conditioning")
            checked_bases.add(frame)
    evaluated = {**models, "conditioned_spatial_shuffle": ShuffledContext(models["conditioned"]).eval()}
    rows = evaluate(evaluated, population, np.arange(count), config, renderer, perceptual, device, dino, save)
    write_csv(output / "per_frame.csv", rows)
    write_json(output / "actual_base_audit.json", {"status": "PASS", "frames": len(checked_bases), **checks,
               "TX_clean_basis_was_not_used_as_RX_context": True, "context_ablation_changes_only_feature_input": True})
    verify_snapshot(metadata["bindings"])
    verify_snapshot(source_hashes)
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise RuntimeError("selected checkpoint changed during development")
    write_json(output / "completion.json", {"status": "DEVELOPMENT_COMPLETE" if arguments.mode == "full" else "ENGINEERING_REPLAY_COMPLETE",
               "sources": count, "normal_rows": len(rows) * 2 // 3, "context_ablation_rows": len(rows) // 3,
               "frozen_reference_rows": len(references), "new_holdout_accessed": False,
               "per_frame_sha256": sha256(output / "per_frame.csv"), "model_selection_used_development": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / f"failure_{time.time_ns()}.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        raise
