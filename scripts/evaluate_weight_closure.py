#!/usr/bin/env python3
"""Evaluate actual frozen operating points; reuse matching 0.01 artifacts only."""

import argparse
import csv
from datetime import datetime
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
from var_comm.weight_closure import load_configs, specifications


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    closure, base, _runtime = load_configs()
    specs = specifications(closure)
    if arguments.mode == "full":
        frozen = json.loads((arguments.freeze / "frozen_comparisons.json").read_text())
        selected = frozen["models"]
        for path, expected in frozen["source_bindings"].items():
            if sha256(path) != expected:
                raise RuntimeError("calibration-frozen comparison changed before development")
    else:
        latest = torch.load(arguments.training / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        selected = {}
        for arm, item in specs.items():
            step = 20000 if item["reused_training"] else int(latest["new_step"])
            root = ROOT / closure["base_run"] / "training_001" if item["reused_training"] else arguments.training
            path = root / "checkpoints" / f"step_{step:07d}.pt"
            selected[arm] = {**item, "step": step, "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256(path),
                             "state_key": item["variant"] if item["reused_training"] else arm}
        frozen = {"engineering_only": True, "models": selected}
        del latest
    write_json(output / "frozen_before_development.json", frozen)
    sources = snapshot(output, [Path(__file__), ROOT / "src/var_comm/weight_closure.py"])
    metadata = json.loads((arguments.training / "metadata.json").read_text())
    verify_snapshot(metadata["bindings"])
    old_root = ROOT / closure["base_run"]
    old_selection = json.loads((old_root / "training_001/selection.json").read_text())["selected"]
    reusable = {arm for arm, item in selected.items() if specs[arm]["reused_training"] and item["step"] == old_selection[item["variant"]]["step"]}
    device = torch.device("cuda:0")
    models = {}
    for arm, point in selected.items():
        if sha256(point["checkpoint"]) != point["checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint changed")
        if arm in reusable:
            continue
        state = torch.load(point["checkpoint"], map_location="cpu", weights_only=False)
        model = BaseConditionedLink(base["model"], point["variant"], base["mechanism"]["context_ablation_seed"]).to(device).eval()
        model.load_state_dict(state["models"][point["state_key"]])
        models[arm] = model
        del state
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    quality_paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    vae, var = load_models(paths, device)
    perceptual, dino, _weights = load_quality_models(quality_paths, device)
    count = 100 if arguments.mode == "full" else 2
    population = Development(vae, var, device, count)
    reference, reference_hashes = frozen_references(population, base)
    write_csv(output / "frozen_reference_rows.csv", reference)
    write_json(output / "reference_bindings.json", reference_hashes)
    images = output / "images"
    images.mkdir()
    for index in range(count):
        np.save(images / f"source_{index:04d}.npy", population.images[index].numpy())
    renderer = DigitalRenderer(vae, var, device, (population,))
    maximum_base_error = 0.
    checked = set()

    def save_image(row, image, base_image):
        nonlocal maximum_base_error
        frame = f"{row['image_index']:04d}_{row['snr_db']:g}_{row['seed']}"
        path = images / f"{row['arm']}_{frame}.npy"
        np.save(path, image)
        row["image_path"], row["image_sha256"] = str(path.resolve()), sha256(path)
        row["lambda"] = selected[row["arm"]]["weight"]
        row["variant"] = selected[row["arm"]]["variant"]
        row["selected_new_updates"] = selected[row["arm"]]["step"]
        row["quality_reused"] = False
        if frame not in checked:
            previous = np.load(ROOT / "outputs/HYBRID-SOURCE-CORRECTION-20260915/development_001/images" / f"shortened_base_{frame}.npy", allow_pickle=False)
            maximum_base_error = max(maximum_base_error, float(np.abs(previous - base_image).max()))
            if maximum_base_error > 1e-6:
                raise RuntimeError("weight sweep changed actual digital receiver information")
            checked.add(frame)
    rows = evaluate(models, population, np.arange(count), base, renderer, perceptual, device, dino, save_image)
    for arm in sorted(reusable):
        variant = selected[arm]["variant"]
        for previous in read_rows(old_root / "development_001/per_frame.csv"):
            if previous["arm"] != variant or int(previous["image_index"]) >= count:
                continue
            row = {name: previous[name] for name in rows[0] if name in previous}
            row.update(arm=arm, **{"lambda": .01}, variant=variant, selected_new_updates=selected[arm]["step"],
                       quality_reused=True, gain_mean=1.0,
                       image_path=str((old_root / "development_001" / previous["image_path"]).resolve()), image_sha256=previous["image_sha256"])
            if sha256(row["image_path"]) != row["image_sha256"]:
                raise RuntimeError("reused 0.01 image changed")
            if set(row) != set(rows[0]):
                raise RuntimeError("reused result schema differs from new measured points")
            rows.append(row)
    if len(rows) != count * 5 * 3 * 6:
        raise RuntimeError("not all registered structures/weights have actual measured development coverage")
    write_csv(output / "per_frame.csv", rows)
    verify_snapshot(metadata["bindings"])
    verify_snapshot(sources)
    write_json(output / "completion.json", {"status": "WEIGHT_CLOSURE_DEVELOPMENT_COMPLETE" if arguments.mode == "full" else "ENGINEERING_REPLAY_COMPLETE",
               "source_images": count, "rows": len(rows), "reused_0p01_arms": sorted(reusable),
               "actual_base_max_error": maximum_base_error, "per_frame_sha256": sha256(output / "per_frame.csv"),
               "new_holdout": False, "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / f"failure_{time.time_ns()}.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        raise
