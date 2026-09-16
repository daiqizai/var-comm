#!/usr/bin/env python3
"""Read-only verification after the registered fixed-weight pipeline completes."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available
from evaluate_communication_modes import Population
from var_comm.hybrid_correction import encode_digital, receive_digital
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.study import seeded_noise, sha256, write_csv, write_json


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load(path):
    return json.loads(Path(path).read_text())


def frame_key(row):
    return int(row["image_index"]), float(row["snr_db"]), int(row["seed"])


def history(directory):
    records = {}
    for path in sorted(directory.glob("updates_from_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                step = record["new_step"]
                if step in records:
                    raise RuntimeError("duplicate executed update requires manual resume audit")
                records[step] = record
    if set(records) != set(range(1, 20001)):
        raise RuntimeError("actual 20000-update opportunity history is incomplete")
    return records


def exact_tree(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise RuntimeError("warm-start optimizer or parameter keys differ")
        for name in first:
            exact_tree(first[name], second[name])
    elif isinstance(first, (tuple, list)):
        if len(first) != len(second):
            raise RuntimeError("warm-start optimizer groups differ")
        for first_item, second_item in zip(first, second):
            exact_tree(first_item, second_item)
    elif first != second:
        raise RuntimeError("warm-start optimizer setting differs")


def verify_model_history(root, base_root, frozen, closure):
    original = torch.load(base_root / "training_001/checkpoints/step_0000000.pt", map_location="cpu", weights_only=False)
    initial = torch.load(root / "training_001/checkpoints/step_0000000.pt", map_location="cpu", weights_only=False)
    final = torch.load(root / "training_001/checkpoints/step_0020000.pt", map_location="cpu", weights_only=False)
    for arm in initial["models"]:
        variant = frozen["models"][arm]["variant"]
        exact_tree(initial["models"][arm], original["models"][variant])
        exact_tree(initial["optimizers"][arm], original["optimizers"][variant])
        initial_state = initial["optimizers"][arm]["state"]
        final_state = final["optimizers"][arm]["state"]
        for parameter, state in final_state.items():
            start = float(initial_state.get(parameter, {}).get("step", 0))
            if float(state["step"]) - start != 20000:
                raise RuntimeError("Adam counter does not match actual update opportunities")
    del original, initial, final
    summaries = {step: load(root / "training_001/calibration" / f"step_{step:07d}_full/summary.json")
                 for step in closure["checkpoint_selection"]["eligible_steps"]}
    for arm, model in frozen["models"].items():
        chosen = min(summaries, key=lambda step: (summaries[step][arm]["primary"]["mse"] + model["weight"] * summaries[step][arm]["primary"]["lpips"], step))
        if model["step"] != chosen:
            raise RuntimeError("checkpoint choice is not the registered calibration-only rule")
    return {"initial_models_and_Adam_exactly_match": True, "final_Adam_counters_match_20000": True,
            "all_six_calibration_selections_independently_recomputed": True}


def verify_histories(root, base_root, closure):
    original = history(base_root / "training_001")
    updated = history(root / "training_001")
    maximum_loss_error = 0.
    for step in range(1, 20001):
        first, second = original[step], updated[step]
        for name in ("indices", "snrs_db", "noise_sha256"):
            if first[name] != second[name]:
                raise RuntimeError(f"unpaired training stream at {step}: {name}")
        for arm, scores in second["arms"].items():
            weight = scores["lambda"]
            if weight not in closure["trained_lambdas"]:
                raise RuntimeError("an unregistered weight was trained")
            error = abs(scores["loss"] - scores["mse"] - weight * scores["lpips"])
            maximum_loss_error = max(maximum_loss_error, error)
            if not all(np.isfinite(value) for value in scores.values()):
                raise RuntimeError("a nonfinite training observation was retained as success")
    if maximum_loss_error > 1e-6:
        raise RuntimeError("training loss differs from the only registered intervention")
    return {"exactly_paired_updates": 20000, "maximum_loss_formula_error": maximum_loss_error}


def verify_statistics(root, measured, references, closure):
    analysis = root / "analysis_001"
    expected_path = analysis / "expected_time_sharing_per_frame.csv"
    expected = rows(expected_path) if expected_path.exists() else []
    frozen = load(root / "CALIBRATION_FREEZE_001/frozen_comparisons.json")
    actual_index = {(row["arm"], row["image_id"], float(row["snr_db"]), int(row["seed"])): row for row in measured + references}
    mixtures = {item["reference_id"]: item for item in frozen["time_sharing"] if item["coverage"]}
    metrics = ("psnr_db", "lpips", "dino", "mse")
    maximum_expectation_error = 0.
    for row in expected:
        mixture = mixtures[row["arm"]]
        key = row["image_id"], float(row["snr_db"]), int(row["seed"])
        digital, deep = actual_index[(mixture["digital"], *key)], actual_index[("perceptual_deepjscc", *key)]
        probability = mixture["p_deep"]
        if not 0 <= probability <= 1 or float(row["p_deep"]) != probability:
            raise RuntimeError("development used an unfrozen selection probability")
        for metric in metrics:
            value = (1 - probability) * float(digital[metric]) + probability * float(deep[metric])
            maximum_expectation_error = max(maximum_expectation_error, abs(value - float(row[metric])))
    if len(expected) != 1500 * len(mixtures) or maximum_expectation_error > 1e-12:
        raise RuntimeError("expected frame-selection reference is incomplete or incorrect")
    all_rows = measured + references + expected
    identifiers = sorted({row["image_id"] for row in all_rows})
    regions = {"primary": [1., 4., 7.], "mechanism": [7., 13., 19.], "high": [13., 19.]}
    regions.update({f"snr_{snr:g}": [snr] for snr in (1., 4., 7., 13., 19.)})
    grouped = {}
    for row in all_rows:
        for region, snrs in regions.items():
            if float(row["snr_db"]) in snrs:
                grouped.setdefault((region, row["arm"], row["image_id"]), []).append(row)
    reduced = {}
    for row in rows(analysis / "quality_summary.csv"):
        for metric in metrics:
            values = np.asarray([np.mean([float(record[metric]) for record in grouped[row["region"], row["arm"], identifier]]) for identifier in identifiers])
            reduced[row["region"], row["arm"], metric] = values
            if abs(values.mean() - float(row[metric])) > 1e-10:
                raise RuntimeError("reported means are not source-aggregated")
    indices = np.random.default_rng(closure["system"]["bootstrap_seed"]).integers(100, size=(10000, 100))
    maximum_interval_error = 0.
    intervals = rows(analysis / "source_paired_intervals.csv")
    for row in intervals:
        differences = reduced[row["region"], row["arm"], row["metric"]] - reduced[row["region"], row["control"], row["metric"]]
        low, high = np.quantile(differences[indices].mean(1), [.025, .975])
        errors = (abs(differences.mean() - float(row["difference"])), abs(low - float(row["ci_low"])), abs(high - float(row["ci_high"])))
        maximum_interval_error = max(maximum_interval_error, *errors)
    if maximum_interval_error > 1e-10:
        raise RuntimeError("reported intervals differ from source-image paired bootstrap")
    return {"expectation_rows": len(expected), "intervals_recomputed": len(intervals),
            "maximum_expectation_error": maximum_expectation_error, "maximum_interval_error": maximum_interval_error}


def run(arguments):
    root = arguments.run.resolve()
    if not (root / "pipeline_001/completion.json").exists():
        raise RuntimeError("wait for the registered queue; do not audit partial performance")
    output = root / "independent_audit_001"
    output.mkdir(exist_ok=False)
    closure = load(ROOT / "configs/hybrid_weight_closure.json")
    base = load(ROOT / closure["base_config"])
    base_root = ROOT / closure["base_run"]
    for path, expected in load(root / "pipeline_001/bindings.json").items():
        if sha256(path) != expected:
            raise RuntimeError("registered pipeline inputs changed")
    freeze_path = root / "CALIBRATION_FREEZE_001/frozen_comparisons.json"
    frozen = load(freeze_path)
    if frozen != load(root / "development_001/frozen_before_development.json"):
        raise RuntimeError("development did not use the calibration-frozen decisions")
    for path, expected in frozen["source_bindings"].items():
        if sha256(path) != expected:
            raise RuntimeError("a calibration decision input changed")
    for model in frozen["models"].values():
        if sha256(model["checkpoint"]) != model["checkpoint_sha256"]:
            raise RuntimeError("the selected model changed")
    history_result = verify_histories(root, base_root, closure)
    model_history = verify_model_history(root, base_root, frozen, closure)
    quality = root / "development_001"
    measured = rows(quality / "per_frame.csv")
    references = rows(quality / "frozen_reference_rows.csv")
    if len(measured) != 9000 or len(references) != 9000:
        raise RuntimeError("measured workpoint or strong-system coverage is incomplete")
    lookup = {(row["arm"], *frame_key(row)): row for row in measured}
    if len(lookup) != 9000:
        raise RuntimeError("duplicate measured frame")
    expected = {(index, snr, seed) for index in range(100) for snr in base["snrs_db"] for seed in base["development_seeds"]}
    for arm in frozen["models"]:
        if {frame_key(row) for row in measured if row["arm"] == arm} != expected:
            raise RuntimeError("a registered source/noise/SNR or failed frame was removed")
    failure_summary = []
    for arm in frozen["models"]:
        for snr in base["snrs_db"]:
            selected = [row for row in measured if row["arm"] == arm and float(row["snr_db"]) == snr]
            failure_summary.append({"arm": arm, "snr_db": snr, "transmissions": len(selected),
                "header_failures": sum(row["header_usable"] == "False" for row in selected),
                "body_crc_failures_with_usable_header": sum(row["header_usable"] == "True" and row["body_crc_accepted"] == "False" for row in selected),
                "false_acceptances": sum(row["false_acceptance"] == "True" for row in selected)})
    write_csv(output / "failure_coverage.csv", failure_summary)
    statistics = verify_statistics(root, measured, references, closure)
    write_json(output / "cpu_audit.json", {"status": "PASS", **history_result, **model_history, **statistics, "measured_rows": 9000, "reference_rows": 9000})
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    for name in ("alexnet_checkpoint", "dino_checkpoint"):
        if sha256(paths[name]) != paths[name + "_sha256"]:
            raise RuntimeError("frozen metric model changed")
    perceptual, dino, _weights = load_quality_models(paths, device)
    original = Population("development", load(ROOT / "configs/communication_decision_study.json"))
    maximum = {"psnr_db": 0., "lpips": 0., "dino": 0., "CPU_mse": 0.}
    limits = {"psnr_db": 1e-4, "lpips": 1e-5, "dino": 1e-5, "CPU_mse": 1e-7}
    for index in range(100):
        pixels, scales, target = original.source(index, None, None)
        np.testing.assert_array_equal(pixels, np.load(quality / "images" / f"source_{index:04d}.npy", allow_pickle=False))
        signal = encode_digital(np.concatenate(scales[:7]), int(target["class_index"]))
        for snr in base["snrs_db"]:
            for seed in base["development_seeds"]:
                received = signal + seeded_noise(target["image_id"], seed, (3060, 2))[:1950] / np.sqrt(10 ** (snr / 10))
                decoded = receive_digital(received, snr)
                correct = bool(decoded["header_usable"] and decoded["label"] == int(target["class_index"]) and np.array_equal(decoded["tokens"], np.concatenate(scales[:7])))
                events = {"header_usable": decoded["header_usable"], "body_crc_accepted": decoded["crc_accepted"],
                          "source_correct": correct, "accepted_correct": correct and decoded["crc_accepted"],
                          "false_acceptance": decoded["crc_accepted"] and not correct}
                for arm in frozen["models"]:
                    row = lookup[arm, index, snr, seed]
                    if any((row[field] == "True") != bool(value) for field, value in events.items()):
                        raise RuntimeError("actual PHY replay does not match stored reception events")
                    if row["received_digital_sha256"] != hashlib.sha256(received.tobytes()).hexdigest():
                        raise RuntimeError("the registered paired digital waveform changed")
                    if int(row["complex_uses"]) != 3060 or abs(float(row["total_energy"]) - 6120) > .01:
                        raise RuntimeError("a measured frame violates the resource ledger")
        selected = [row for row in measured if int(row["image_index"]) == index]
        reference = pixels.astype(np.float32) / 255
        for offset in range(0, len(selected), 30):
            batch = selected[offset:offset + 30]
            images = []
            for row in batch:
                path = quality / row["image_path"]
                if sha256(path) != row["image_sha256"]:
                    raise RuntimeError("a stored reconstruction changed")
                image = np.load(path, allow_pickle=False)
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise RuntimeError("invalid reconstructed image")
                if row["header_usable"] == "False" and not np.all(image == .5):
                    raise RuntimeError("header failure did not retain the fixed output rule")
                images.append(image)
            scores, _original_features, _features = quality_metrics(reference, images, perceptual, dino, device)
            for row, score, image in zip(batch, scores, images):
                for stored, measured_metric in (("psnr_db", "psnr_db"), ("lpips", "lpips_alex"), ("dino", "dino_cosine")):
                    maximum[stored] = max(maximum[stored], abs(float(row[stored]) - score[measured_metric]))
                mse = float(np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2))
                maximum["CPU_mse"] = max(maximum["CPU_mse"], abs(mse - float(row["mse"])))
        if any(maximum[name] > limits[name] for name in maximum):
            raise RuntimeError(f"independent metric recomputation differs: {maximum}")
        if (index + 1) % 10 == 0:
            print(f"audited {index + 1}/100 sources, {(index + 1) * 90}/9000 images", flush=True)
    receipt = {"status": "PASS", **history_result, **model_history, **statistics, "PHY_replays": 1500,
               "recomputed_images": 9000, "source_images": 100, "metric_max_errors": maximum,
               "tolerances": limits, "frozen_inputs_unchanged": True, "new_training_or_holdout": False,
               "script_sha256": sha256(Path(__file__)), "finished_at": datetime.now().astimezone().isoformat()}
    write_json(output / "completion.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run(parser.parse_args())
