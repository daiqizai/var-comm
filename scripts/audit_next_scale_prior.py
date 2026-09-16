#!/usr/bin/env python3
"""Independently audit cached priors with NumPy/PIL, without importing the runner."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
VOCABULARY = 4096
PRIORS = ("uniform", "static_scale_frequency", "same_class_other_prefix", "true_source_prefix")
METRICS = (
    "nll_bits_per_token", "ideal_cross_entropy_bits_per_scale",
    "predictive_entropy_bits_per_token", "top1_match_rate",
    "fraction_gt_probability_below_uniform", "bit_marginal_nll_bits_per_token",
    "token_nll_p95_bits", "token_nll_max_bits",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rgb_digest(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        factor = 256 / min(width, height)
        resized = image.resize((round(width * factor), round(height * factor)), Image.Resampling.BICUBIC)
        left = round((resized.width - 256) / 2)
        top = round((resized.height - 256) / 2)
        pixels = np.asarray(resized.crop((left, top, left + 256, top + 256)))
    return hashlib.sha256(pixels.transpose(2, 0, 1).copy().tobytes()).hexdigest()


def verify_provenance(run, completion):
    expected_files = set(completion["output_hashes"])
    actual_files = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()}
    require(actual_files == expected_files | {"completion.json"}, "run file inventory changed")
    for relative, expected in completion["output_hashes"].items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run), "output hash path escapes run directory")
        require(digest_file(path) == expected, f"output checksum mismatch: {relative}")
    for original, record in completion["source_files"].items():
        require(digest_file(original) == record["sha256"], f"live source changed: {original}")
        require(digest_file(run / record["snapshot"]) == record["sha256"], "source snapshot mismatch")
    require(completion["frozen_before"] == completion["frozen_after"], "model freeze record mismatch")
    require(set(completion["frozen_before"]) == {"vae", "var"}, "incomplete model freeze record")
    config_path = run / "snapshots/project/configs/next_scale_prior_diagnostic.yaml"
    require(digest_file(config_path) == completion["config_sha256"], "config checksum mismatch")
    config = yaml.safe_load(config_path.read_text())
    for name in ("development_manifest", "split_manifest", "vae_checkpoint", "var_checkpoint"):
        require(digest_file(config["paths"][name]) == config["paths"][f"{name}_sha256"], f"asset changed: {name}")
    require(tuple(config["model"]["patch_nums"]) == SCHEDULE, "scale schedule changed")
    require(config["model"]["target_scales"] == [8, 9], "target scales changed")
    require(tuple(config["controls"]) == PRIORS, "prior controls changed")
    checks = read_json(run / "selfcheck.json")
    require(len(checks["numerical"]) == 4 and set(checks["numerical"].values()) == {"PASS"}, "numerical check failed")
    tolerance = config["evaluation"]["official_forward_absolute_tolerance"]
    expected_causal = {(row["image_id"], row["scale"]) for row in checks["causality"]}
    require(len(expected_causal) == 4 == len(checks["causality"]), "incomplete causality checks")
    for check in checks["causality"]:
        require(check["changed_target_and_future_max_abs"] == 0, "target/future influenced probabilities")
        require(check["receiver_vs_official_max_abs"] <= tolerance, "official forward disagreement")
    return config, checks


def verify_population(run, config, completion):
    manifest = read_json(run / "population_manifest.json")
    targets = read_json(Path(config["paths"]["development_manifest"]))
    split = read_json(Path(config["paths"]["split_manifest"]))["entries"]
    split_lookup = {record["image_id"]: record for record in split}
    fit_candidates = [record for record in split if record["split"] == "calibration"]
    if completion["mode"] == "smoke":
        targets, fit_candidates = targets[:2], fit_candidates[:16]
    else:
        require(completion["mode"] == "full" and len(targets) == 100, "unexpected full population")
    donors = [split_lookup[target["same_class_donor_id"]] for target in targets]
    for target, donor in zip(targets, donors):
        require(target["class_index"] == donor["class_index"], "donor class mismatch")
        require(target["image_id"] != donor["image_id"], "donor is the target")
    protected = {field: set() for field in ("image_id", "path", "file_sha256", "preprocessed_rgb_sha256")}
    expected, excluded = [], []
    for role, population in (("target_development", targets), ("same_class_donor_development", donors), ("frequency_fit", fit_candidates)):
        for source in population:
            path = Path(source["path"]).resolve()
            record = {**source, "path": str(path), "role": role, "original_split": source["split"],
                      "file_sha256": digest_file(path), "preprocessed_rgb_sha256": rgb_digest(path)}
            overlaps = any(record[field] in values for field, values in protected.items())
            if overlaps:
                require(role == "frequency_fit", "duplicate target/donor identity or image content")
                excluded.append({"image_id": source["image_id"], "path": str(path), "reason": "overlap_with_target_or_donor"})
                continue
            expected.append(record)
            if role != "frequency_fit":
                for field, values in protected.items():
                    values.add(record[field])
    require(manifest == expected, "population manifest or file/RGB identity mismatch")
    require(read_json(run / "excluded_frequency_fit.json") == excluded, "incorrect overlap exclusions")
    require(len(excluded) == completion["exclusions"], "excluded population count mismatch")
    require(dict(Counter(record["role"] for record in manifest)) == completion["population_roles"], "role counts mismatch")
    fit_count = sum(record["role"] == "frequency_fit" for record in manifest)
    minimum = config["population"]["minimum_frequency_fit_after_exclusions"] if completion["mode"] == "full" else 1
    require(fit_count >= minimum, "insufficient disjoint fitting images")
    with np.load(run / "source_tokens.npz", allow_pickle=False) as cache:
        identifiers, tokens = cache["image_ids"].tolist(), cache["tokens"].copy()
        require(identifiers == [record["image_id"] for record in manifest], "token cache image order mismatch")
        require(cache["roles"].tolist() == [record["role"] for record in manifest], "token roles mismatch")
        require(cache["classes"].tolist() == [record["class_index"] for record in manifest], "token classes mismatch")
    require(len(set(identifiers)) == len(identifiers), "duplicate cached image IDs")
    require(tokens.shape == (len(manifest), sum(size ** 2 for size in SCHEDULE)), "invalid cached token shape")
    require(tokens.dtype == np.uint16 and int(tokens.max()) < VOCABULARY, "invalid cached token representation")
    fitting = np.asarray([record["role"] == "frequency_fit" for record in manifest])
    counts = {}
    with np.load(run / "frequency_counts.npz", allow_pickle=False) as cache:
        for scale in (8, 9):
            start = sum(size ** 2 for size in SCHEDULE[:scale - 1])
            length = SCHEDULE[scale - 1] ** 2
            counts[scale] = np.bincount(tokens[fitting, start:start + length].ravel(), minlength=VOCABULARY)
            require(np.array_equal(counts[scale], cache[f"scale_{scale}"]), "frequency table used incorrect source tokens")
            require(int(counts[scale].sum()) == fit_count * length, "incorrect static token count")
    return targets, donors, manifest, dict(zip(identifiers, tokens)), counts


def recompute_metrics(log_probs, targets):
    logarithms = np.asarray(log_probs, dtype=np.float64)
    require(logarithms.shape == (len(targets), VOCABULARY), "invalid probability table shape")
    require(np.isfinite(logarithms).all(), "nonfinite source log probability")
    mass = np.exp(logarithms)
    normalization_error = float(np.max(np.abs(np.log(mass.sum(axis=1)))))
    require(normalization_error <= 2e-5, "unnormalized probability table")
    true_log = logarithms[np.arange(len(targets)), targets]
    token_losses = -true_log / np.log(2)
    bit_loss = np.zeros(len(targets))
    indices = np.arange(VOCABULARY)
    for shift in reversed(range(12)):
        source_bits = ((targets >> shift) & 1).astype(bool)
        codebook_bits = ((indices >> shift) & 1).astype(bool)
        matching_mass = np.where(source_bits, mass[:, codebook_bits].sum(axis=1), mass[:, ~codebook_bits].sum(axis=1))
        require(np.all(matching_mass > 0), "zero target bit-marginal probability")
        bit_loss -= np.log2(matching_mass)
    return {
        "nll_bits_per_token": float(token_losses.mean()),
        "ideal_cross_entropy_bits_per_scale": float(token_losses.sum()),
        "predictive_entropy_bits_per_token": float(-(mass * logarithms).sum(axis=1).mean() / np.log(2)),
        "top1_match_rate": float(np.mean(np.argmax(logarithms, axis=1) == targets)),
        "fraction_gt_probability_below_uniform": float(np.mean(true_log < -np.log(VOCABULARY) - 1e-7)),
        "bit_marginal_nll_bits_per_token": float(bit_loss.mean()),
        "token_nll_p95_bits": float(np.percentile(token_losses, 95)),
        "token_nll_max_bits": float(token_losses.max()),
    }, normalization_error


def audit_metrics(run, output, config, completion, targets, donors, tokens, counts):
    recorded = read_csv(run / "per_image_prior_metrics.csv")
    original = {(row["image_id"], int(row["scale"]), row["prior"]): row for row in recorded}
    require(len(original) == len(recorded) == len(targets) * 8 == completion["rows"], "duplicate or missing metric rows")
    require(completion["target_count"] == len(targets), "wrong target count")
    require({path.name for path in (run / "probabilities").iterdir()} == {f"{index:03d}.npz" for index in range(len(targets))}, "probability cache inventory mismatch")
    losses, audited = {}, []
    max_metric_error, max_normalization_error = 0.0, 0.0
    for image_index, (target, donor) in enumerate(zip(targets, donors)):
        with np.load(run / "probabilities" / f"{image_index:03d}.npz", allow_pickle=False) as tables:
            require(set(tables.files) == {f"{prior}_scale_{scale}" for prior in PRIORS[2:] for scale in (8, 9)}, "incomplete full prior cache")
            for scale in (8, 9):
                start = sum(size ** 2 for size in SCHEDULE[:scale - 1])
                length = SCHEDULE[scale - 1] ** 2
                ground_truth = tokens[target["image_id"]][start:start + length]
                for prior in PRIORS:
                    key = target["image_id"], scale, prior
                    record = original[key]
                    require(int(record["image_index"]) == image_index and int(record["tokens"]) == length, "wrong row image index/length")
                    require(int(record["class_index"]) == target["class_index"] and record["donor_id"] == donor["image_id"], "wrong row class/donor")
                    require(np.isfinite(float(record["prior_table_seconds_per_image"])) and float(record["prior_table_seconds_per_image"]) >= 0, "invalid timing record")
                    if prior == "uniform":
                        probabilities = np.full((length, VOCABULARY), -np.log(VOCABULARY))
                    elif prior == "static_scale_frequency":
                        weights = counts[scale].astype(np.float64) + config["static_frequency"]["smoothing_pseudocount"]
                        probabilities = np.broadcast_to(np.log(weights / weights.sum()), (length, VOCABULARY))
                    else:
                        probabilities = tables[f"{prior}_scale_{scale}"]
                        require(probabilities.dtype == np.float32, "prior cache is not full FP32")
                    metrics, normalization_error = recompute_metrics(probabilities, ground_truth)
                    for metric in METRICS:
                        error = abs(metrics[metric] - float(record[metric]))
                        require(error <= 1e-8, f"metric mismatch: {key}, {metric}")
                        max_metric_error = max(max_metric_error, error)
                    max_normalization_error = max(max_normalization_error, normalization_error)
                    losses[key] = metrics
                    audited.append({"image_id": target["image_id"], "scale": scale, "prior": prior, **metrics})
        if (image_index + 1) % 20 == 0:
            print(f"audit probabilities {image_index + 1}/{len(targets)}", flush=True)
    summary, comparisons, gates = [], [], {}
    original_summary = {(int(row["scale"]), row["prior"]): row for row in read_csv(run / "summary.csv")}
    original_gains = {(int(row["scale"]), row["control"], row["metric"]): row for row in read_csv(run / "paired_gains.csv")}
    require(len(original_summary) == 8 and len(original_gains) == 12, "incomplete aggregate tables")
    image_ids = sorted(target["image_id"] for target in targets)
    resamples = config["evaluation"]["bootstrap_resamples"]
    selections = np.random.default_rng(config["evaluation"]["bootstrap_seed"]).integers(len(image_ids), size=(resamples, len(image_ids)))
    max_aggregate_error = 0.0
    for scale in (8, 9):
        checks = {}
        for prior in PRIORS:
            means = {metric: float(np.mean([losses[(identifier, scale, prior)][metric] for identifier in image_ids])) for metric in METRICS}
            require(int(original_summary[(scale, prior)]["images"]) == len(image_ids), "wrong aggregate population")
            for metric in METRICS[:6]:
                error = abs(means[metric] - float(original_summary[(scale, prior)][metric]))
                require(error <= 1e-8, "aggregate metric mismatch")
                max_aggregate_error = max(max_aggregate_error, error)
            summary.append({"scale": scale, "prior": prior, "images": len(image_ids), **means})
            if prior == "true_source_prefix":
                continue
            for metric in ("nll_bits_per_token", "bit_marginal_nll_bits_per_token"):
                differences = np.asarray([losses[(identifier, scale, prior)][metric] - losses[(identifier, scale, "true_source_prefix")][metric] for identifier in image_ids])
                samples = np.add.reduce(differences[selections], axis=1) / len(image_ids)
                lower, upper = np.percentile(samples, [2.5, 97.5])
                values = {"control_minus_true_gain": float(np.mean(differences)), "ci_low": float(lower), "ci_high": float(upper),
                          "true_prefix_win_fraction": float(np.count_nonzero(differences > 0) / len(image_ids))}
                for name, value in values.items():
                    error = abs(value - float(original_gains[(scale, prior, metric)][name]))
                    require(error <= 1e-8, "paired image bootstrap mismatch")
                    max_aggregate_error = max(max_aggregate_error, error)
                comparisons.append({"scale": scale, "control": prior, "metric": metric, **values})
                if metric == "nll_bits_per_token":
                    checks[prior] = values["control_minus_true_gain"] >= config["gate"]["minimum_mean_gain_bits_per_token_over_each_control"] and values["ci_low"] > 0
        gates[str(scale)] = {"passed": all(checks.values()), "checks": checks}
    require(gates == completion["gate"], "preregistered gate verdict mismatch")
    expected_status = "PASS_PRIOR_GATE_ONLY" if all(gate["passed"] for gate in gates.values()) else "STOP_UNCALIBRATED_PRIOR_PROTOTYPE"
    if completion["mode"] == "smoke":
        expected_status = "SMOKE_COMPLETE_NOT_SCIENTIFIC_RESULT"
    require(completion["status"] == expected_status, "completion status mismatch")
    write_csv(output / "recomputed_per_image.csv", audited)
    write_csv(output / "recomputed_summary.csv", summary)
    write_csv(output / "recomputed_paired_gains.csv", comparisons)
    retained = {}
    for scale in (8, 9):
        joint, marginal = [next(row["control_minus_true_gain"] for row in comparisons if row["scale"] == scale and row["control"] == "uniform" and row["metric"] == metric)
                           for metric in ("nll_bits_per_token", "bit_marginal_nll_bits_per_token")]
        retained[str(scale)] = {"uniform_minus_joint_nll": joint, "uniform_minus_marginal_nll": marginal,
                                "bit_marginal_fraction_of_uniform_relative_gain": marginal / joint if joint != 0 else None}
    return {"rows_recomputed": len(audited), "max_per_image_metric_abs_error": max_metric_error,
            "max_aggregate_or_ci_abs_error": max_aggregate_error, "max_log_normalization_error": max_normalization_error,
            "gate": gates, "bit_marginal_information_retention": retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/VAR-NEXT-SCALE-PRIOR-AUDIT-001")
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    boundary = ROOT / "outputs"
    require(run.is_relative_to(boundary) and output.is_relative_to(boundary), "audit paths must physically remain in VAR_COMM/outputs")
    require(not output.is_relative_to(run) and not run.is_relative_to(output), "audit must not modify the frozen run")
    completion = read_json(run / "completion.json")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "audit_source.py")
    metadata = {"local_started": datetime.now().astimezone().isoformat(), "command": sys.argv,
                "run": str(run), "completion_sha256": digest_file(run / "completion.json"),
                "audit_source_sha256": digest_file(__file__), "python": sys.version, "numpy": np.__version__}
    try:
        config, checks = verify_provenance(run, completion)
        targets, donors, manifest, tokens, counts = verify_population(run, config, completion)
        metrics = audit_metrics(run, output, config, completion, targets, donors, tokens, counts)
        result = {**metadata, "status": "AUDIT_PASS", "diagnostic_status": completion["status"],
                  "source_roles": dict(Counter(record["role"] for record in manifest)),
                  "input_file_and_rgb_hashes_recomputed": len(manifest), "output_hashes_verified": len(completion["output_hashes"]),
                  "frequency_tokens": {str(scale): int(count.sum()) for scale, count in counts.items()},
                  "model_freeze_record_verified": True, "models_reloaded_for_audit": False,
                  "causality_record_verified": checks["causality"], **metrics,
                  "limitations": ["cached probability audit, not an independent second inference run",
                                  "bootstrap conditional on fixed fit table and donor assignment",
                                  "development prior-predictiveness only; no FEC, CRC or image-quality result"],
                  "output_hashes": {path.name: digest_file(path) for path in sorted(output.iterdir()) if path.is_file()}}
        write_json(output / "audit.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_json(output / "audit_failed.json", {**metadata, "status": "AUDIT_FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
