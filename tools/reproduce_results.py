#!/usr/bin/env python3
"""Recompute published statistics from CSV only; never load pixels or neural models."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/hybrid_weight_closure"
METRICS = ("psnr_db", "lpips", "dino", "mse")
REGIONS = {"primary": [1., 4., 7.], "mechanism": [7., 13., 19.], "high": [13., 19.]}
REGIONS.update({f"snr_{snr:g}": [snr] for snr in (1., 4., 7., 13., 19.)})


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def frame_key(row):
    return row["image_id"], float(row["snr_db"]), int(row["seed"])


def expected_value(digital, deep, probability):
    if not 0 <= probability <= 1:
        raise ValueError("selection probability must be in [0,1]")
    return (1 - probability) * digital + probability * deep


def assemble_rows(directory=RESULTS):
    measured = read_rows(directory / "development.csv")
    references = read_rows(directory / "references.csv")
    frozen = json.loads((directory / "frozen_comparisons.json").read_text())
    indexed = {(row["arm"], *frame_key(row)): row for row in measured + references}
    if len(measured) != 9000 or len(references) != 9000 or len(indexed) != 18000:
        raise RuntimeError("incomplete or duplicated measured/reference population")
    identities = sorted({row["image_id"] for row in measured})
    if len(identities) != 100:
        raise RuntimeError("source-image count changed")
    expected = []
    for mixture in frozen["time_sharing"]:
        if not mixture["coverage"]:
            continue
        for identifier in identities:
            for snr in (1., 4., 7., 13., 19.):
                for seed in (2001, 2002, 2003):
                    digital = indexed[mixture["digital"], identifier, snr, seed]
                    deep = indexed["perceptual_deepjscc", identifier, snr, seed]
                    expected.append({"arm": mixture["reference_id"], "image_id": identifier, "snr_db": snr, "seed": seed,
                                     **{metric: expected_value(float(digital[metric]), float(deep[metric]), mixture["p_deep"]) for metric in METRICS}})
    previous = {(row["arm"], *frame_key(row)): row for row in read_rows(directory / "expected_time_sharing.csv")}
    if len(expected) != len(previous):
        raise RuntimeError("fixed probability reference population changed")
    maximum = 0.
    for row in expected:
        saved = previous[(row["arm"], *frame_key(row))]
        maximum = max(maximum, *(abs(row[metric] - float(saved[metric])) for metric in METRICS))
    if maximum > 1e-12:
        raise RuntimeError("expectation no longer matches calibration-frozen proportions")
    return measured + references + expected, identities, maximum


def aggregate(records, identities):
    indexed = {(row["arm"], *frame_key(row)): row for row in records}
    methods = sorted({row["arm"] for row in records})
    summary, reduced = [], {}
    for region, snrs in REGIONS.items():
        for arm in methods:
            values = {metric: [] for metric in METRICS}
            for identifier in identities:
                frames = [indexed[arm, identifier, snr, seed] for snr in snrs for seed in (2001, 2002, 2003)]
                for metric in METRICS:
                    values[metric].append(float(np.mean([float(row[metric]) for row in frames])))
            reduced[region, arm] = {metric: np.asarray(entries) for metric, entries in values.items()}
            summary.append({"region": region, "arm": arm, "source_images": len(identities),
                            **{metric: float(np.mean(entries)) for metric, entries in values.items()}})
    return summary, reduced


def write_rows(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(output):
    output = output.resolve()
    if output == RESULTS.resolve() or output.is_relative_to(RESULTS.resolve()):
        raise RuntimeError("cannot overwrite published scientific results")
    output.mkdir(parents=True, exist_ok=True)
    records, identities, expectation_error = assemble_rows()
    summary, reduced = aggregate(records, identities)
    previous = {(row["region"], row["arm"]): row for row in read_rows(RESULTS / "quality_summary.csv")}
    mean_error = max(abs(row[metric] - float(previous[row["region"], row["arm"]][metric])) for row in summary for metric in METRICS)
    samples = np.random.default_rng(20260916).integers(100, size=(10000, 100))
    intervals, interval_error = [], 0.
    for row in read_rows(RESULTS / "paired_intervals.csv"):
        difference = reduced[row["region"], row["arm"]][row["metric"]] - reduced[row["region"], row["control"]][row["metric"]]
        lower, upper = np.percentile(difference[samples].mean(1), [2.5, 97.5])
        measured = {"difference": float(difference.mean()), "ci_low": float(lower), "ci_high": float(upper)}
        interval_error = max(interval_error, *(abs(value - float(row[name])) for name, value in measured.items()))
        intervals.append({"region": row["region"], "arm": row["arm"], "control": row["control"], "metric": row["metric"], **measured})
    if mean_error > 1e-10 or interval_error > 1e-10:
        raise RuntimeError(f"published statistics do not reproduce: {mean_error}, {interval_error}")
    write_rows(output / "quality_summary.csv", summary)
    write_rows(output / "paired_intervals.csv", intervals)
    receipt = {"status": "PASS", "source_images": len(identities), "measured_plus_reference_plus_expected_rows": len(records),
               "paired_intervals": len(intervals), "maximum_mean_error": mean_error, "maximum_interval_error": interval_error,
               "maximum_expectation_error": expectation_error, "pixels_or_neural_models_loaded": False,
               "new_training_or_holdout": False, "finite_random_selector_deployment": False}
    (output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    run(parser.parse_args().output)
