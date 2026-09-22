#!/usr/bin/env python3
"""Reproduce completed external-baseline statistics using published CSVs only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/external_baselines"
sys.path[:0] = [str(ROOT / "experiments/external-baseline-positioning-20260916/src"), str(ROOT / "src")]

from external_positioning.analysis import summarize, validate_groups
from var_comm.mode_policies import fit_actions, summarize_candidates


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_public_rows(directory=RESULTS):
    sources = {int(item["image_index"]): item for item in json.loads((directory / "development_manifest.json").read_text())}
    rows = read_csv(directory / "reference_per_frame.csv")
    for row in rows:
        row["protocol"] = "common_paid_information"
    authors = []
    for path in sorted((directory / "author_per_frame").glob("*.csv")):
        authors.extend(read_csv(path))
    for row in authors:
        row["total_energy"] = row["actual_total_energy"]
    digital = read_csv(directory / "digital_development.csv")
    if len(rows) != 6000 or len(authors) != 13500 or len(digital) != 6000 or len(sources) != 100:
        raise RuntimeError("public development partition coverage changed")
    combined = rows + authors + digital
    validate_groups(combined, sources)
    return combined, sources


def check_calibration(directory=RESULTS):
    policies = json.loads((directory / "calibration/digital_policies.json").read_text())
    if policies["DINO_used_for_selection"] or policies["development_accessed_for_selection"]:
        raise RuntimeError("mode fitting used forbidden information")
    config = json.loads((ROOT / "configs/communication_decision_study.json").read_text())
    config["snrs_db"] = [1., 4., 7., 13., 19.]
    source_ids, count = set(), 0
    for budget in (4204, 4498):
        rows = read_csv(directory / "calibration/digital_per_frame" / f"{budget}.csv")
        keys = {(int(row["image_index"]), row["family"], int(row["mode"]), float(row["snr_db"]), int(row["seed"])) for row in rows}
        expected = {(index, family, mode, snr, seed) for index in policies["calibration_sources"]
                    for family in ("raw", "arithmetic") for mode in (7, 8, 9)
                    for snr in (1., 4., 7., 13., 19.) for seed in (4101, 4102, 4103)}
        if len(rows) != 9000 or keys != expected:
            raise RuntimeError("calibration is incomplete or duplicated")
        if any(int(row["complex_uses"]) != budget or row["population"] != "calibration" for row in rows):
            raise RuntimeError("calibration budget/population mix")
        # Historical CSVs predate decoder identity columns. Bind this replay
        # to its exact archive bytes; do not invent a verified model SHA.
        if {row["protocol"] for row in rows} != {"common_paid_information"}:
            raise RuntimeError("calibration protocol mix")
        archive_sha = hashlib.sha256((directory / "calibration/digital_per_frame" / f"{budget}.csv").read_bytes()).hexdigest()
        context = {"budget": budget, "renderer": "historical_archive_only",
                   "decoder_sha": "UNRECORDED:archive:" + archive_sha,
                   "protocol_id": "common_paid_information"}
        actions, family, _primary = fit_actions(summarize_candidates(rows, context=context), config, context=context)
        previous = policies["budgets"][str(budget)]
        if actions != previous["actions"] or family != previous["calibration_global_family"]:
            raise RuntimeError("calibration does not reproduce frozen actions")
        source_ids.update(row["image_id"] for row in rows)
        count += len(rows)
    for row in read_csv(directory / "digital_development.csv"):
        selected = policies["budgets"][row["complex_uses"]]["actions"][row["family"]]["quality"][str(float(row["snr_db"]))]
        if int(row["mode"]) != selected or row["image_id"] in source_ids:
            raise RuntimeError("development mode changed or calibration leaked")
    return count


def verify_table(actual, path, keys, numeric):
    saved = {tuple(row[key] for key in keys): row for row in read_csv(path)}
    measured = {tuple(str(row[key]) for key in keys): row for row in actual}
    if len(actual) != len(saved) or set(measured) != set(saved):
        raise RuntimeError(f"table population differs: {path.name}")
    maximum = max(abs(float(row[name]) - float(saved[key][name])) for key, row in measured.items() for name in numeric)
    if maximum > 1e-10:
        raise RuntimeError(f"statistics differ: {path.name}; {maximum}")
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced/external_baselines")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output == RESULTS or output.is_relative_to(RESULTS) or output.exists():
        raise ValueError("use a fresh output; never overwrite published evidence")
    rows, sources = load_public_rows()
    calibration_rows = check_calibration()
    summary, paired, per_source = summarize(rows, sources)
    metrics = ("psnr_db", "ssim", "lpips", "dino")
    numbers = [name + suffix for name in metrics for suffix in ("", "_ci_low", "_ci_high")]
    errors = {
        "summary": verify_table(summary, RESULTS / "analysis/summary.csv", ("method", "protocol", "complex_uses", "scope"), numbers),
        "paired": verify_table(paired, RESULTS / "analysis/paired.csv", ("method", "method_complex_uses", "control", "control_complex_uses", "scope", "metric"),
                               ("mean_method_minus_control", "ci_low", "ci_high")),
        "per_source": verify_table(per_source, RESULTS / "analysis/per_source.csv", ("method", "protocol", "complex_uses", "scope", "image_index"), metrics),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("summary", summary), ("paired", paired), ("per_source", per_source)):
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    receipt = {"status": "PUBLISHED_EXTERNAL_STATISTICS_REPRODUCED", "development_rows": len(rows), "calibration_rows": calibration_rows,
               "source_images": len(sources), "summary_rows": len(summary), "paired_metric_rows": len(paired), "maximum_errors": errors,
               "GPU_used": False, "source_pixels_or_weights_read": False, "image_metrics_recomputed": False, "HiFi_ranked": False}
    (output / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
