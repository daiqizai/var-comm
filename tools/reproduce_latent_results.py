"""Recompute published latent-enhancement statistics without models or source pixels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/latent_enhancement"
METRICS = ("psnr_db", "lpips_alex", "dino_cosine")
SNRS = (1, 4, 7, 13, 19)
SELECTED = ("m8_D0", "m8_Dc_only", "m8_receiver_only_refiner", "m8_plus_latent_512", "m8_plus_latent_1024",
            "raw_N3572_m8_Dc", "arithmetic_N3572_m8_Dc", "raw_N4084_m8_Dc", "arithmetic_N4084_m8_Dc")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_cube(rows=None):
    rows = read_rows(RESULTS / "per_source_snr.csv") if rows is None else rows
    methods = sorted({row["method"] for row in rows})
    sources = sorted({int(row["source_index"]) for row in rows})
    if len(methods) != 41 or sources != list(range(100)) or len(rows) != 20500:
        raise ValueError("expected 41 methods, 100 sources and five SNRs")
    arrays = {method: np.full((100, len(SNRS), len(METRICS)), np.nan) for method in methods}
    seen = set()
    for row in rows:
        method, source, snr = row["method"], int(row["source_index"]), int(float(row["snr_db"]))
        key = method, source, snr
        if key in seen or snr not in SNRS:
            raise ValueError("duplicate source/SNR record or unexpected SNR")
        seen.add(key)
        arrays[method][source, SNRS.index(snr)] = [float(row[metric]) for metric in METRICS]
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("missing or nonfinite quality records")
    return arrays


def aggregate(arrays):
    overall, per_snr = [], []
    for method, values in sorted(arrays.items()):
        mean = values.mean(axis=1).mean(axis=0)
        overall.append({"method": method, "sources": len(values), **dict(zip(METRICS, map(float, mean)))})
        for position, snr in enumerate(SNRS):
            mean = values[:, position].mean(axis=0)
            per_snr.append({"method": method, "snr_db": snr, "sources": len(values),
                            **dict(zip(METRICS, map(float, mean)))})
    return overall, per_snr


def paired(arrays):
    reference = arrays["m8_D0"].mean(axis=1)
    bootstrap = {metric: np.random.default_rng(2026091801 + position).integers(100, size=(10000, 100))
                 for position, metric in enumerate(METRICS)}
    fields = ("delta_psnr_db", "delta_lpips", "delta_dino")
    results = []
    for method in sorted(arrays):
        if method == "m8_D0":
            continue
        differences = arrays[method].mean(axis=1) - reference
        result = {"method": method, "reference": "m8_D0"}
        for position, metric in enumerate(METRICS):
            values = differences[:, position]
            means = values[bootstrap[metric]].mean(axis=1)
            result[fields[position]] = {"gain": float(values.mean()), "ci_low": float(np.percentile(means, 2.5)),
                                       "ci_high": float(np.percentile(means, 97.5))}
        results.append(result)
    return results


def verify_results(overall, comparisons):
    expected = {row["method"]: row for row in read_rows(RESULTS / "method_summary.csv")}
    maximum = 0.0
    for row in overall:
        for metric in METRICS:
            maximum = max(maximum, abs(row[metric] - float(expected[row["method"]][metric])))
    old_comparisons = {row["method"]: row for row in json.loads((RESULTS / "paired_comparisons.json").read_text())}
    for row in comparisons:
        for metric in ("delta_psnr_db", "delta_lpips", "delta_dino"):
            for statistic in ("gain", "ci_low", "ci_high"):
                maximum = max(maximum, abs(row[metric][statistic] - old_comparisons[row["method"]][metric][statistic]))
    if maximum > 1e-10:
        raise RuntimeError(f"published numeric mismatch: {maximum}")
    failures = read_rows(RESULTS / "failure_summary.csv")
    if len(failures) != 205 or any(int(row["frames"]) != 300 for row in failures):
        raise RuntimeError("failure coverage is incomplete")
    return maximum


def plot_all_source_means(per_snr, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["method"], row["snr_db"]): row for row in per_snr}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for method in SELECTED:
        values = [lookup[method, snr] for snr in SNRS]
        axes[0].plot(SNRS, [row["psnr_db"] for row in values], marker="o", label=method)
        axes[1].plot(SNRS, [row["lpips_alex"] for row in values], marker="o", label=method)
    axes[0].set(xlabel="SNR (dB)", ylabel="PSNR (dB)", title="100-source development mean (3 noises/source)")
    axes[1].set(xlabel="SNR (dB)", ylabel="LPIPS", title="100-source development mean (3 noises/source)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced/latent_enhancement")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT / "reproduced").resolve()):
        raise ValueError("statistical reproduction must not overwrite published evidence")
    destination.mkdir(parents=True, exist_ok=True)
    arrays = load_cube()
    overall, per_snr = aggregate(arrays)
    comparisons = paired(arrays)
    maximum = verify_results(overall, comparisons)
    write_rows(destination / "method_summary.csv", overall)
    write_rows(destination / "per_snr_summary.csv", per_snr)
    (destination / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    if args.figures:
        plot_all_source_means(per_snr, destination / "quality_per_snr_all_sources.png")
    result = {"status": "PASS", "source_SNR_rows": 20500, "methods": 41, "sources": 100,
              "paired_metric_intervals": len(comparisons) * len(METRICS), "maximum_absolute_difference": maximum,
              "GPU_used": False, "pixel_metrics_recomputed": False,
              "mode_adaptation_or_online_timing_validated": False}
    (destination / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
