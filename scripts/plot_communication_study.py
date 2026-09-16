#!/usr/bin/env python3
"""Plot frozen-policy results with explicit equality and strong-reference labels."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    receipt_path = arguments.analysis_dir / "completion.json"
    receipt = json.loads(receipt_path.read_text())
    summary_path = arguments.analysis_dir / "summary.csv"
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != receipt["output_hashes"]["summary.csv"]:
        raise RuntimeError("frozen policy results changed")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    snrs = [1., 4., 5., 6., 7., 13., 19.]
    lookup = {(row["method"], row["scope"]): row for row in rows}
    output = arguments.output_dir.resolve()
    allowed = Path(__file__).resolve().parents[1] / "outputs/COMMUNICATION-CONVERGENCE-20260915"
    if not output.is_relative_to(allowed):
        raise ValueError("figures must remain in the communication study")
    output.mkdir(parents=True, exist_ok=False)
    labels = {"raw_reliability": "Raw + BLER target", "raw_goodput": "Raw + goodput", "raw_quality": "Raw + image quality",
              "arithmetic_quality": "Arithmetic: quality = BLER", "arithmetic_goodput": "Arithmetic + goodput",
              "r3__full_grid_prediction_features": "Frozen R3", "perceptual_deepjscc": "Perceptual DeepJSCC (supported)",
              "wetok_8PSK_FEC": "WeTok 8PSK + FEC"}
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for method, label in labels.items():
        for axis, metric, ylabel in zip(axes, ("psnr_db", "lpips", "dino"), ("PSNR (dB), higher is better", "LPIPS, lower is better", "DINO similarity, higher is better")):
            axis.plot(snrs, [float(lookup[method, str(snr)][metric]) for snr in snrs], marker="o", label=label,
                      linestyle="--" if "goodput" in method else "-")
            axis.set(xlabel="SNR (dB)", ylabel=ylabel, xticks=snrs)
            axis.grid(alpha=.25)
    figure.suptitle(f"{receipt['population']}: {receipt['source_images']} source images, 3 noise realizations; N=3060, E=6120\n"
                   "Policies fixed on calibration only; all failure outputs retained", fontsize=12)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.subplots_adjust(top=.80, bottom=.25, wspace=.25)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"quality_all_snrs.{extension}", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for method, label in list(labels.items())[:5]:
        selected = [4., 5., 6., 7.]
        axis.plot(selected, [float(lookup[method, str(snr)]["lpips"]) for snr in selected], marker="o", label=label,
                  linestyle="--" if "goodput" in method else "-")
    axis.set(xlabel="SNR (dB)", ylabel="LPIPS, lower is better", xticks=[4, 5, 6, 7],
             title=f"{receipt['population']}: transition-region diagnostic (not a changed primary range)")
    axis.grid(alpha=.25)
    axis.legend(fontsize=9)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"transition_lpips.{extension}", dpi=180)
    plt.close(figure)
    result = {"analysis_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
              "population": receipt["population"], "new_measurements": False,
              "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "output_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}}
    (output / "completion.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
