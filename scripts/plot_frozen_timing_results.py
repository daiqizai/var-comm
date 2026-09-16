#!/usr/bin/env python3
"""Render timing figures with source/repetition counts taken from the completed receipt."""

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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    receipt_path = arguments.run_dir / "completion.json"
    receipt = json.loads(receipt_path.read_text())
    summary_path = arguments.run_dir / "summary.csv"
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != receipt["output_hashes"]["summary.csv"]:
        raise RuntimeError("timing summary changed")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = {"whole_m7": "Digital VAR m7", "whole_m8": "Digital VAR m8", "whole_m9": "Digital VAR m9",
              "whole_adaptive": "Digital VAR adaptive", "r2__full_grid_innovation": "Frozen R2 residual",
              "perceptual_deepjscc": "Perceptual DeepJSCC", "wetok_8PSK_FEC": "WeTok 8PSK + FEC"}
    snrs = (1, 4, 7, 13, 19)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for arm, label in labels.items():
        selected = [next(row for row in rows if row["scope"] == str(snr) and row["arm"] == arm) for snr in snrs]
        for axis, metric in zip(axes, ("TX_mean_ms", "RX_mean_ms")):
            axis.plot(snrs, [float(row[metric]) for row in selected], marker="o", label=label)
    for axis, title in zip(axes, ("CPU RGB to CPU waveform", "CPU waveform to CPU RGB")):
        axis.set(xlabel="SNR (dB)", ylabel="Mean processing time (ms)", title=title, xticks=snrs)
        axis.grid(alpha=.25)
    scope = "Engineering smoke only; not a system ranking" if receipt["mode"] == "smoke" else "Frozen systems: N=3060, E=6120"
    figure.suptitle(f"{scope}\n{receipt['source_images']} development sources; {receipt['repeats']} timing repeat(s); batch 1\n"
                   "Common CPU endpoints; excludes airtime, queueing, IO and model loading", fontsize=11)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.subplots_adjust(top=.76, bottom=.23, wspace=.24)
    output = arguments.output_dir.resolve()
    allowed = Path(__file__).resolve().parents[1] / "outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915"
    if not output.is_relative_to(allowed):
        raise ValueError("figures must remain in the declared timing output")
    output.mkdir(parents=True, exist_ok=False)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"same_endpoint_latency.{extension}", dpi=180)
    plt.close(figure)
    proof = {"timing_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
             "source_images": receipt["source_images"], "repeats": receipt["repeats"], "mode": receipt["mode"],
             "new_measurements": False, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "output_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}}
    (output / "completion.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
