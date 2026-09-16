#!/usr/bin/env python3
"""Plot frozen, audited development results without rerunning any experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MPLCONFIGDIR"] = str(ROOT / "outputs/cache/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, write_json


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/VAR-NEXT-SCALE-DECODING-FIGURES-001")
    args = parser.parse_args()
    output = create_output(args.output_dir)
    snapshot(output, [Path(__file__)])
    single = ROOT / "outputs/VAR-SINGLE-SCALE-CHANNEL-001"
    full = ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001"
    audits = [ROOT / "outputs/VAR-SINGLE-SCALE-CHANNEL-AUDIT-001/audit.json", ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-AUDIT-001/audit.json"]
    for audit in audits:
        if json.loads(audit.read_text())["status"] != "AUDIT_PASS":
            raise RuntimeError("refusing to plot unaudited results")
    single_rows = read_rows(single / "summary.csv")
    figure, axes = pyplot.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    styles = {"uniform": ("ML (uniform)", "#777777", "o"),
              "bit_marginal_true": ("Independent-bit prior", "#D98428", "s"),
              "true_source_prefix": ("Full-token VAR prior", "#1769AA", "D")}
    for arm, (label, color, marker) in styles.items():
        selected = sorted((row for row in single_rows if row["arm"] == arm), key=lambda row: float(row["snr_db"]))
        axes.plot([float(row["snr_db"]) for row in selected], [float(row["accepted_source_recovery"]) for row in selected],
                  label=label, color=color, marker=marker, linewidth=2)
    axes.set(xlabel="Complex-symbol SNR (dB)", ylabel="CRC-accepted correct source recovery",
             title="Scale 9: same FEC and received waveform", ylim=(-0.02, 1.02), xticks=[4, 7, 10])
    axes.grid(alpha=0.25)
    axes.legend(loc="lower right")
    figure.savefig(output / "single_scale_recovery.png", dpi=180)
    pyplot.close(figure)
    full_rows = read_rows(full / "summary.csv")
    figure, axes = pyplot.subplots(1, 3, figsize=(13.6, 4.1), constrained_layout=True)
    styles = {"group_ml": ("Grouped ML", "#777777", "o"), "group_var": ("Grouped VAR-MAP", "#1769AA", "D"),
              "group_entropy": ("Entropy coding + FEC", "#228B57", "s"), "whole_adaptive": ("Whole-frame adaptive", "#B3424A", "^")}
    metrics = [("psnr_db", "PSNR (dB), higher is better"), ("lpips_alex", "LPIPS, lower is better"), ("dino_cosine", "DINO cosine, higher is better")]
    for axis, (metric, ylabel) in zip(axes, metrics):
        for arm, (label, color, marker) in styles.items():
            selected = sorted((row for row in full_rows if row["arm"] == arm), key=lambda row: float(row["snr_db"]))
            axis.plot([float(row["snr_db"]) for row in selected], [float(row[metric]) for row in selected],
                      label=label, color=color, marker=marker, linewidth=1.8)
        axis.set(xlabel="Complex-symbol SNR (dB)", ylabel=ylabel, xticks=[1, 4, 7, 13, 19])
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    figure.suptitle("3060 complex uses, including headers / CRC / tails; 100 development images")
    figure.savefig(output / "fixed_budget_image_quality.png", dpi=180)
    pyplot.close(figure)
    write_json(output / "completion.json", {"status": "PLOTS_FROM_AUDITED_FROZEN_RESULTS",
        "single_completion_sha256": sha256(single / "completion.json"), "full_completion_sha256": sha256(full / "completion.json"),
        "audit_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in audits},
        "note": "Lines connect measured points; no additional SNRs were evaluated.", "output_hashes": artifact_hashes(output)})
    print(output)


if __name__ == "__main__":
    main()
