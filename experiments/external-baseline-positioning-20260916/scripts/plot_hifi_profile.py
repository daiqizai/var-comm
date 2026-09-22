#!/usr/bin/env python3
"""Visualize actual calibration profiling without claiming population-wide acceleration."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(arguments):
    stage = arguments.stage
    output = arguments.output if arguments.output is not None else stage / "runtime_figures_001"
    output.mkdir(exist_ok=False)
    profile = stage / "attention_profile_001"
    measured = rows(profile / "uninstrumented_speed_comparison.csv")
    positions = np.arange(len(measured))
    names = [f"cal{row['image_index']} / {float(row['snr_db']):g}dB" for row in measured]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for shift, variant, colour in ((-.18, "original", "#1f77b4"), (.18, "direct", "#ff7f0e")):
        elapsed = [float(row[f"{variant}_RX_seconds"]) for row in measured]
        memory = [int(row[f"{variant}_peak_allocated_bytes"]) / 2**30 for row in measured]
        axes[0].bar(positions + shift, elapsed, .36, label=variant, color=colour)
        axes[1].bar(positions + shift, memory, .36, label=variant, color=colour)
    for axis in axes:
        axis.set_xticks(positions, names)
        axis.legend()
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Complete RX seconds, uninstrumented")
    axes[1].set_ylabel("Peak allocated GiB (not nvidia-smi reserved usage)")
    figure.suptitle("Three fixed calibration workpoints; one paired uninstrumented full pass each")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"actual_speed_memory.{extension}", dpi=170)
    plt.close(figure)
    breakdown = json.loads((profile / "case_00/original_instrumentation.json").read_text())["GPU_seconds_inclusive"]
    values = [breakdown["setup_receive"], breakdown["unet_forward"], breakdown["consistency_loss_forward"],
              breakdown["input_gradient_inclusive"] - breakdown["attention_backward_recompute"], breakdown["attention_backward_recompute"]]
    labels = ["RX setup", "U-Net forward (including first attention forward)", "Consistency loss forward",
              "Other input backward (including checkpoint inner grads)", "Attention forward recomputation inside backward"]
    figure, axis = plt.subplots(figsize=(12, 5))
    offset = 0.
    for index, (value, label) in enumerate(zip(values, labels)):
        axis.barh([0], [value], left=offset, color=plt.cm.tab10(index), label=f"{label}: {value:.3f}s")
        offset += value
    axis.set(yticks=[], xlabel="GPU event seconds (nested attention term split out, not added twice)",
             title="Original cal0 / 1dB, 251 reverse steps; not 3 diffusion runs")
    handles, names = axis.get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=1, fontsize=9)
    figure.tight_layout(rect=(0, .40, 1, .98))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"inclusive_cost_breakdown.{extension}", dpi=170)
    plt.close(figure)
    gradients = rows(profile / "matched_input_gradients.csv")
    outputs = [row for row in rows(profile / "full_output_comparisons.csv") if row["comparison"] == "attention_direct_vs_original_full_output"]
    repeated = rows(stage / "repeatability_001/original_repeat_comparisons.csv")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for position, row in enumerate(gradients):
        axes[0].scatter(position, float(row["relative_L2_error"]), color="green" if row["input_gradient_passed"] == "True" else "red")
    axes[0].axhline(1e-4, linestyle="--", color="black", label="relative L2 threshold")
    axes[0].set(yscale="log", xlabel="Nine matched states; colour includes elementwise allclose check", ylabel="Input-gradient relative L2 error")
    axes[0].legend(fontsize=8)
    errors = [float(row["max_abs_RGB_error"]) for row in outputs] + [float(repeated[0]["max_abs_RGB_error"])]
    axes[1].bar(range(4), errors, color=["#ff7f0e"] * 3 + ["#1f77b4"])
    axes[1].axhline(2e-4, linestyle="--", color="black", label="unchanged strict RGB threshold")
    axes[1].set(yscale="log", xticks=range(4), xticklabels=["A/B cal0", "A/B cal500", "A/B cal999", "A/A cal0"], ylabel="Complete output max abs RGB difference")
    axes[1].legend(fontsize=8)
    figure.suptitle("Validation limits: do not claim bit-exact outputs or relax thresholds after seeing results")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"gradient_output_checks.{extension}", dpi=170)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
