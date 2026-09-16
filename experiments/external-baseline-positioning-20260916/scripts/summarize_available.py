#!/usr/bin/env python3
"""Report completed work without waiting for, or ranking, partial diffusion runs."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from external_positioning.analysis import METRICS, SCOPES, SNRS, summarize
from var_comm.study import sha256, write_csv, write_json


LABELS = {"raw_adaptive": "VAR raw adaptive", "arithmetic_adaptive": "VAR arithmetic adaptive",
          "perceptual_deepjscc": "Perceptual DeepJSCC", "wetok_r3": "WeTok R3",
          **{f"swin_ra{rate}": f"SwinJSCC RA{rate}" for rate in (32, 64, 96)},
          **{f"adjscc_c{rate}": f"ADJSCC C{rate}" for rate in (2, 4, 6)}}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def label(method):
    for family in ("raw", "arithmetic"):
        if method.startswith(f"{family}_adaptive_N"):
            return f"VAR {family} adaptive N{method.split('_N')[-1]}"
    return LABELS.get(method, method.replace("_", " "))


def color(method):
    for family, value in (("swin", "#1f77b4"), ("adjscc", "#ff7f0e"), ("raw", "#2ca02c"),
                          ("arithmetic", "#17becf"), ("perceptual", "#9467bd"), ("wetok", "#d62728")):
        if method.startswith(family):
            return value
    return "black"


def load_rows(root, digital_runs):
    sources = {int(item["image_index"]): item for item in json.loads((root / "inputs_001/development_inputs.json").read_text())}
    paths = [root / "inputs_001/reference_per_frame.csv", root / "fast_author_metrics_001/per_frame.csv"]
    paths.extend(run / "per_frame.csv" for run in digital_runs)
    rows = []
    for path in paths:
        if not (path.parent / "completion.json").exists():
            raise RuntimeError(f"only completed measurements may enter this summary: {path}")
        for original in read_csv(path):
            if "method" not in original:
                raise RuntimeError("pass selected development digital results, not calibration candidate rows")
            row = dict(original)
            row.setdefault("protocol", "common_paid_information")
            row["total_energy"] = row.get("total_energy", row.get("actual_total_energy"))
            rows.append(row)
    bindings = {str(path): sha256(path) for path in paths}
    bindings[str(root / "inputs_001/development_inputs.json")] = sha256(root / "inputs_001/development_inputs.json")
    return rows, sources, bindings


def timing_rows(rows, digital_timing=None):
    result = []
    old = PROJECT / "outputs/COMMUNICATION-CONVERGENCE-20260915/COST_ANALYSIS_001/summary.csv"
    aliases = {"raw_quality": "raw_adaptive", "arithmetic_quality": "arithmetic_adaptive",
               "r3__full_grid_prediction_features": "wetok_r3", "perceptual_deepjscc": "perceptual_deepjscc"}
    for original in read_csv(old):
        if original["method"] not in aliases:
            continue
        scope = original["scope"]
        if scope not in ("primary_1_4_7", "high_13_19"):
            try:
                scope = f"snr_{float(scope):g}"
            except ValueError:
                continue
        result.append({"method": aliases[original["method"]], "complex_uses": 3060, "scope": scope,
                       "source_images": int(original["source_images"]), "timing_rows": int(original["timing_rows"]),
                       "TX_mean_ms": float(original["TX_mean_ms"]), "RX_mean_ms": float(original["RX_mean_ms"]),
                       "TX_p95_ms": float(original["TX_p95_ms"]), "RX_p95_ms": float(original["RX_p95_ms"]),
                       "single_method_peak_allocated_MiB": "", "communication_parameters": "", "NFE_mean": 0,
                       "scope_note": "sealed_previous_session_CPU_to_CPU; co-resident memory not attributed per model"})
    methods = sorted({row["method"] for row in rows if row.get("RX_seconds") and row["protocol"] == "common_paid_information"})
    for method in methods:
        for scope, support in SCOPES:
            chosen = [row for row in rows if row["method"] == method and row["protocol"] == "common_paid_information"
                      and float(row["snr_db"]) in support]
            tx = np.array([float(row["TX_seconds"]) * 1000 for row in chosen])
            rx = np.array([float(row["RX_seconds"]) * 1000 for row in chosen])
            result.append({"method": method, "complex_uses": int(chosen[0]["complex_uses"]), "scope": scope,
                           "source_images": len({row["image_id"] for row in chosen}), "timing_rows": len(chosen),
                           "TX_mean_ms": float(tx.mean()), "RX_mean_ms": float(rx.mean()),
                           "TX_p95_ms": float(np.quantile(tx, .95)), "RX_p95_ms": float(np.quantile(rx, .95)),
                           "single_method_peak_allocated_MiB": max(float(row["GPU_peak_allocated_bytes"]) for row in chosen) / 2**20,
                           "communication_parameters": int(chosen[0]["parameters_communication"]),
                           "NFE_mean": float(np.mean([int(row["NFE"]) for row in chosen])),
                           "scope_note": "current_author_FP32_batch1_CPU_to_CPU; load_IO_metrics_airtime_excluded"})
    if digital_timing is not None:
        receipt = json.loads((digital_timing / "completion.json").read_text())
        if receipt["status"] != "NEW_BUDGET_DIGITAL_TIMING_COMPLETE" or sha256(digital_timing / "summary.csv") != receipt["summary_sha256"]:
            raise RuntimeError("supplementary timing is incomplete or changed")
        result.extend(read_csv(digital_timing / "summary.csv"))
    return result


def figures(output, summary):
    paid = [row for row in summary if row["protocol"] == "common_paid_information"]
    methods = sorted({row["method"] for row in paid})
    for metric in METRICS:
        figure, axis = plt.subplots(figsize=(10, 6))
        for method in methods:
            chosen = [next(row for row in paid if row["method"] == method and row["scope"] == f"snr_{snr:g}") for snr in SNRS]
            axis.plot(SNRS, [row[metric] for row in chosen], marker="o", label=f"{label(method)} (N={chosen[0]['complex_uses']})")
        axis.set(xlabel="Physical SNR (dB)", ylabel=metric, title="Measured development results; budgets differ, not an equal-N ranking")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        figure.savefig(output / f"quality_{metric}.png", dpi=160)
        figure.savefig(output / f"quality_{metric}.pdf")
        plt.close(figure)
    for metric in ("lpips", "psnr_db"):
        figure, axes = plt.subplots(1, 5, figsize=(21, 4.8), sharey=True)
        for axis, snr in zip(axes, SNRS):
            for family, style in (("swin", "s-"), ("adjscc", "o-")):
                chosen = sorted((row for row in paid if row["method"].startswith(family) and row["scope"] == f"snr_{snr:g}"), key=lambda row: row["complex_uses"])
                axis.plot([row["complex_uses"] for row in chosen], [row[metric] for row in chosen], style, label=family, color=color(family))
            for method in methods:
                if method.startswith(("swin", "adjscc")):
                    continue
                chosen = next(row for row in paid if row["method"] == method and row["scope"] == f"snr_{snr:g}")
                axis.scatter(chosen["complex_uses"], chosen[metric], label=label(method), s=36, color=color(method))
            axis.set(title=f"{snr:g} dB", xlabel="Total complex uses", xscale="log")
            axis.grid(alpha=.25)
        axes[0].set_ylabel(metric)
        handles, names = axes[-1].get_legend_handles_labels()
        figure.legend(handles, names, loc="lower center", ncol=4, fontsize=8)
        figure.suptitle("Measured quality-resource points, no untrained masking or inferred frontier")
        figure.tight_layout(rect=(0, .17, 1, .94))
        figure.savefig(output / f"quality_resource_{metric}.png", dpi=160)
        figure.savefig(output / f"quality_resource_{metric}.pdf")
        plt.close(figure)


def cost_figure(output, summary, costs):
    chosen = [row for row in summary if row["scope"] == "primary_1_4_7" and row["protocol"] == "common_paid_information"]
    lookup = {(row["method"], row["scope"]): row for row in costs}
    figure, axis = plt.subplots(figsize=(12, 7))
    for row in chosen:
        timing = lookup.get((row["method"], row["scope"]))
        if timing is None:
            continue
        axis.scatter(float(timing["RX_mean_ms"]), row["lpips"], s=48, color=color(row["method"]),
                     label=f"{label(row['method'])} (N={row['complex_uses']})")
    axis.set(xscale="log", xlabel="Complete receiver processing time (ms), no airtime/queueing",
             ylabel="LPIPS (lower is better)", title="Primary1/4/7dB: measured quality vs processing cost, different budgets labelled")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "quality_RX_cost.png", dpi=160)
    figure.savefig(output / "quality_RX_cost.pdf")
    plt.close(figure)


def preview_images(output, rows, sources):
    directory = output / "preselected_images"
    directory.mkdir()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    lookup = {(row["method"], int(row["image_index"]), float(row["snr_db"]), int(row["seed"])): row
              for row in rows if row["protocol"] == "common_paid_information"}
    methods = ["raw_adaptive", "arithmetic_adaptive", "wetok_r3", "perceptual_deepjscc", "swin_ra32", "adjscc_c2"]
    manifest = []
    for index in (0, 25, 50, 75):
        source = np.load(sources[index]["path"], allow_pickle=False)
        original = Image.fromarray(source.transpose(1, 2, 0))
        original.save(directory / f"source_{index:04d}.png")
        for snr in (1., 7., 19.):
            board = Image.new("RGB", (256 * (1 + len(methods)), 324), "white")
            drawer = ImageDraw.Draw(board)
            board.paste(original, (0, 60))
            drawer.text((5, 5), f"Source {index:04d}\n{snr:g} dB / seed 2001", fill="black", font=font)
            for position, method in enumerate(methods, 1):
                row = lookup[method, index, snr, 2001]
                with np.load(row["image_archive"], allow_pickle=False) as archive:
                    reconstructed = archive["images"][int(row["image_slot"])]
                picture = Image.fromarray(np.rint(np.clip(reconstructed, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
                filename = f"{method}_source{index:04d}_snr{snr:g}_seed2001.png"
                picture.save(directory / filename)
                board.paste(picture, (256 * position, 60))
                drawer.text((256 * position + 5, 4), f"{label(method)} N={row['complex_uses']}\nPSNR {float(row['psnr_db']):.2f}\nLPIPS {float(row['lpips']):.4f}", fill="black", font=font)
                manifest.append({"file": filename, "image_index": index, "snr_db": snr, "seed": 2001,
                                 "method": method, "complex_uses": row["complex_uses"],
                                 "image_archive": row["image_archive"], "image_slot": row["image_slot"],
                                 "selection": "preregistered_indices_and_noise_not_quality_selected"})
            board.save(directory / f"comparison_source{index:04d}_snr{snr:g}.png")
            equal_methods = ("raw_adaptive_N4498", "arithmetic_adaptive_N4498", "swin_ra32")
            if all((method, index, snr, 2001) in lookup for method in equal_methods):
                equal_board = Image.new("RGB", (1024, 324), "white")
                equal_drawer = ImageDraw.Draw(equal_board)
                equal_board.paste(original, (0, 60))
                equal_drawer.text((5, 5), f"Source {index:04d}\nEqual N=4498 / E=8996\n{snr:g}dB / seed2001", fill="black", font=font)
                for position, method in enumerate(equal_methods, 1):
                    row = lookup[method, index, snr, 2001]
                    if int(row["complex_uses"]) != 4498:
                        raise RuntimeError("an equal-budget preview must not mix budgets")
                    with np.load(row["image_archive"], allow_pickle=False) as archive:
                        reconstructed = archive["images"][int(row["image_slot"])]
                    picture = Image.fromarray(np.rint(np.clip(reconstructed, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
                    filename = f"{method}_source{index:04d}_snr{snr:g}_seed2001.png"
                    picture.save(directory / filename)
                    equal_board.paste(picture, (256 * position, 60))
                    equal_drawer.text((256 * position + 5, 4), f"{label(method)}\nPSNR {float(row['psnr_db']):.2f}\nLPIPS {float(row['lpips']):.4f}", fill="black", font=font)
                    manifest.append({"file": filename, "image_index": index, "snr_db": snr, "seed": 2001,
                                     "method": method, "complex_uses": row["complex_uses"],
                                     "image_archive": row["image_archive"], "image_slot": row["image_slot"],
                                     "selection": "preregistered_equal_N4498_sources_and_noise_not_quality_selected"})
                equal_board.save(directory / f"equal_N4498_source{index:04d}_snr{snr:g}.png")
    write_csv(directory / "manifest.csv", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digital-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--digital-timing", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    rows, sources, bindings = load_rows(arguments.root, arguments.digital_runs)
    for path in (Path(__file__), EXPERIMENT / "src/external_positioning/analysis.py"):
        bindings[str(path)] = sha256(path)
    if arguments.digital_timing is not None:
        bindings[str(arguments.digital_timing / "summary.csv")] = sha256(arguments.digital_timing / "summary.csv")
    summary, paired, per_source = summarize(rows, sources)
    costs = timing_rows(rows, arguments.digital_timing)
    for name, chosen in (("summary", summary), ("paired", paired), ("per_source", per_source), ("online_costs", costs)):
        write_csv(arguments.output / f"{name}.csv", chosen)
    figures(arguments.output, summary)
    cost_figure(arguments.output, summary, costs)
    preview_images(arguments.output, rows, sources)
    for protocol, filename in (("common_paid_information", "common_protocol_tables.md"), ("author_assumed_information", "author_protocol_tables.md")):
        lines = [f"# {protocol}", "", "All points: 100 original development images, seeds 2001/2002/2003; unequal budgets are explicitly labelled.", ""]
        for scope, _support in SCOPES:
            lines.extend([f"## {scope}", "", "| Method | Total complex uses | Energy | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DINO ↑ |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
            for row in summary:
                if row["scope"] == scope and row["protocol"] == protocol:
                    lines.append(f"| {label(row['method'])} | {row['complex_uses']} | {row['total_energy']} | {row['psnr_db']:.4f} | {row['ssim']:.5f} | {row['lpips']:.5f} | {row['dino']:.5f} |")
            lines.append("")
        (arguments.output / filename).write_text("\n".join(lines) + "\n")
    if any(sha256(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("completed input results changed during analysis")
    write_json(arguments.output / "completion.json", {"status": "COMPLETED_NON_DIFFUSION_POINTS_SUMMARIZED",
               "finished_at": datetime.now().astimezone().isoformat(), "input_rows": len(rows), "summary_rows": len(summary),
               "paired_metric_rows": len(paired), "source_bootstrap_resamples": 10000, "bindings": bindings,
               "HiFi_partial_results_ranked": False, "equal_budget_digital_included": bool(arguments.digital_runs),
               "research_positioning_complete": False, "GPU_used": False})
    print(arguments.output)


if __name__ == "__main__":
    main()
