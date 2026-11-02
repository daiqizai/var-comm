#!/usr/bin/env python3
"""Reflow frozen NeST-Com V9 Figure 2 data into an exact-size 1x4 figure.

Example:
  python draw_figure2_1x4.py --data figure2_data.csv --out-dir figures
Only the displayed equal-budget x unit changes: bits / 1000 = kbit.
The main figure shows six methods. --include-arpc retains the complete
seven-method comparison in a supplementary figure. All 92 input rows remain
unchanged; no model, summary statistic, interval, or smoothing is recomputed.
Dependencies: matplotlib, numpy, Pillow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from PIL import Image

DEFAULT_WIDTH_MM = 181.35325861353257
STYLE = {
    "nest": dict(label="NeST-Com", color="#1F67AA", marker="o", linestyle="-", linewidth=1.5, markersize=3.1, zorder=7),
    "whole": dict(label="WHOLE", color="#566F85", marker="s", linestyle="--", linewidth=1.15, markersize=2.9, zorder=6),
    "ada": dict(label="Ada-TokenCom", color="#C66A24", marker="v", linestyle="-.", linewidth=1.25, markersize=3.1, zorder=5),
    "arpc": dict(label="ARPC + LDPC", color="#9364A1", marker="^", linestyle="--", linewidth=.95, markersize=2.7, zorder=3),
    "latent": dict(label="Latent JSCC", color="#657F3C", marker="D", linestyle=":", linewidth=.95, markersize=2.7, zorder=3),
    "swin": dict(label="SwinJSCC", color="#A65D63", marker="P", linestyle="-.", linewidth=.95, markersize=2.7, zorder=3),
    "bpg": dict(label="BPG + LDPC", color="#676767", marker="x", linestyle=(0, (4, 2, 1, 2)), linewidth=.95, markersize=2.9, zorder=3),
}
FIXED = [("final_NeST", "nest"), ("WHOLE_final_receiver", "whole"), ("Ada_frozen", "ada"), ("ARPC", "arpc"), ("Latent_JSCC", "latent"), ("SwinJSCC", "swin"), ("BPG_LDPC", "bpg")]
EQUAL = [("Token_NeST", "nest"), ("WHOLE", "whole"), ("Ada", "ada")]
PANELS = [
    ("a", "fixed_symbols", "dists", (0.10, 0.42), [0.1, 0.2, 0.3, 0.4]),
    ("b", "fixed_symbols", "dreamsim", (0.05, 0.62), [0.1, 0.3, 0.5]),
    ("c", "equal_information_bits", "dists", (0.10, 0.32), [0.1, 0.2, 0.3]),
    ("d", "equal_information_bits", "dreamsim", (0.07, 0.42), [0.1, 0.2, 0.3, 0.4]),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "figures/figure2_data.csv")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    p.add_argument("--width-mm", type=float, default=DEFAULT_WIDTH_MM)
    p.add_argument("--height-mm", type=float, default=64.0)
    p.add_argument("--include-arpc", action="store_true", help="Render the complete comparison for the supplement.")
    p.add_argument("--stem", default="figure2_1x4")
    p.add_argument("--no-aliases", action="store_true", help="Do not overwrite the main Figure 2 compatibility filenames.")
    return p.parse_args()


def load_data(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"panel", "experiment", "method", "metric", "N", "x", "mean", "ci95_low", "ci95_high"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Input must be the frozen V9 figure2_data.csv with its original columns.")
    if len(rows) != 92:
        raise ValueError(f"Expected exactly 92 frozen records, got {len(rows)}. Review a changed data contract explicitly.")
    keys = [(r["experiment"], r["method"], r["metric"], float(r["x"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate experiment/method/metric/x records.")
    return rows


def main():
    args = parse_args()
    rows = load_data(args.data)
    w, h = args.width_mm, args.height_mm
    if not (175 <= w <= 185 and 60 <= h <= 70):
        raise ValueError("This layout is calibrated for 175-185 mm width and 60-70 mm height.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    left, right = 8.8, 3.0
    gaps = [8.2, 11.5, 8.2]
    pw = (w - left - right - sum(gaps)) / 4
    bottom, ph = 10.5, 31.0
    xs = [left]
    for g in gaps:
        xs.append(xs[-1] + pw + g)
    counts = {}
    plotted_keys = []
    interval_extents = {}
    rc = {
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.labelsize": 8.0, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": .55, "ytick.major.width": .55,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2.1, "ytick.major.pad": 2.1,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(w / 25.4, h / 25.4), dpi=150)
        axes = []
        for j, (letter, exp, metric, ylim, yticks) in enumerate(PANELS):
            ax = fig.add_axes([xs[j] / w, bottom / h, pw / w, ph / h])
            axes.append(ax)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E2E5E8", linewidth=.45)
            mapping = FIXED if exp == "fixed_symbols" else EQUAL
            if not args.include_arpc:
                mapping = [(method, key) for method, key in mapping if key != "arpc"]
            expected_x = [4., 7., 10., 19.] if exp == "fixed_symbols" else [620., 1212., 1302., 1940., 2248., 2577.]
            n = 0
            all_low, all_high = [], []
            for method, stylekey in mapping:
                subset = sorted((r for r in rows if r["experiment"] == exp and r["metric"] == metric and r["method"] == method), key=lambda r: float(r["x"]))
                raw_x = [float(r["x"]) for r in subset]
                if raw_x != expected_x:
                    raise ValueError(f"Unexpected x values in {letter}/{method}: {raw_x}")
                if exp == "fixed_symbols" and any(float(r["N"]) != 1024 for r in subset):
                    raise ValueError("Fixed-symbol panels must use N=1024.")
                x = np.asarray(raw_x) / (1000 if exp == "equal_information_bits" else 1)
                y, lo, hi = (np.array([float(r[k]) for r in subset]) for k in ("mean", "ci95_low", "ci95_high"))
                if not (np.isfinite(np.r_[x, y, lo, hi]).all() and (lo <= y).all() and (y <= hi).all()):
                    raise ValueError(f"Invalid mean/CI records in {letter}/{method}")
                if min(lo) < ylim[0] or max(hi) > ylim[1]:
                    raise ValueError(f"An interval would be clipped in panel {letter}.")
                st = STYLE[stylekey]
                ax.errorbar(x, y, yerr=[y-lo, hi-y], capsize=1.15, elinewidth=.5,
                            capthick=.5, markeredgewidth=.7,
                            markerfacecolor=st["color"] if stylekey == "nest" else "white", **st)
                n += len(subset)
                plotted_keys.extend((r["experiment"], r["method"], r["metric"], float(r["x"])) for r in subset)
                all_low.extend(lo.tolist()); all_high.extend(hi.tolist())
            counts[letter] = n
            interval_extents[letter] = [min(all_low), max(all_high)]
            ax.set_ylim(*ylim); ax.set_yticks(yticks)
            if exp == "fixed_symbols":
                ax.set_xlim(3.2, 19.8); ax.set_xticks([4, 7, 10, 19]); ax.set_xlabel("SNR (dB)", labelpad=2.6)
            else:
                ax.set_xlim(.52, 2.68); ax.set_xticks([.6, 1.3, 1.9, 2.6]); ax.set_xlabel("Total budget (kbit)", labelpad=2.6)
            metric_name = "DISTS" if metric == "dists" else "DreamSim"
            fig.text(xs[j] / w, 44.2 / h, f"({letter}) {metric_name} ↓", fontsize=8.5, ha="left", va="baseline")
        for start, end, label in [(0, 1, r"Fixed symbols: $N=1024$"), (2, 3, "Equal information budgets")]:
            center = (xs[start] + xs[end] + pw) / 2
            fig.text(center / w, 49.7 / h, label, fontsize=8.2, ha="center", va="baseline", weight="medium")
            fig.add_artist(Line2D([xs[start]/w, (xs[end]+pw)/w], [48.2/h]*2,
                                 transform=fig.transFigure, color="#C8CED4", lw=.5))
        legends = []
        lower_legend = [key for key in list(STYLE)[3:] if args.include_arpc or key != "arpc"]
        for keys, top_mm in [(list(STYLE)[:3], h-1.2), (lower_legend, h-6.3)]:
            handles = []
            for key in keys:
                st = STYLE[key].copy(); st.pop("zorder")
                handles.append(Line2D([], [], markeredgewidth=.7,
                                      markerfacecolor=st["color"] if key == "nest" else "white", **st))
            legends.append(fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, top_mm/h),
                           ncol=len(keys), frameon=False, borderpad=0, borderaxespad=0,
                           handlelength=2.0, handletextpad=.5, columnspacing=1.8))
        expected_count = 92 if args.include_arpc else 84
        expected_keys = {(r["experiment"], r["method"], r["metric"], float(r["x"]))
                         for r in rows if args.include_arpc or r["method"] != "ARPC"}
        if len(plotted_keys) != expected_count or set(plotted_keys) != expected_keys:
            raise ValueError("The figure must consume every declared display record exactly once.")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        clipped = []
        for artist in fig.findobj(Text):
            if not artist.get_visible() or not artist.get_text():
                continue
            bb = artist.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1:
                clipped.append(artist.get_text())
        if clipped:
            raise ValueError(f"Text outside the page: {clipped}")
        for legend in legends:
            lb = legend.get_window_extent(renderer)
            if any(lb.overlaps(ax.get_window_extent(renderer)) for ax in axes):
                raise ValueError("Legend overlaps a data panel.")
        stem = args.out_dir / args.stem
        for ext in ("pdf", "svg", "png"):
            fig.savefig(stem.with_suffix("."+ext), dpi=600, bbox_inches=None, facecolor="white")
        svg = stem.with_suffix(".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
        fig.savefig(args.out_dir / (args.stem + "_print_size.png"), dpi=150, bbox_inches=None)
        plt.close(fig)
    # Grayscale is a rendering preview only; it does not modify scientific data.
    with Image.open(args.out_dir / (args.stem + "_print_size.png")) as im:
        im.convert("L").save(args.out_dir / (args.stem + "_grayscale.png"))
    # Compatibility aliases use exactly the same output bytes.
    if not args.no_aliases:
        for ext in ("pdf", "svg", "png"):
            shutil.copyfile(args.out_dir / (args.stem+"."+ext), args.out_dir / ("figure2."+ext))
        for suffix in ("_print_size.png", "_grayscale.png"):
            shutil.copyfile(args.out_dir / (args.stem+suffix), args.out_dir / ("figure2"+suffix))
    copy = args.out_dir / "figure2_data.csv"
    if args.data.resolve() != copy.resolve():
        shutil.copyfile(args.data, copy)
    validation = {
        "source_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "copied_data_sha256": hashlib.sha256(copy.read_bytes()).hexdigest(),
        "source_records": len(rows), "plotted_means": len(plotted_keys),
        "plotted_confidence_intervals": len(plotted_keys), "panel_counts": counts,
        "display_scope": "complete_with_arpc_adaptation" if args.include_arpc else "main_six_methods",
        "arpc_rows_retained_in_input": 8,
        "main_omission_reason": None if args.include_arpc else "ARPC is attributed as source-coding prior work; its digital-link adaptation is reported in the complete supplement.",
        "width_mm": w, "height_mm": h, "panel_width_mm": pw,
        "y_limits": {p[0]: p[3] for p in PANELS}, "source_ci_extents": interval_extents,
        "display_transform": "Equal-information x: bits / 1000 = kbit; all other values unchanged.",
        "minimum_nominal_font_pt": 7.5, "text_outside_canvas": clipped,
        "legend_overlaps_axes": False, "matplotlib_version": matplotlib.__version__,
        "note": "Intervals are loaded from V9; no bootstrap or metric recomputation. Line segments join observed means only. Visual review remains required.",
    }
    (args.out_dir / (args.stem + "_validation.json")).write_text(json.dumps(validation, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(stem), "means": len(plotted_keys), "intervals": len(plotted_keys),
                      "size_mm": [w, h], "text_outside_canvas": clipped}, indent=2))


if __name__ == "__main__":
    main()
