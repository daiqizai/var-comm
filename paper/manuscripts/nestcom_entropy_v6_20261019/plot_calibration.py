"""Deterministic drawing of already delivered cal100 endpoint means.

This script does not score images, interpolate metrics, or resample intervals.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evidence/cal100_clean_endpoint_means.csv"
OUT = ROOT / "figures"
WIDTH_MM, HEIGHT_MM = 88.56787, 62.0

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}

with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 32
assert {int(row["sources"]) for row in rows} == {100}
styles = {
    "0": {"label": "NeST-Com", "color": "#3775BA", "marker": "o", "linestyle": "-", "markersize": 3.6},
    "1": {"label": "ARPC", "color": "#B64342", "marker": "s", "linestyle": "--", "markersize": 4.0, "markerfacecolor": "white"},
}
outputs = []
with plt.rc_context(STYLE):
    for gray in (False, True):
        fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.215, top=0.82)
        for codec, style in styles.items():
            current = sorted((row for row in rows if row["codec"] == codec), key=lambda row: float(row["charged_source_bits_mean"]))
            assert len(current) == (25 if codec == "0" else 7)
            options = dict(style)
            if gray:
                options["color"] = "#222222" if codec == "0" else "#777777"
            ax.plot([float(row["charged_source_bits_mean"]) / 1000 for row in current],
                    [float(row["dino_mean"]) for row in current],
                    linewidth=1.5, markeredgewidth=0.8, clip_on=False, **options)
        ax.set_xlim(0, 9.2)
        ax.set_ylim(0.1, 0.9)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_xlabel("Mean charged source length (kbit)", labelpad=3)
        ax.set_ylabel("Clean DINO-L cosine similarity", labelpad=4)
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.5)
        ax.set_axisbelow(True)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.56, 0.98), ncol=2,
                   frameon=False, handlelength=2.4, columnspacing=1.4, handletextpad=0.55)
        suffix = "_gray" if gray else ""
        exts = ("png",) if gray else ("pdf", "svg", "png")
        for ext in exts:
            target = OUT / f"cal100_payload_clean{suffix}.{ext}"
            fig.savefig(target, dpi=600, metadata={"Creator": "NeST-Com deterministic manuscript drawing"} if ext in {"pdf", "svg"} else None)
            raw = target.read_bytes()
            outputs.append({"path": target.relative_to(ROOT).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        plt.close(fig)

manifest = {
    "schema": "MANUSCRIPT_CAL100_CLEAN_PLOT_V1",
    "input": {"path": SOURCE.relative_to(ROOT).as_posix(), "bytes": SOURCE.stat().st_size,
              "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
    "data_rows": 32, "source_count_per_endpoint": 100,
    "width_mm": WIDTH_MM, "height_mm": HEIGHT_MM,
    "statistics_computed": [], "scoring_calls": 0, "bootstrap_calls": 0,
    "joining_lines": "Straight guides between saved endpoint means only; not unmeasured equal-rate observations.",
    "style": STYLE, "series": styles, "outputs": outputs,
}
(OUT / "cal100_payload_clean_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"outputs": outputs, "scientific_calls": 0}))
