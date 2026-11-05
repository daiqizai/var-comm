"""Plot the completed 256-only source-packet comparison; reads fixed summary CSV."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})
with (ROOT / "summary.csv").open(newline="", encoding="utf8") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 64
budgets = [512, 768, 1024, 1536, 2048, 3072, 4096, 8192]
metrics = [("psnr_db", "PSNR (dB), higher is better"),
           ("lpips_alex", "LPIPS-Alex, lower is better"),
           ("dists", "DISTS, lower is better"),
           ("dreamsim", "DreamSim, lower is better")]
styles = {"original": ("Original RDVQ codec", "#767676", "s", "--"),
          "new": ("New codec, step 82,858", "#3775BA", "o", "-")}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.5))
count = 0
for i, (ax, (metric, title)) in enumerate(zip(axs.flat, metrics)):
    for method, (label, color, marker, linestyle) in styles.items():
        group = sorted([r for r in rows if r["method"] == method and r["metric"] == metric],
                       key=lambda r: int(r["budget_bits"]))
        assert [int(r["budget_bits"]) for r in group] == budgets
        y, lo, hi = [np.array([float(r[k]) for r in group]) for k in ("mean", "ci_low", "ci_high")]
        ax.errorbar(budgets, y, yerr=[y-lo, hi-y], label=label, color=color,
                    marker=marker, linestyle=linestyle, markersize=3.4,
                    linewidth=1.25, elinewidth=.65, capsize=1.8, capthick=.65)
        count += len(group)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(budgets))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlim(470, 9000)
    ax.set_title(f"({chr(97+i)}) {title}", loc="left", pad=7)
    ax.set_xlabel("Packet budget limit (bit; log scale)")
    ax.grid(axis="y", color="#E7E7E7", linewidth=.6)
    ax.set_axisbelow(True)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2,
           bbox_to_anchor=(.5, .995), fontsize=9, handlelength=2.7)
fig.subplots_adjust(left=.085, right=.985, bottom=.18, top=.885, hspace=.66, wspace=.22)
fig.text(.5, .025, "256 x 256 | 500 fixed sources | pointwise 95% source-bootstrap CIs", ha="center", fontsize=8)
assert count == 64
for suffix in ("pdf", "svg", "png"):
    fig.savefig(ROOT / f"quality_vs_budget.{suffix}", dpi=600 if suffix == "png" else 300)
svg = ROOT / "quality_vs_budget.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines()) + "\n", encoding="utf8")
fig.savefig(ROOT / "quality_vs_budget_preview.png", dpi=150)
(ROOT / "figure_identity.json").write_text(json.dumps({
    "input": "summary.csv", "input_sha256": hashlib.sha256((ROOT / "summary.csv").read_bytes()).hexdigest(),
    "means": count, "confidence_intervals": count, "width_mm": 182.88, "height_mm": 139.7,
    "x_axis": "registered paid packet upper budget; logarithmic base 2, all eight numeric positions retained",
    "font": "DejaVu Sans", "font_point_sizes": {"title": 10, "axis": 9, "ticks": 8},
    "scope": "Source-packet codec comparison, not a new PHY evaluation or same-actual-bitrate comparison",
    "skills": ["scientific-figure-making/SKILL.md", "references/design-theory.md", "references/common-patterns.md"],
}, indent=2) + "\n", encoding="utf8")
print("64 means and 64 intervals exported to PDF/SVG/600-dpi PNG")
