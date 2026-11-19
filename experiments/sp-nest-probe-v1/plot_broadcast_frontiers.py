"""Plot measured non-dominated Pilot50 policies for each receiver pair."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


STYLES = {
    "sp": ("SP, within-scale", "#3775BA", "o", "-"),
    "sp_whole": ("SP, whole-scale", "#9A4D8E", "^", ":"),
    "oma": ("Time division", "#42949E", "D", "--"),
    "single": ("Single codeword", "#575757", "s", "-."),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    with (a.evidence / "observed_frontiers.csv").open(encoding="utf8") as f:
        points = list(csv.DictReader(f))
    with (a.evidence / "selected_operating_points.csv").open(encoding="utf8") as f:
        selected = {r["pair"]: r for r in csv.DictReader(f) if r["family"] == "SP" and float(r["lambda_weak"]) == .5}
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "font.size":8.5, "axes.labelsize":9.1, "axes.titlesize":10, "axes.spines.right":False,
        "axes.spines.top":False, "axes.linewidth":.65, "xtick.labelsize":8, "ytick.labelsize":8,
        "pdf.fonttype":42, "svg.fonttype":"none", "savefig.facecolor":"white"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex="row", sharey="row")
    fig.subplots_adjust(top=.865, bottom=.105, left=.099, right=.98, wspace=.16, hspace=.39)
    for row, (metric, title) in enumerate((("dino","DINOv2 cosine"),("dists","DISTS"))):
        for col, strong in enumerate((19,13)):
            ax = axes[row, col]
            key = f"4_{strong}"
            for name, (_, colr, marker, style) in STYLES.items():
                xy = sorted((float(r["weak"]), float(r["strong"])) for r in points
                            if r["pair"] == key and r["metric"] == metric and r["family"] == name)
                if not xy:
                    raise ValueError((key, metric, name))
                x, y = zip(*xy)
                ax.plot(x, y, color=colr, marker=marker, linestyle=style, markersize=4.1,
                        markerfacecolor="white", markeredgewidth=1., linewidth=1.05, alpha=.95)
            picked = selected[key]
            ax.scatter([float(picked[metric+"_weak"])], [float(picked[metric+"_strong"])], s=91,
                       facecolors="none", edgecolors=STYLES["sp"][1], linewidths=1.05, zorder=5)
            suffix = "higher is better" if row == 0 else "lower is better"
            ax.set_title(f"({chr(97+2*row+col)})  4 / {strong} dB", loc="left", fontweight="semibold", pad=5)
            ax.set_xlabel(f"Weak receiver {title}")
            if col == 0:
                ax.set_ylabel(f"Strong receiver {title}")
            if col == 1:
                ax.text(.99,1.035,suffix,fontsize=7.4,color="#515151",transform=ax.transAxes,ha="right",va="bottom")
            ax.grid(axis="y", color="#dedede", lw=.55)
            ax.set_axisbelow(True)
            ax.margins(x=.045,y=.075)
            ax.tick_params(length=3,width=.65,pad=3)
    handles = [Line2D([], [], label=label, color=c, marker=m, ls=style, ms=4.5, lw=1.05,
                      markerfacecolor="white") for label,c,m,style in STYLES.values()]
    fig.legend(handles=handles,ncol=4,loc="upper center",bbox_to_anchor=(.52,.995),frameon=False,
               handlelength=2.1,handletextpad=.5,columnspacing=1.8,fontsize=8.4)
    fig.text(.5,.022,"Markers are measured policies. Rings mark the SP policy chosen by equal-weight DINOv2.",
             ha="center",fontsize=8.1,color="#383838")
    a.out.mkdir(exist_ok=True)
    for ext in ("pdf","svg","png"):
        path = a.out / ("receiver_frontiers."+ext)
        fig.savefig(path, dpi=300)
        if ext == "svg":
            # The SVG serializer puts blanks before newlines inside path coordinates.
            lines = path.read_text(encoding="utf8").splitlines()
            path.write_bytes(("\n".join(line.rstrip(" \t") for line in lines) + "\n").encode("utf8"))
    plt.close(fig)


if __name__ == "__main__":
    main()
