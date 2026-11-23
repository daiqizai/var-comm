"""Plot the stored, source-paired digital replay results at their report size."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def draw(root):
    digital = root / "digital"
    risk = read(digital / "risk/paired.csv")
    quality = read(digital / "paired.csv")
    losses = read(digital / "risk/summary.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                        "axes.spines.right":False,"axes.linewidth":.8,"pdf.fonttype":42,"svg.fonttype":"none",
                        "svg.hashsalt":"digital_feedback_risk_20261123"})
    fig,(left,right) = plt.subplots(1,2,figsize=(8.2,3.33),gridspec_kw={"width_ratios":[1.22,1]})
    fig.subplots_adjust(left=.095,right=.975,top=.80,bottom=.18,wspace=.39)
    colors = {"0.6":"#B64342","0.95":"#3775BA"}
    for p, offset, marker in [("0.6",-.08,"s"),("0.95",.08,"o")]:
        q = next(r for r in quality if r["p"]==p and r["metric"]=="dino_cosine"
                 and r["left"]=="static" and r["right"]=="two_every_two")
        vals = [(-float(q["mean_delta"]),-float(q["ci_high"]),-float(q["ci_low"]))]
        for cap in ("0.05","0.01"):
            row = next(r for r in risk if r["p_design"]==r["p_actual"]==p and r["cap"]==cap
                       and r["reference"]=="static" and r["metric"]=="dino_cosine")
            vals.append(tuple(float(row[k]) for k in ("mean_delta","ci_low","ci_high")))
        val,lo,hi = np.array(vals).T
        label = ("Fast" if p=="0.6" else "Slow") + f" state changes ($p={float(p):.2f}$)"
        left.errorbar(np.arange(3)+offset,val,yerr=[val-lo,hi-val],color=colors[p],fmt=marker,
                      ms=5,capsize=3,lw=1.3,label=label)
    left.axhline(0,color="#767676",lw=.85)
    left.set(xticks=[0,1,2],xticklabels=["Unconstrained","5% cap","1% cap"],xlim=(-.34,2.35),
             ylabel="DINO cosine gain over static",ylim=(-.032,.165))
    left.set_title("(a) Image quality at matched persistence",loc="left",fontsize=10,pad=9)
    xy=np.arange(2)
    for p,dx in [("0.6",-.18),("0.95",.18)]:
        values=[100*float(next(r["mean"] for r in losses if r["p_design"]=="0.95" and r["p_actual"]==p
                        and r["cap"]==cap and r["metric"]=="rejected_or_null_fraction")) for cap in ("0.01","0.05")]
        bars=right.bar(xy+dx,values,width=.34,color=colors[p],edgecolor=colors[p],
                       hatch="//" if p=="0.6" else None,alpha=.91)
        right.bar_label(bars,labels=[f"{v:.2f}%" for v in values],padding=3,fontsize=8)
    right.set(xticks=xy,xticklabels=["1% design cap","5% design cap"],ylabel="Rejected or NULL (%)",ylim=(0,32))
    right.set_title("(b) Policy designed for slow changes",loc="left",fontsize=10,pad=9)
    for ax in (left,right):
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="#e5e5e5",lw=.7)
    fig.legend(*left.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.997),ncol=2,frameon=False)
    out=root/"figures"
    out.mkdir(exist_ok=True)
    for ext, metadata in (("pdf",{"CreationDate":None}),("png",{}),("svg",{"Date":None})):
        path = out/f"digital_feedback_risk.{ext}"
        fig.savefig(path,dpi=300,facecolor="white",metadata=metadata)
        if ext=="svg":
            path.write_bytes(("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n").encode())
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("report_dir",type=Path)
    draw(parser.parse_args().report_dir)
