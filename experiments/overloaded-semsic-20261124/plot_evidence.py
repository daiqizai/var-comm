"""Plot primary four-user paired contrasts, with pointwise group-bootstrap intervals."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("paired", type=Path)
    p.add_argument("output", type=Path)
    args=p.parse_args()
    with args.paired.open(newline="") as f:
        records={(r["metric"],r["condition"],r["comparison"]):r for r in csv.DictReader(f) if r["split"]=="common500"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig, axes = plt.subplots(1,2,figsize=(6.65,2.5),sharey=True)
    comparisons=[("latent_minus_linear_fixed_gate","#295E96","o","Latent minus linear (fixed gates)"),
                 ("latent_minus_rgb_reencode","#B95D36","s","Latent minus RGB re-encoding")]
    for ax,metric,title in zip(axes,("weak_psnr_db","interference_nmse_db_r3"),
                               ("(a) Weak-user PSNR","(b) Residual interference, round 3")):
        ax.axvline(0, color="#777777", lw=.9, linestyle="--")
        for base,cond in [(1,"nearfar_14db"),(0,"nearfar_18db_ood")]:
            for shift,(key,color,marker,_) in zip((.14,-.14),comparisons):
                row=records[metric,cond,key]
                mean,lo,hi=(float(row[k]) for k in ("mean","ci_low","ci_high"))
                ax.errorbar(mean,base+shift,xerr=[[mean-lo],[hi-mean]],color=color,marker=marker,
                            ms=5,capsize=3,lw=1.4,elinewidth=1.4,clip_on=True)
        ax.set_title(title,loc="left",fontsize=9.5,pad=7)
        ax.set_xlabel("Paired difference (dB)",fontsize=9)
        ax.set_ylim(-.38,1.38)
        ax.set_yticks([0,1],["18 dB spread","14 dB spread"])
        ax.grid(axis="x",color="#DDE1E6",linewidth=.45)
        ax.set_axisbelow(True)
        ax.tick_params(length=3,labelsize=8)
    axes[0].set_xlim(-1.24,3.04);axes[0].set_xticks([-1,0,1,2,3])
    axes[1].set_xlim(-3.05,1.66);axes[1].set_xticks([-3,-2,-1,0,1])
    handles=[Line2D([],[],marker=m,linestyle="none",color=c,markersize=5,label=t) for _,c,m,t in comparisons]
    fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,fontsize=8.5,
               columnspacing=1.8,handletextpad=.2,bbox_to_anchor=(.5,0))
    fig.subplots_adjust(left=.13,right=.985,top=.86,bottom=.34,wspace=.18)
    args.output.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","png"):
        fig.savefig(args.output / f"replica_contrasts.{ext}",dpi=220,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
