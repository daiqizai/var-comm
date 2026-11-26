"""Source quality and measured batch-one wall time for the same frozen receivers."""
from pathlib import Path
import csv,sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def data(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def pick(rows,**keys):
    matches=[r for r in rows if all(r[k]==v for k,v in keys.items())]
    assert len(matches)==1,keys
    p,lo,hi=[float(matches[0][k]) for k in ("estimate","lo","hi")]
    return p,[[p-lo],[hi-p]]

def main(base):
    q=data(base/"analysis/quality_means.csv")
    t=data(base/"analysis/timing.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.8,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42,"svg.hashsalt":"4x2-cost"})
    fig,axes=plt.subplots(1,2,figsize=(7.25,3.3),sharex=True)
    fig.subplots_adjust(left=.1,right=.985,top=.72,bottom=.19,wspace=.24)
    schemes=[("latent","Latent only (0)","o","#757575"),("best_fixed","Power order (0,1,2)","s","#22669A"),
             ("adaptive","Final round (0,0,3)","D","#A64836"),("full","All users (4,4,4)","^","#358077")]
    for ax,cond,title in zip(axes,("nearfar_14db","nearfar_18db_ood"),("(a) 14 dB near–far spread","(b) 18 dB near–far spread")):
        for arm,label,marker,color in schemes:
            y,ey=pick(q,layout="rotated",condition=cond,metric="weak_psnr",arm=arm)
            x,ex=pick(t,condition=cond,quantity="mean_rx_ms",comparison=arm)
            ax.errorbar(x,y,xerr=ex,yerr=ey,fmt=marker,color=color,markerfacecolor="white" if arm=="adaptive" else color,
                        markersize=5.5,markeredgewidth=1.1,capsize=2.2,lw=1.1,label=label)
        ax.grid(axis="y",color=".91",lw=.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Four-user receive time (ms)")
        ax.set_title(title,fontsize=9.5,loc="left",pad=7,fontweight="medium")
    axes[0].set_ylabel("Weak-user PSNR (dB)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,bbox_to_anchor=(.53,1.015),columnspacing=2.3)
    for ext in ("pdf","svg","png"):
        fig.savefig(base/"figures"/f"quality_cost.{ext}",dpi=400,facecolor="white")
    svg = base/"figures"/"quality_cost.svg"
    # SVG path coordinates remain separated by newlines after stripping padding.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)

if __name__=="__main__":
    main(Path(sys.argv[1]))
