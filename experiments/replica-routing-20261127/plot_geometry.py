"""Publication-size figure of the registered, final-round, matched-gate contrasts."""
import csv
import os
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main(base):
    with (base/"analysis/mechanism_paired.csv").open(newline="") as f:
        values={(r["condition"],r["comparison"],r["metric"]):[float(r[k]) for k in ("estimate","lo","hi")]
                for r in csv.DictReader(f) if r["round"]=="3"}
    dest=base/"figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.linewidth":.75,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42,
                         "svg.hashsalt":"sem4-geometry-20261127"})
    keys=["antenna_db","combiner_db","interference_noise_db","latent_db","frontend_db"]
    names=["At both antennas","After weak-user combining","After combining, with noise","Demixed source latent","Decoder input latent"]
    fig,axes=plt.subplots(1,2,figsize=(7.25,2.95),sharey=True)
    fig.subplots_adjust(left=.245,right=.985,top=.78,bottom=.19,wspace=.13)
    for index,(ax,comp,title) in enumerate(zip(axes,("latent-linear","rgb-latent"),
                                        ("(a) Latent minus linear","(b) RGB minus latent"))):
        for j,(cond,label,color,marker) in enumerate([
                ("nearfar_14db","14 dB near–far spread","#255B91","o"),
                ("nearfar_18db_ood","18 dB near–far spread","#A34839","s")]):
            val=np.asarray([values[cond,comp,k] for k in keys])
            y=np.arange(len(keys))+(j-.5)*.23
            ax.errorbar(val[:,0],y,xerr=np.array([val[:,0]-val[:,1],val[:,2]-val[:,0]]),fmt=marker,
                        color=color,markerfacecolor=color if j==0 else "white",markeredgewidth=1.15,
                        ms=4.5,capsize=2,lw=1.2,label=label,zorder=3)
        ax.axvline(0,color="#484848",lw=.85)
        ax.set_xlim(-.49,2.79)
        ax.set_xticks([0,1,2])
        ax.grid(axis="x",color=".91",lw=.7)
        ax.set_axisbelow(True)
        ax.set_title(title,fontsize=9.5,loc="left",pad=8,fontweight="medium")
        ax.set_xlabel("Difference in error (dB)")
        ax.tick_params(axis="y",length=0,pad=5)
        ax.set_yticks(np.arange(len(keys)),names)
        ax.set_ylim(4.48,-.5)
        ax.spines["left"].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.55,.995),ncol=2,
               columnspacing=2.6,handletextpad=.4)
    for ext in ("pdf","svg","png"):
        fig.savefig(dest/f"replica_geometry.{ext}",dpi=400,facecolor="white")
    svg = base/"figures"/"replica_geometry.svg"
    # SVG path coordinates remain separated by newlines after stripping padding.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)

if __name__=="__main__":
    main(Path(sys.argv[1]))
