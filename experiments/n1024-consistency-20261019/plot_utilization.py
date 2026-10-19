"""Deterministic source-only empirical distribution; no model/score/statistical resampling."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(folder):
    folder=Path(folder); src=folder/'source_records_3000.csv'
    with src.open(encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==3000
    out=folder/'figures';out.mkdir()
    style={'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8.5,
           'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':8,
           'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
           'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'}
    plt.rcParams.update(style)
    for gray in (False,True):
        fig,axes=plt.subplots(1,3,figsize=(180/25.4,58/25.4),sharex=True,sharey=True)
        fig.subplots_adjust(left=.077,right=.985,bottom=.235,top=.735,wspace=.22)
        handles=[]
        for k,(ax,snr) in enumerate(zip(axes,(4,10,19))):
            for family,label,color,ls in [('WHOLE','Complete scales','#767676','--'),('PARTIAL','Partial scales','#3775BA','-')]:
                a=np.sort(np.array([float(r['utilization']) for r in rows if r['family']==family and int(r['snr_db'])==snr]));assert len(a)==500 and np.all((a>=0)&(a<=1))
                x=np.r_[0,a,1]; y=np.r_[0,np.arange(1,501)/500,1]
                line,=ax.step(x,y,where='post',color=('#272727' if family=='PARTIAL' else '#767676') if gray else color,lw=1.6,ls=ls,label=label)
                if k==0:handles.append(line)
            ax.set_title(f'({chr(97+k)}) {snr} dB',pad=5)
            ax.set_xlim(0,1.02);ax.set_ylim(0,1.03);ax.set_xticks([0,.25,.5,.75,1],['0','.25','.50','.75','1'])
            ax.set_yticks([0,.25,.5,.75,1],['0','.25','.50','.75','1'])
            ax.set_xlabel('Source-capacity utilization',labelpad=3)
            ax.tick_params(length=3,width=.7,pad=2)
        axes[0].set_ylabel('Empirical cumulative fraction',labelpad=4)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.995),ncol=2,frameon=False,handlelength=2.7,columnspacing=2.2)
        stem='source_utilization_ecdf'+('_grayscale' if gray else '')
        if not gray:
            for ext in ('pdf','svg'):fig.savefig(out/(stem+'.'+ext))
        fig.savefig(out/(stem+'.png'),dpi=300 if gray else 600)
        plt.close(fig)
    caption='Empirical distributions of source-capacity utilization over common500, counting each source once per family and SNR (3,000 records). Utilization is the actual terminated arithmetic bitstream length divided by k - 13 - 16; byte padding is not charged as arithmetic bits. Both families use the same frozen H800 source implementation. The original 4/10 dB fallback policies and the corrected T1-C m10 policies at 19 dB are retained. The curves describe source packing, not wireless image quality. Historical 4/10 dB whole-scale quality may only be associated with these curves after the separately documented event-version repair. No source, channel, reconstruction, metric, or bootstrap call is added by this plot.\n'
    (out/'caption_en.txt').write_text(caption,encoding='utf8')
    (out/'style.json').write_text(json.dumps(style,indent=2)+'\n',encoding='utf8')
    meta=dict(input={'path':str(src),'sha256':sha(src)},source_records=3000,sources_per_panel_family=500,canvas_mm=[180,58],png_dpi=600,grayscale_dpi=300,svg_editable_text=True,skill_actual_path='C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md',references=['references/design-theory.md','references/common-patterns.md','references/demos.md','.research/figures_redesign_v1/references/figure_ophthal_review__plot_trend.py'],files=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir())])
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    import sys
    run(sys.argv[1])
