"""Plot saved calibration metrics only; no models or recomputation of scores."""
import argparse,csv,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
    root=pathlib.Path(root);out=root/'decoder_delivery_v1/figures';out.mkdir(exist_ok=False)
    with(root/'decoder_delivery_v1/summary.csv').open()as f:rows=list(csv.DictReader(f))
    with(root/'seed_delivery_v1/source_rd.csv').open()as f:base=[r for r in csv.DictReader(f)if r['split']=='calibration']
    ends=[(7,0),(7,25),(8,0),(8,84),(8,126),(9,0),(9,128),(10,0)]
    rates=[np.mean([float(r['bits_per_pixel'])for r in base if(int(r['m']),int(r['K']))==e])for e in ends]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none'})
    spec=[('current_dc','Frozen current Dc','#767676','--','s'),('old_recipe_control','Equal-update old recipe','#B64342',':','D'),('multiprefix_unconditional','Multi-prefix unconditional','#3775BA','-','o'),('endpoint_conditioned','Endpoint-conditioned','#5E966B','-.','^')]
    fig,axs=plt.subplots(1,3,figsize=(178/25.4,65/25.4));fig.subplots_adjust(left=.075,right=.99,bottom=.22,top=.75,wspace=.36)
    for ax,metric,title in zip(axs,['psnr_db','lpips_alex','ms_ssim'],['(a) PSNR (dB) ↑','(b) LPIPS-Alex ↓','(c) MS-SSIM ↑']):
        for arm,label,color,line,marker in spec:
            v=[next(r for r in rows if r['arm']==arm and r['metric']==metric and(int(r['m']),int(r['K']))==e)for e in ends]
            mean=np.array([float(r['mean'])for r in v]);lo=np.array([float(r['ci_low'])for r in v]);hi=np.array([float(r['ci_high'])for r in v])
            ax.errorbar(rates,mean,yerr=[mean-lo,hi-mean],color=color,ls=line,marker=marker,ms=3.5,lw=1.4,capsize=2,elinewidth=.7,label=label)
        ax.set_title(title,loc='left');ax.set_xlabel('Mean terminated source rate (bpp)');ax.tick_params(length=3,width=.6);ax.grid(axis='y',lw=.4,color='#dddddd')
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,frameon=False,columnspacing=1.4)
    for suffix in ['pdf','svg','png']:fig.savefig(out/f'calibration_rd.{suffix}',dpi=600,facecolor='white')
    fig.savefig(out/'calibration_rd_print.png',dpi=160,facecolor='white');plt.close(fig)
    (out/'caption.txt').write_text('Calibration100 source-rate versus reconstruction quality after 5,000 matched decoder updates. Points are the eight registered complete or within-scale endpoints; horizontal coordinates are population mean terminated arithmetic bits per pixel, not wireless symbol budgets. Error bars are pointwise 95% source-bootstrap intervals using one shared bank. Endpoints, encoder, codebook and VAR are fixed. These are calibration results, not a held-out or wireless performance claim.\n')
    (out/'style_provenance.json').write_text(json.dumps(dict(skill='C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md',references=['references/design-theory.md','references/common-patterns.md','references/demos.md'],size_mm=[178,65],type='vector PDF and editable SVG',dpi=600),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
