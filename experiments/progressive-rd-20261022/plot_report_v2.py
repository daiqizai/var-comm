"""Render only saved pilot statistics; preserve the first layout for diagnosis."""
import argparse, csv, json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ENDS=[(7,0),(7,25),(8,0),(8,84),(8,126),(9,0),(9,128),(10,0)]
STYLES=[('current_dc','Frozen current Dc','#767676','--','s'),
 ('old_recipe_control','Equal-update old recipe','#B64342',':','D'),
 ('multiprefix_unconditional','Multi-prefix unconditional','#3775BA','-','o'),
 ('endpoint_conditioned','Endpoint-conditioned','#5E966B','-.','^')]

def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def export(fig,out,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Check all nonempty text against the final fixed-size canvas.
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,(name,text.get_text(),box)
    for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=600,facecolor='white')
    fig.savefig(out/(name+'_print.png'),dpi=160,facecolor='white')
    plt.close(fig)

def main(root):
    root=pathlib.Path(root);out=root/'decoder_delivery_v1/figures_v2';out.mkdir(exist_ok=False)
    rows=read(root/'decoder_delivery_v1/summary.csv')
    pairs=read(root/'decoder_delivery_v1/paired_differences.csv')
    base=[r for r in read(root/'seed_delivery_v1/source_rd.csv') if r['split']=='calibration']
    rates=[np.mean([float(r['bits_per_pixel'])for r in base if(int(r['m']),int(r['K']))==e])for e in ENDS]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(178/25.4,65/25.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.75,wspace=.36)
    for ax,metric,title in zip(axs,['psnr_db','lpips_alex','ms_ssim'],['(a) PSNR (dB) ↑','(b) LPIPS-Alex ↓','(c) MS-SSIM ↑']):
        for arm,label,color,line,marker in STYLES:
            v=[next(r for r in rows if r['arm']==arm and r['metric']==metric and(int(r['m']),int(r['K']))==e)for e in ENDS]
            mean=np.array([float(r['mean'])for r in v]);lo=np.array([float(r['ci_low'])for r in v]);hi=np.array([float(r['ci_high'])for r in v])
            ax.errorbar(rates,mean,yerr=[mean-lo,hi-mean],color=color,ls=line,marker=marker,ms=3.5,lw=1.4,capsize=2,elinewidth=.7,label=label)
        ax.set_title(title,loc='left');ax.tick_params(length=3,width=.6);ax.grid(axis='y',lw=.4,color='#ddd')
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,frameon=False,columnspacing=1.4)
    fig.supxlabel('Mean terminated arithmetic rate (bits/pixel)',y=.035,fontsize=8.5)
    export(fig,out,'calibration_rd')

    fig,axs=plt.subplots(2,2,figsize=(178/25.4,105/25.4))
    fig.subplots_adjust(left=.105,right=.98,bottom=.14,top=.91,wspace=.34,hspace=.54)
    comparisons=[('multiprefix_unconditional_minus_old_recipe_control','Multi-prefix − old recipe','#3775BA','o'),('endpoint_conditioned_minus_multiprefix_unconditional','Conditioned − unconditional','#5E966B','^')]
    for i,(comp,label,color,marker) in enumerate(comparisons):
        for j,metric in enumerate(['psnr_db','lpips_alex']):
            ax=axs[i,j];v=[next(r for r in pairs if r['comparison']==comp and r['metric']==metric and(int(r['m']),int(r['K']))==e)for e in ENDS]
            mean=np.array([float(r['mean'])for r in v]);lo=np.array([float(r['ci_low'])for r in v]);hi=np.array([float(r['ci_high'])for r in v])
            ax.axhline(0,color='#555555',lw=.7,ls='--',zorder=0)
            ax.errorbar(rates,mean,yerr=[mean-lo,hi-mean],color=color,marker=marker,lw=1.4,ms=4,capsize=2,elinewidth=.9)
            ax.set_title('('+chr(97+i*2+j)+') '+label,loc='left',fontsize=8.5)
            ax.set_ylabel('ΔPSNR (dB) ↑' if j==0 else 'ΔLPIPS-Alex ↓');ax.tick_params(length=3,width=.6);ax.grid(axis='y',lw=.4,color='#ddd')
    fig.supxlabel('Mean terminated arithmetic rate (bits/pixel)',y=.035,fontsize=8.5)
    export(fig,out,'calibration_paired')
    (out/'caption.txt').write_text('Calibration100 results after 5,000 matched decoder updates per arm. The eight points retain the registered complete/within-scale endpoints and population mean terminated arithmetic rates. Overview error bars are pointwise 95% source-bootstrap intervals. Paired panels use direct within-source differences and the same single registered bootstrap bank; Δ is the first named arm minus the second. No score, bootstrap bank, endpoint, model or sampling setting is recomputed by this plotter. These calibration results do not establish held-out or wireless performance.\n')
    (out/'style_provenance.json').write_text(json.dumps(dict(skill='C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md',references=['references/design-theory.md','references/common-patterns.md','references/demos.md'],size_mm={'calibration_rd':[178,65],'calibration_paired':[178,105]},dpi=600,vector_pdf=True,editable_svg_text=True,prior_layout_issue='Repeated long x-axis labels overlapped and the rightmost label was clipped; use one shared label. Original figures remain preserved.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
