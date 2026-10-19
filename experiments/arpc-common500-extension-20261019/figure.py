"""Deterministic rendering of completed statistics; never computes scores or intervals."""
import argparse,time,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from common import *
PANELS=[('psnr_db','(a) PSNR (dB) ↑',1),('lpips_alex','(b) LPIPS ↓',1),('dreamsim','(c) DreamSim ↓',1),('convnext_top1_source_prediction','(d) Agreement (%) ↑',100)]
def gray(hexcolor):
    v=[int(hexcolor[i:i+2],16)/255 for i in (1,3,5)];y=sum(a*b for a,b in zip(v,(.2126,.7152,.0722)));return (y,y,y)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    assert not out.exists() and out.is_absolute() and out.parent.is_dir()
    d=O/'delivery_v1';done=read(d/'completion.json');assert done['status']=='COMMON500_FIVE_METHOD_STATISTICS_COMPLETE'and done['sources']==500
    for pin in done['outputs']:verify(pin)
    style=read(E/'figure_style.json');rows=csvrows(d/'common500_summary.csv');lookup={(r['method'],r['metric'],int(r['snr_db'])):r for r in rows}
    methods=list(style['methods']);assert len(methods)==5
    plotted=[]
    for metric,title,scale in PANELS:
        for method in methods:
            for snr in (4,10,19):
                r=lookup[method,metric,snr];assert int(r['source_count'])==500
                mean,lo,hi=[float(r[k])*scale for k in ('mean','ci_low','ci_high')];assert lo<=mean<=hi
                plotted.append({'method':method,'label':style['methods'][method]['label'],'metric':metric,'snr_db':snr,'mean':float(r['mean']),'ci_low':float(r['ci_low']),'ci_high':float(r['ci_high']),'display_scale':scale,'display_mean':mean,'display_ci_low':lo,'display_ci_high':hi,'sources':500,'noise_per_source':3})
    out.mkdir();write_csv(out/'figure2_data.csv',plotted)
    rc={'font.family':'DejaVu Sans','font.size':style['tick_label_pt'],'axes.titlesize':style['panel_title_pt'],'axes.labelsize':style['axis_label_pt'],'xtick.labelsize':style['tick_label_pt'],'ytick.labelsize':style['tick_label_pt'],'legend.fontsize':style['legend_pt'],'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','axes.facecolor':'white','figure.facecolor':'white'}
    with plt.rc_context(rc):
        for grayscale in (False,True):
            fig,axes=plt.subplots(2,2,figsize=(style['width_mm']/25.4,style['height_mm']/25.4))
            fig.subplots_adjust(left=.078,right=.984,bottom=.11,top=.835,wspace=.23,hspace=.62)
            handles=[];axis_bounds=[]
            for ax,(metric,title,scale)in zip(axes.flat,PANELS):
                values=[];handles_here=[]
                for order,method in enumerate(methods):
                    opt=style['methods'][method];color=gray(opt['color'])if grayscale else opt['color'];line=opt['linestyle'];line=(line[0],tuple(line[1]))if isinstance(line,list)else line
                    rr=[lookup[method,metric,snr]for snr in (4,10,19)];y=np.array([float(r['mean'])*scale for r in rr]);lo=np.array([float(r['ci_low'])*scale for r in rr]);hi=np.array([float(r['ci_high'])*scale for r in rr]);values.extend(lo);values.extend(hi)
                    ax.errorbar([4,10,19],y,yerr=[y-lo,hi-y],fmt='none',ecolor=color,elinewidth=.8,capsize=2.2,capthick=.8,zorder=2)
                    ax.plot([4,10,19],y,color=color,linestyle=line,linewidth=opt['linewidth'],zorder=3)
                    ax.plot([4,10,19],y,linestyle='none',marker=opt['marker'],markersize=opt['markersize'],markerfacecolor=color if opt['filled']else'white',markeredgecolor=color,markeredgewidth=.95,zorder=5+(method=='EC_VAR_PARTIAL'))
                    handles_here.append(Line2D([],[],color=color,linestyle=line,linewidth=opt['linewidth'],marker=opt['marker'],markersize=opt['markersize'],markerfacecolor=color if opt['filled']else'white',markeredgewidth=.95,label=opt['label']))
                if not handles:handles=handles_here
                low,high=min(values),max(values);pad=max((high-low)*.1,.02 if scale==1 else 1)
                low=max(0,low-pad);high=min(100,high+pad)if scale==100 else high+pad
                ax.set(xlim=(2.8,20.2),ylim=(low,high),xticks=[4,10,19],xlabel='SNR (dB)');ax.set_title(title,loc='left',pad=7)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.tick_params(direction='out',length=3,width=.7,pad=3)
                ax.grid(axis='y',color='.90',linewidth=.5,zorder=0);ax.set_axisbelow(True)
                axis_bounds.append({'metric':metric,'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim())})
            fig.legend(handles[:2],[h.get_label()for h in handles[:2]],loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,frameon=False,handlelength=2.7,columnspacing=2.5)
            fig.legend(handles[2:],[h.get_label()for h in handles[2:]],loc='upper center',bbox_to_anchor=(.53,.94),ncol=3,frameon=False,handlelength=2.7,columnspacing=2)
            fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
            for text in fig.findobj(matplotlib.text.Text):
                if not text.get_visible()or not text.get_text():continue
                box=text.get_window_extent(renderer)
                if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:clipped.append(text.get_text())
            assert not clipped,clipped
            stem='figure2_common500_grayscale'if grayscale else'figure2_common500'
            for ext in ('pdf','svg','png'):fig.savefig(out/(stem+'.'+ext),dpi=style['dpi'],bbox_inches=None)
            if not grayscale:put(out/'axis_bounds.json',axis_bounds)
            plt.close(fig)
    for path in out.glob('*.svg'):
        tree=ET.parse(path);assert not tree.findall('.//{http://www.w3.org/2000/svg}image')
        assert tree.findall('.//{http://www.w3.org/2000/svg}text')
    caption='Fig. 2. Five-method performance on the original common500 population at N = 1024 complex channel symbols. Points average three noise realizations within each source and then average 500 sources; error bars show pointwise 95% source-bootstrap intervals using the registered 10,000 × 500 index bank. Agreement is the fraction of images whose ConvNeXt top-1 prediction matches that of the source image. NeST-Com uses the T1-C m10-corrected entropy-coded partial-scale result at 19 dB; the continuous baselines use the fixed P100k and Swin200k checkpoints. ARPC uses its frozen N1024 policies and actual per-image payload fallback. All receive failures remain in the denominator.\n'
    (out/'caption.txt').write_text(caption,encoding='utf8')
    (out/'outline_changes.md').write_text('# Outline changes only\n\n- Fig. 2: replace four-method common500 with five-method common500 at N1024.\n- The separate N1024 ARPC external_common100 comparison table is superseded as main evidence by the new Fig. 2; retain its original result as historical provenance.\n- Retain the same-prior WHOLE/PARTIAL ablation and source-capacity utilization evidence.\n- N2048 remains the completed 100-source experiment. Distinguish adaptive endpoint fallback at 4/10 dB from complete-representation transmission at 19 dB.\n- No full manuscript rewrite, new training, new timing, or N2048 expansion.\n',encoding='utf8')
    put(out/'input_manifest.json',{'summary':desc(d/'common500_summary.csv'),'paired':desc(d/'nest_minus_arpc_paired.csv'),'bootstrap_protocol':desc(d/'statistics_protocol.json'),'style':desc(E/'figure_style.json'),'plot_source':desc(__file__),'template_textwidth_mm':style['width_mm'],'height_mm':style['height_mm'],'dpi':style['dpi'],'new_statistics':False,'uncertainty_read_from_saved_results':True,'display_transform':'Agreement fraction ×100; all other values unmodified','skill':'scientific-figure-making, local installed version; scientific-visualization cached reference audited','publisher_compliance':'Prepared at measured current template width; no claim of final submission-rule certification'})
    put(out/'completion.json',{'status':'RENDERED_REQUIRES_ACTUAL_VISUAL_REVIEW','outputs':[desc(p)for p in sorted(out.iterdir())if p.is_file()],'actual_visual_review':False,'new_scientific_calls':0})
if __name__=='__main__':main()
