"""Deterministic publication rendering of already closed actual A3 statistics."""
import argparse,csv,hashlib,json,os,pathlib,sys
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019');sys.path.insert(0,str(E))
from a1_owner import O,L,desc,verify,put
METHODS={
 'EC_VAR_PARTIAL':('NeST-Com (entropy-coded)','#3775BA','o','-',True),
 'ARPC_ADAPTED':('ARPC + LDPC–QAM (adapted)','#8064A2','P','--',False),
 'P100000':('Latent-space JSCC','#B64342','s','-',False),
 'Swin200000':('SwinJSCC (adapted)','#5E966B','^','-.',False),
 'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':('Adaptive BPG + LDPC','#767676','D',':',False)}
METRICS=[('psnr_db','PSNR (dB)',1,'higher'),('lpips_alex','LPIPS',1,'lower'),
 ('dreamsim','DreamSim',1,'lower'),('dists','DISTS',1,'lower'),
 ('dinov2_vitl14_cosine','DINO-L cosine',1,'higher'),('convnext_top1_source_prediction','Agreement (%)',100,'higher'),
 ('ms_ssim','MS-SSIM',1,'higher')]
def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-closure',required=True);a=p.parse_args()
    source=O/'a3_metrics_v1';review=source/'root_closure_review.json';assert desc(review)['sha256']==a.expected_closure
    closed=json.loads(review.read_text());assert closed['status']=='EVAL100_FINAL_ANALYSIS_CLOSED' and closed['all_processes_gone'] and closed['unresolved']==0
    for pin in closed['controls']+closed['outputs']:verify(pin)
    out=O/'final_figures_v1';assert not out.exists();out.mkdir();cache=L/'final_figures_v1_cache';assert not cache.exists();cache.mkdir()
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    with (source/'matched100_summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (source/'nest_minus_arpc_paired.csv').open() as f:paired=list(csv.DictReader(f))
    assert len(summary)==105 and len(paired)==21
    for row in summary+paired:assert int(row['source_count'])==100 and float(row['ci_low'])<=float(row['ci_high'])
    style={'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,
      'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
      'legend.fontsize':8,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'}
    plt.rcParams.update(style);put(out/'style.json',style);put(out/'method_mapping.json',METHODS)
    sizes=[];limits={}
    def base():
        fig,axes=plt.subplots(2,4,figsize=(180/25.4,102/25.4));fig.subplots_adjust(left=.07,right=.985,bottom=.105,top=.865,wspace=.47,hspace=.72)
        fig.suptitle('100-source matched comparison',fontsize=10,y=.975)
        for ax in axes.flat:
            ax.set_xlim(3,20);ax.set_xticks([4,10,19]);ax.set_xlabel('SNR (dB)',labelpad=2);ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(length=2.5,width=.65,pad=2)
        return fig,list(axes.flat)
    def save(fig,name):
        fig.canvas.draw()
        for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=600)
        sizes.append({'name':name,'width_mm':float(fig.get_figwidth()*25.4),'height_mm':float(fig.get_figheight()*25.4),'dpi':600,'vector_pdf_svg':True})
        plt.close(fig)
    # Exact numeric dB spacing, linear segments, CI bars below opaque markers.
    fig,axes=base();handles=[]
    for n,(metric,title,scale,direction) in enumerate(METRICS):
        ax=axes[n];values=[]
        ax.set_title(f'({chr(97+n)}) {title} '+('↑' if direction=='higher' else '↓'),loc='left',pad=5)
        for method,(label,color,marker,ls,filled) in METHODS.items():
            rows=sorted([r for r in summary if r['metric']==metric and r['method']==method],key=lambda r:int(r['snr_db']));assert len(rows)==3
            x=np.array([int(r['snr_db']) for r in rows]);y=np.array([float(r['mean'])*scale for r in rows]);lo=np.array([float(r['ci_low'])*scale for r in rows]);hi=np.array([float(r['ci_high'])*scale for r in rows])
            values.extend(lo);values.extend(hi);values.extend(y)
            ax.vlines(x,lo,hi,color=color,linewidth=.8,zorder=1);ax.hlines(lo,x-.18,x+.18,color=color,linewidth=.8,zorder=1);ax.hlines(hi,x-.18,x+.18,color=color,linewidth=.8,zorder=1)
            ax.plot(x,y,color=color,linestyle=ls,linewidth=1.55 if method=='EC_VAR_PARTIAL' else 1.25,marker=marker,markersize=4.5,markerfacecolor=color if filled else 'white',markeredgewidth=.85,zorder=3,label=label)
            if n==0:handles.append(Line2D([],[],color=color,linestyle=ls,marker=marker,markersize=4.5,markerfacecolor=color if filled else 'white',linewidth=1.4,label=label))
        margin=max(max(values)-min(values),.01)*.12;ax.set_ylim(min(values)-margin,max(values)+margin);limits[metric]=list(ax.get_ylim())
    legend_labels=['NeST-Com\n(entropy-coded)','ARPC + LDPC–QAM\n(adapted)','Latent-space JSCC','SwinJSCC (adapted)','Adaptive BPG + LDPC']
    axes[7].set_axis_off();axes[7].legend(handles=handles,labels=legend_labels,loc='center left',bbox_to_anchor=(-.08,.5),handlelength=1.7,labelspacing=.65,fontsize=7.2)
    save(fig,'matched100_seven_metrics')
    fig,axes=base();fig.suptitle('Paired source differences: NeST-Com − ARPC',fontsize=10,y=.975)
    for n,(metric,title,scale,direction) in enumerate(METRICS):
        ax=axes[n];rows=sorted([r for r in paired if r['metric']==metric],key=lambda r:int(r['snr_db']));assert len(rows)==3
        x=np.array([int(r['snr_db']) for r in rows]);y=np.array([float(r['mean'])*scale for r in rows]);lo=np.array([float(r['ci_low'])*scale for r in rows]);hi=np.array([float(r['ci_high'])*scale for r in rows])
        delta_title='Agreement (pp)' if scale==100 else ('DINO-L' if metric=='dinov2_vitl14_cosine' else title)
        ax.set_title(f'({chr(97+n)}) Δ{delta_title}',loc='left',pad=5)
        ax.axhline(0,color='#767676',linewidth=.75,linestyle='--',zorder=0)
        ax.vlines(x,lo,hi,color='#3775BA',linewidth=1,zorder=1);ax.hlines(lo,x-.2,x+.2,color='#3775BA',linewidth=1);ax.hlines(hi,x-.2,x+.2,color='#3775BA',linewidth=1)
        ax.plot(x,y,color='#3775BA',marker='o',markersize=4.5,linewidth=1.5,zorder=3)
        vals=[0,*lo,*hi,*y];margin=max(max(vals)-min(vals),.001)*.15;ax.set_ylim(min(vals)-margin,max(vals)+margin)
        ax.text(.02,.96,'NeST favorable: '+('below 0' if direction=='lower' else 'above 0'),transform=ax.transAxes,fontsize=6.8,va='top')
    axes[7].set_axis_off();axes[7].text(0,.6,'Direct paired 95% intervals\n100 sources × 3 noise realizations\n10,000 source resamples\nPointwise, unadjusted',fontsize=8,linespacing=1.6,va='center')
    save(fig,'matched100_paired_differences')
    # Numerical table uses the same means, without a second statistical pass.
    table=['\\begin{table*}[t]','\\centering','\\caption{100-source matched comparison. Each entry is a source-balanced mean after averaging three noise realizations. Full pointwise intervals are supplied with the results.}','\\label{tab:arpc-matched100}','\\begin{tabular}{llrrrrrrr}','\\toprule','SNR & Method & PSNR & LPIPS & DreamSim & DISTS & DINO-L & Agr. (\\%) & MS-SSIM \\\\','\\midrule']
    for snr in [4,10,19]:
        for method,(label,*_) in METHODS.items():
            vals=[]
            for metric,_,scale,_ in METRICS:
                row=next(r for r in summary if r['method']==method and r['metric']==metric and int(r['snr_db'])==snr)
                vals.append(f"{float(row['mean'])*scale:.2f}" if scale==100 or metric=='psnr_db' else f"{float(row['mean']):.4f}")
            table.append(f'{snr} dB & '+label.replace('–','--')+' & '+' & '.join(vals)+' \\\\')
        if snr!=19:table.append('\\midrule')
    table+=['\\bottomrule','\\end{tabular}','\\end{table*}'];(out/'matched100_table.tex').write_text('\n'.join(table)+'\n')
    caption='''100-source matched comparison. The fixed external_common100_v1 subset is drawn from the original common500 population. All methods use the same100 source IDs; each source averages three noise realizations. Error bars are pointwise95% source-bootstrap percentile intervals (10000 resamples, seed2026100701). NeST-Com19dB uses the revised T1-C m10 policy; Latent-space JSCC and SwinJSCC use their final100k and200k checkpoints. Agreement is the fraction of images whose ConvNeXt top-1 prediction matches that of the source image. ARPC uses the official adapted weights with the paid caption/container and full N1024 frame, including failure cases.

Paired source differences. Each difference is NeST-Com (entropy-coded) minus ARPC before the same source resampling. Intervals are directly paired, pointwise and unadjusted. Negative differences favor NeST-Com for LPIPS, DreamSim and DISTS; positive differences favor it for PSNR, DINO-L, agreement and MS-SSIM. Agreement differences are percentage points. Historical baseline noise seeds differ, so pairing is by source. This100-source comparison is not the full500-source result or a fresh blind test.
'''
    (out/'captions.txt').write_text(caption)
    manifest={'status':'RENDERED_PENDING_ACTUAL_VISUAL_REVIEW','data':[desc(source/n) for n in ['root_closure_review.json','matched100_summary.csv','nest_minus_arpc_paired.csv','bootstrap_identity.json']],
      'source':desc(pathlib.Path(__file__)),'sizes':sizes,'limits':limits,'new_scoring':0,'new_bootstrap':0,
      'files':[desc(p) for p in sorted(out.iterdir()) if p.is_file()]}
    put(out/'manifest.json',manifest)
if __name__=='__main__':main()
