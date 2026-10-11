"""Read closed diagnostic/statistical tables and export publication figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/baseline_strength_20261011'
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
LABELS=['PSNR (dB)', 'LPIPS', 'DINOv2-L cosine similarity', 'ConvNeXt agreement (%)']
SNRS=[1,4,7,10,13,19]
COLORS=['#000000','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
MARKERS=['o','s','^','v','D','x']
METHODS={
    'P40000':('P1024, 40k','#0072B2','--','s'),
    'P100000':('P1024, continued 100k','#0072B2','-','o'),
    'Swin80000':('SwinJSCC, 80k (adapted)','#D55E00','--','^'),
    'Swin200000':('SwinJSCC, continued 200k (adapted)','#D55E00','-','D'),
    'RAW_PARTIAL':('Proposed (raw + partial + VAR)','#009E73','-.','v'),
    'EC_VAR_PARTIAL':('Entropy-coded partial + VAR','#CC79A7','-.','v')}


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def save(fig,out,name):
    for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=600,facecolor='white')
    plt.close(fig)


def panels():
    fig,axes=plt.subplots(2,2,figsize=(7,5.2))
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.83,wspace=.32,hspace=.40)
    return fig,axes.flatten()


def legend(fig,ax,ncol=2):
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=ncol,frameon=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['diagnostics','final']);args=parser.parse_args()
    out=BASE/('diagnostic_figures_v1' if args.mode=='diagnostics' else 'final_comparison_figures_v1');out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':8.5,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.grid':True,'grid.alpha':.25,'axes.spines.top':False,'axes.spines.right':False,
                         'lines.linewidth':1.2})
    bindings={};used=[];captions=[]
    if args.mode=='diagnostics':
        folder=BASE/'automatic_followthrough_v1'
        for stage in ['milestones','bottleneck']:
            wait=read(folder/(stage+'_actual_wait.json'));assert wait['actual_child_waited'] and wait['exit_code']==0
            done=read(folder/stage/'completion.json');assert done['status']=='COMPLETE_FROZEN_FRAME_SCORING'
            p=folder/stage/'descriptive_summary.csv';assert sha(p)==done['outputs'][str(p)];bindings[str(p)]=sha(p)
        data=rows(folder/'milestones/descriptive_summary.csv');used.extend(dict(population='calibration1000',**r) for r in data)
        for branch,steps in [('P',[40000,60000,80000,100000]),('Swin',[80000,120000,160000,200000])]:
            fig,axes=panels()
            for ax,metric,label in zip(axes,METRICS,LABELS):
                scale=100 if metric==METRICS[-1] else 1
                for j,s in enumerate(SNRS):
                    curve=[next(r for r in data if r['method']==branch+str(step) and int(r['snr_db'])==s and r['metric']==metric) for step in steps]
                    assert all(int(r['source_count'])==1000 and int(r['noise_count'])==(3 if branch=='P' else 1) for r in curve)
                    ax.plot(np.array(steps)/1000,[float(r['mean'])*scale for r in curve],color=COLORS[j],marker=MARKERS[j],markersize=3,
                            linestyle=['-','--','-.',':','-','--'][j],label=f'{s} dB')
                ax.set(xlabel='Total updates (thousands)',ylabel=label,xticks=np.array(steps)/1000)
            legend(fig,axes[0],3);save(fig,out,branch+'_milestone_metrics')
        data=rows(folder/'bottleneck/descriptive_summary.csv');used.extend(dict(population='calibration_first100',**r) for r in data)
        order=['F_direct_Dc_free_latent_reference','P40000_zero_noise','P40000_19dB','P100000_zero_noise','P100000_19dB']
        names=['Full latent\nreference','40k\nno noise','40k\n19 dB','100k\nno noise','100k\n19 dB']
        fig,axes=panels()
        for ax,metric,label in zip(axes,METRICS,LABELS):
            values=[float(next(r for r in data if r['method']==m and r['metric']==metric)['mean'])*(100 if metric==METRICS[-1] else 1) for m in order]
            ax.plot(range(5),values,'o',color='#0072B2',markersize=4)
            ax.set(xticks=range(5),xticklabels=names,ylabel=label);ax.tick_params(axis='x',labelsize=7)
        fig.subplots_adjust(top=.97,hspace=.45);save(fig,out,'P_bottleneck_diagnostic')
        captions=[r'\paragraph{Milestone diagnostics.} Four frozen metrics on all 1,000 original calibration sources at N1024. P1024 uses three original noise replicates; SwinJSCC uses its original one-noise calibration protocol. Curves show descriptive means, with no bootstrap intervals and no feedback into model selection or stopping. LPIPS is lower-is-better; the other three metrics are higher-is-better. ConvNeXt agreement measures consistency with the source-image prediction, not classification accuracy. The original SwinJSCC checkpoint at 80k was trained/calibrated over 1--13 dB; its 19 dB evaluation is outside that range.',
                  r'\paragraph{Bottleneck diagnostic.} Original calibration first 100 sources. The full-latent reference feeds the original latent directly to the same frozen decoder and is not an equal-bandwidth achievable upper bound. Both no-noise paths retain the real N1024 communication bottleneck and receiver conditioning at 19 dB. No-noise paths have one deterministic output per source; 19 dB paths average the original three noise replicates. All points are retained without confidence intervals.']
    else:
        stats=BASE/'paired_supplement_v1';assert read(stats/'completion.json')['status']=='COMPLETE_BOTH_SOURCE_PAIRED_SUPPLEMENTS'
        for population in ['common500','new100']:
            folder=stats/population;done=read(folder/'completion.json');assert done['status']=='COMPLETE_SOURCE_PAIRED_SUPPLEMENT'
            for name,h in done['outputs'].items():assert sha(folder/name)==h
            data=rows(folder/'summary.csv')+read(folder/'original_summary_reused.json');pairs=rows(folder/'paired.csv')
            for n in ['summary.csv','paired.csv','original_summary_reused.json']:bindings[str(folder/n)]=sha(folder/n)
            methods=['RAW_PARTIAL','P40000','P100000','Swin80000','Swin200000'] if population=='common500' else ['EC_VAR_PARTIAL','P100000','Swin200000']
            snrs=SNRS if population=='common500' else [4,10,19]
            fig,axes=panels()
            for ax,metric,label in zip(axes,METRICS,LABELS):
                scale=100 if metric==METRICS[-1] else 1
                for method in methods:
                    curve=sorted([r for r in data if r['method']==method and r['metric']==metric],key=lambda r:int(r['snr_db']))
                    assert [int(r['snr_db']) for r in curve]==snrs
                    y,lo,hi=[np.array([float(r[k])*scale for r in curve]) for k in ['mean','ci_low','ci_high']]
                    assert np.all(lo<=y) and np.all(y<=hi)
                    name,color,style,marker=METHODS[method]
                    if method=='Swin80000':
                        ax.plot(snrs[:-1],y[:-1],color=color,linestyle=style,marker=marker,markersize=3,label=name)
                        ax.plot(snrs[-2:],y[-2:],color=color,linestyle='--');ax.plot([19],[y[-1]],color=color,marker=marker,markerfacecolor='white',markersize=4,linestyle='none')
                    else:ax.plot(snrs,y,color=color,linestyle=style,marker=marker,markersize=3,label=name)
                    ax.fill_between(snrs,lo,hi,color=color,alpha=.11)
                    if metric==METRICS[0]:used.extend(dict(population=population,kind='single_method',**r) for r in data if r['method']==method)
                ax.set(xlabel='SNR (dB)',ylabel=label,xticks=snrs);ax.margins(y=.1)
            legend(fig,axes[0],2 if population=='common500' else 1);save(fig,out,population+'_new_old_baselines')
            keys=list(dict.fromkeys((r['method'],r['reference']) for r in pairs));fig,axes=panels()
            for ax,metric,label in zip(axes,METRICS,LABELS):
                scale=100 if metric==METRICS[-1] else 1
                for j,(method,reference) in enumerate(keys):
                    curve=sorted([r for r in pairs if (r['method'],r['reference'],r['metric'])==(method,reference,metric)],key=lambda r:int(r['snr_db']))
                    assert [int(r['snr_db']) for r in curve]==snrs and all(r['delta_definition']=='method - reference' for r in curve)
                    y,lo,hi=[np.array([float(r[k])*scale for r in curve]) for k in ['mean','ci_low','ci_high']]
                    label_pair=METHODS[method][0]+' minus '+METHODS[reference][0]
                    ax.errorbar(snrs,y,yerr=np.array([y-lo,hi-y]),color=COLORS[j+1],linestyle=['-','--','-.',':'][j],marker=MARKERS[j],markersize=3,capsize=2,label=label_pair)
                ax.axhline(0,color='.4',linewidth=.7)
                ax.set(xlabel='SNR (dB)',ylabel=('Delta ConvNeXt agreement (pp)' if metric==METRICS[-1] else 'Delta '+label),xticks=snrs);ax.margins(y=.12)
            legend(fig,axes[0],1);save(fig,out,population+'_paired_deltas')
            used.extend(dict(population=population,kind='paired_delta',**r) for r in pairs)
        captions=[r'\paragraph{Frozen baseline supplement.} N1024 comparison on the original common500 sources and, separately, the previously published new100 sources. Every point averages three registered noise replicates per source. Shading shows pointwise 95\% source bootstrap intervals. Original method intervals are reused unchanged; only new method intervals and required source-paired contrasts are computed with the original 10,000-replicate implementation and seed 2026100701. These are subsequent baseline comparisons on existing evaluation cohorts, not newly blind tests. All failed frames remain in the denominators.',
                  r'\paragraph{Paired contrasts.} Deltas are method minus reference after averaging each source over its three noise replicates; error bars are the directly computed source-paired 95\% intervals. LPIPS retains its original sign, so negative deltas are better. ConvNeXt agreement and its intervals are shown as percentages in the main curves and percentage points in contrasts, not relative gains or classification accuracy. The common500 and new100 populations are never combined or subtracted from one another.',
                  r'\paragraph{SwinJSCC qualification.} The 80k adapted baseline at 19 dB is outside its training and calibration range, shown with an open marker and a dashed final segment. The continued adapted model uses the authorized weights-only initialization with fresh optimizer, sampling and random states, and expanded 1--19 dB training coverage. It reaches its 120k additional-update budget while still improving; neither official optimality nor convergence is claimed. P1024 resumes its full original training state and meets the registered empirical plateau rule at 100k.']
    fields=list(dict.fromkeys(k for r in used for k in r))
    with (out/'plot_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(used)
    (out/'captions.tex').write_text('\n\n'.join(captions)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(status='FIGURES_GENERATED_VISUAL_REVIEW_REQUIRED',input_sha256=bindings,
        bootstrap_calls=0,inference_calls=0,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json'}),indent=2)+'\n')


if __name__=='__main__':main()
