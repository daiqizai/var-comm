#!/usr/bin/env python3
"""Render existing within-budget paired CIs only after actual T6 statistics close."""
import argparse
from decimal import Decimal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_t5_entropy_quality as g

OLD_COMMIT='252176e041758ecb2d3e81b6fde5b587e7e17bb7'
OLD_PAIRED_SHA='dd8221b1fe8aa9d86f9cbf78ea281fd2da860e2ffdcd6fb83daa487eebc0c235'
OLD_PAIRS_SHA='524f95dfec6278bdfe9b8fc03765bd2f678854816e1ad2e73a38bb854a485468'
EVALUATOR='8313f062d2435be361add61aceb9848ccb8844c1f3c9c4743f662480057489e2'
METRICS=g.METRICS
BUDGETS=((1024,(1,4,7,10,13,19),500,'N = 1024 · original 500 sources','#0072B2','-','o'),
         (2048,(4,10,19),100,'N = 2048 · new 100 sources','#D55E00','--','s'))


def one_output(done,name,folder):
    expected=g.one([h for p,h in done['outputs'].items() if p.replace('\\','/').rsplit('/',1)[-1]==name])
    path=folder/name;g.require(path.is_file()and g.sha(path)==expected,'Missing or unsealed actual statistics file: '+str(path))
    return path,expected


def select_pairs(rows,N,pairs):
    snrs=(1,4,7,10,13,19)if N==1024 else(4,10,19)
    prefix='RAW64_'if N==1024 else'N2048_RAW_'
    count=500 if N==1024 else 100
    definitions={(p['method'],p['reference'])for p in pairs}
    require_keys=[(r['method'],r['reference'],int(r['snr_db']),r['metric'])for r in rows]
    g.require(len(set(require_keys))==len(require_keys),'Duplicate paired CSV identities')
    result=[]
    for metric,_,_,_,scale,_ in METRICS:
        for snr in snrs:
            method=prefix+'PARTIAL_VAR_COMPLETION_SNR_'+str(snr)
            reference=prefix+'WHOLE_VAR_COMPLETION_SNR_'+str(snr)
            r=g.one([r for r in rows if r['method']==method and r['reference']==reference
                     and int(r['snr_db'])==snr and r['metric']==metric])
            g.require((method,reference)in definitions and r['delta_definition']=='method minus reference',
                      'Exact within-budget partial-minus-complete direction required')
            g.require((int(r['source_count']),int(r['noise_count']),int(r['frame_count']))==(count,3,count*3)
                      and int(r['bootstrap_replicates'])==10000 and int(r['bootstrap_seed'])==2026100701,
                      'Paired CI population/statistics protocol differs')
            g.require(all(Decimal(r[k]).is_finite()for k in('mean','ci_low','ci_high'))
                      and Decimal(r['ci_low'])<=Decimal(r['mean'])<=Decimal(r['ci_high']),'Invalid stored paired CI')
            if N==2048:g.require(int(r['N'])==2048,'Other budget data cannot be relabelled N2048')
            result.append(dict(r,N=N,figure='within_budget_partial_minus_complete',
                population='original500 holdout'if N==1024 else'registered new100 confirmation',
                source_csv='original500/paired.csv'if N==1024 else'actual_T6/paired.csv',
                display_scale=scale,display_unit='percentage_points'if scale==100 else'dB'if metric=='psnr_db'else'dimensionless',
                **{f'plot_{k}':str(Decimal(r[k])*scale)for k in('mean','ci_low','ci_high')}))
    if N==1024:
        g.require(all(Decimal(r[k])==0 for r in result if int(r['snr_db'])==7 for k in('mean','ci_low','ci_high')),
                  'Original7dB exact-zero comparison must be retained')
    return result


def gather(args):
    old=args.original_paired.resolve();oldpairs=args.original_pairs.resolve();new=args.new_completion.resolve()
    g.require(len(args.new_completion_sha)==64 and g.sha(new)==args.new_completion_sha.lower(),
              'Explicit SHA of actual closed T6 statistics required')
    g.require(g.sha(old)==OLD_PAIRED_SHA and g.sha(oldpairs)==OLD_PAIRS_SHA,'Original published paired data changed')
    done=g.read(new)
    g.require(done['status']=='T6_N2048_CONFIRMATION100_SOURCE_PAIRED_STATISTICS_COMPLETE'
              and done['N']==2048 and done['source_count']==100 and done['noise_count']==3 and done['frame_count']==2700
              and done['snrs']==[4,10,19]and done['noise_seeds']==[9201,9202,9203]
              and len(done['source_ids'])==len(set(done['source_ids']))==100
              and done['summary_rows']==done['paired_rows']==36 and done['source_mean_rows']==3600
              and done['source_mean_first']is True and done['selection_used_confirmation']is False
              and done['holdout_used_for_selection']is False and done['metric_evaluator_identity']==EVALUATOR
              and done['bootstrap_seed']==2026100701 and done['bootstrap_replicates']==10000
              and done['multiple_comparison_adjustment']is False and done['old_statistics_file_unchanged']is True,
              'Only actual completed frozen N2048 confirmation100 statistics are admitted')
    inputs={str(old):OLD_PAIRED_SHA,str(oldpairs):OLD_PAIRS_SHA,str(new):g.sha(new)}
    tables={}
    for name in('paired.csv','summary.csv','points.json','pairs.json'):
        p,h=one_output(done,name,new.parent);inputs[str(p)]=h;tables[name]=g.read(p)if name.endswith('json')else g.readcsv(p)
    g.require(len(tables['paired.csv'])==len(tables['summary.csv'])==36 and len(tables['points.json'])==9
              and len(tables['pairs.json'])==9,'All actual methods and paired comparisons must remain in source artifacts')
    expected_methods={'N2048_RAW_WHOLE_VAR_COMPLETION','N2048_RAW_PARTIAL_VAR_COMPLETION',
                      'N2048_'+done['family']+'_VAR_COMPLETION'}
    g.require(done['family']in('EC_STATIC_WHOLE','EC_VAR_WHOLE')and set(done['methods'])==expected_methods,'Frozen T6 method identity differs')
    for p,identity in tables['points.json'].items():
        g.require(identity['N']==2048 and identity['source_count']==100 and identity['noise_count']==3
                  and identity['snr_db']in(4,10,19)and identity['method_id']in expected_methods
                  and p==identity['method_id']+'_SNR_'+str(identity['snr_db']),'Misidentified T6 point metadata')
    plotted=select_pairs(g.readcsv(old),1024,g.read(oldpairs))+select_pairs(tables['paired.csv'],2048,tables['pairs.json'])
    g.require(len(plotted)==36,'Exactly24 original plus12 N2048 paired metric points required')
    return plotted,inputs,done


def panel(ax,rows,metric):
    key,title,_,ylabel,_,_=metric;extent=[0]
    for N,snrs,count,label,color,ls,marker in BUDGETS:
        chosen=[g.one([r for r in rows if int(r['N'])==N and r['metric']==key and int(r['snr_db'])==snr])for snr in snrs]
        y,lo,hi=[np.asarray([float(r['plot_'+k])for r in chosen])for k in('mean','ci_low','ci_high')]
        ax.errorbar(snrs,y,yerr=[y-lo,hi-y],color=color,ls=ls,marker=marker,markerfacecolor='white'if N==2048 else color,
                    ms=4.3,lw=1.25,capsize=2.8,elinewidth=.9,zorder=3)
        extent.extend(lo);extent.extend(hi)
    low,high=min(extent),max(extent);pad=max((high-low)*.12,1e-5)
    ax.set_ylim(low-pad,high+pad);ax.set_xlim(.2,19.8);ax.set_xticks([1,4,7,10,13,19])
    ax.axhline(0,color='#777777',lw=.8,ls=':',zorder=1)
    ax.set_xlabel('SNR (dB)');ax.set_ylabel(ylabel);ax.set_title(title,pad=7)
    ax.grid(True);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)


def figures(rows,out,engineering=False):
    handles=[Line2D([],[],color=c,ls=ls,marker=m,markerfacecolor='white'if N==2048 else c,
                    label=label,ms=4,lw=1.25)for N,snrs,count,label,c,ls,m in BUDGETS]
    stem='fig_t6_cross_budget_partial_minus_complete'
    fig,axes=plt.subplots(2,2,figsize=(7,5.3))
    for ax,metric in zip(axes.flat,METRICS):panel(ax,rows,metric)
    fig.suptitle('Partial-scale minus complete-scale raw transmission + VAR',y=.985,fontsize=9.5)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.951),ncol=1,frameon=False,labelspacing=.4)
    fig.subplots_adjust(left=.113,right=.985,bottom=.125,top=.795,wspace=.4,hspace=.59)
    fig.text(.5,.025,'Within-budget source-paired differences · Different source populations · Existing 95% CIs',ha='center',fontsize=7.2)
    if engineering:fig.text(.5,.51,'SYNTHETIC LAYOUT ONLY',ha='center',fontsize=20,color='gray',alpha=.7)
    g.save(fig,out,stem);plt.close(fig)
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(3.5,3.3));panel(ax,rows,metric)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.993),ncol=1,frameon=False,fontsize=7.2)
        fig.subplots_adjust(left=.2,right=.97,bottom=.24,top=.72)
        fig.text(.53,.055,'Partial-scale minus complete-scale raw + VAR\nDifferent source populations; existing 95% CIs',ha='center',fontsize=7)
        if engineering:fig.text(.5,.51,'SYNTHETIC ONLY',ha='center',fontsize=14,color='gray',alpha=.7)
        g.save(fig,out,stem+'_'+metric[-1]);plt.close(fig)


CAPTION=r'''\newcommand{\TsixCrossBudgetCaption}{Within-budget source-paired differences between partial-scale and complete-scale raw-token transmission, both with the same frozen VAR completion model.
The $N=1024$ curve reproduces the original published500-source holdout at1,4,7,10,13 and19~dB; the $N=2048$ curve uses the separately registered100-source confirmation set at4,10 and19~dB.
These are different source populations, not paired measurements of a bandwidth change on the same images. Each source is averaged over three registered noises before source-level inference.
Markers and error bars reproduce the existing paired means and pointwise95\% source-bootstrap intervals (10,000 draws, seed2026100701), without multiple-comparison adjustment; this plot performs no new bootstrap.
Differences always mean partial-scale minus complete-scale. Positive PSNR, DINOv2-L and ConvNeXt agreement differences favor partial-scale transmission; negative LPIPS differences favor it.
Agreement differences are percentage points, not relative percentages or classification accuracy. The zero line, negative results, the original7~dB exact zero,13~dB small increment and all measured new-budget zero increments are retained.
Within each budget both granularities have the registered modulation/coding permissions and partial-scale transmission may fall back to a complete-scale configuration; $K>0$ is not forced at every SNR. Failed outputs are retained.
N2048 policies were frozen on calibration before confirmation outcomes. This figure compares the within-budget mechanism at two tested budgets; it does not estimate a cross-budget causal effect, interpolate an equal-quality crossing or infer a percentage bandwidth saving.}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('original-paired','original-pairs','new-completion','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--new-completion-sha',required=True)
    a=p.parse_args();g.require(not a.out.exists(),'Fresh independent output directory required')
    rows,inputs,done=gather(a);a.out.mkdir(parents=True);g.style();figures(rows,a.out)
    g.csvout(a.out/'plot_data.csv',rows);(a.out/'captions.tex').write_text(CAPTION,encoding='utf-8')
    (a.out/'README.md').write_text('# Cross-budget within-budget mechanism comparison\n\n'
        'This is a rendering of existing paired intervals, not new inference, PHY, scoring or bootstrap. '
        'N1024 uses the original500 sources at all six SNRs; N2048 uses the registered new100 confirmation sources at4/10/19dB. '
        'Both contrasts are partial-scale raw minus complete-scale raw, including all failure outcomes. '
        'The two curves do not report raw absolute quality or a same-source causal budget effect. LPIPS signs are preserved; agreement deltas are percentage points.\n\n'
        'All5 figures have vector PDF/editable SVG and600dpi PNG exports. The source CSV precision is preserved in plot_data.csv. '
        'No zeros, negative means or confidence limits are discarded; real numeric SNR spacing is used. '
        'Only two budgets were evaluated; no bandwidth-saving percentage or equal-quality crossing is inferred. '
        'New statistics admission requires the explicit actual completion SHA and verifies its sealed paired/summary/point files,100-source/3-noise/2700-frame closure, frozen evaluator and selection flags.\n\n'
        'Reproduce into a fresh directory:\n\n```text\n'
        'python experiments/wcl-evidence-closure-20261009/scripts/plot_t6_cross_budget.py '
        f'--original-paired "{a.original_paired}" --original-pairs "{a.original_pairs}" '
        f'--new-completion "{a.new_completion}" --new-completion-sha {a.new_completion_sha} --out "<fresh-output>"\n```\n',encoding='utf-8')
    g.write(a.out/'completion.json',dict(status='T6_CROSS_BUDGET_EXISTING_PAIRED_CI_PLOTS_COMPLETE',plot_rows=36,
        original_published_commit=OLD_COMMIT,budgets=[1024,2048],source_counts=[500,100],noise_count=3,
        N2048_actual_statistics_sha256=g.sha(a.new_completion),new_N2048_frame_count=2700,
        paired_direction='partial-scale raw minus complete-scale raw',different_source_populations=True,
        same_source_budget_effect_claimed=False,bandwidth_saving_inferred=False,lpips_sign_unchanged=True,
        agreement_display='percentage points',code_sha256=g.sha(__file__),plot_helper_sha256=g.sha(g.__file__),
        input_bindings=inputs,new_model_calls=0,new_packet_calls=0,new_bootstrap_calls=0,
        outputs={p.name:g.sha(p)for p in sorted(a.out.iterdir())if p.is_file()}))
    print('T6_CROSS_BUDGET_EXISTING_PAIRED_CI_PLOTS_COMPLETE')


if __name__=='__main__':main()
