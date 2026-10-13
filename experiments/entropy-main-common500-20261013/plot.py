"""Render completed EMAIN tables only; no resampling, inference or smoothing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyze import METRICS, NEW, SNRS, rows, read, require, sha, csv_write, write

STYLE = {
    NEW: ('NeST-Com (entropy-coded)', '#6A3D9A', 'D', '-'),
    'RAW_PARTIAL': ('NeST-Com (raw)', '#0072B2', 'o', '-'),
    'P100000': ('Latent-space JSCC', '#D55E00', 's', '--'),
    'Swin200000': ('SwinJSCC (adapted)', '#009E73', '^', '-.'),
    'BPG_ADAPTIVE': ('Adaptive-resolution BPG + LDPC', '#CC79A7', 'v', (0, (4, 1, 1, 1))),
    'RAW_WHOLE': ('Whole-scale prefix + VAR (raw)', '#666666', 'x', ':'),
    'EC_VAR_WHOLE': ('Whole-scale prefix + VAR (entropy-coded)', '#272727', '+', '--'),
}
MAIN = (NEW, 'RAW_PARTIAL', 'P100000', 'Swin200000', 'BPG_ADAPTIVE')
MECHANISM = ('RAW_WHOLE', 'RAW_PARTIAL', 'EC_VAR_WHOLE', NEW)
LABELS = ('PSNR (dB) ↑', 'LPIPS ↓', 'DINOv2-L cosine similarity ↑', 'ConvNeXt prediction agreement (%) ↑')
DELTAS = ('ΔPSNR (dB)\nPositive is better', 'ΔLPIPS\nNegative is better',
          'ΔDINOv2-L\nPositive is better', 'ΔConvNeXt agreement (pp)\nPositive is better')


def setup():
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans', 'Arial', 'Helvetica'],
                        'font.size':8.5, 'axes.labelsize':8.5, 'axes.titlesize':9, 'legend.fontsize':7.5,
                        'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.linewidth':0.65,
                        'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False,
                        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
                        'savefig.facecolor':'white', 'figure.facecolor':'white'})


def axes_style(ax, snr=True):
    ax.grid(axis='y', color='#dddddd', linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6)
    if snr:
        ax.set_xticks(SNRS)
        ax.set_xlim(3, 20)
        ax.set_xlabel('SNR (dB)')


def limits(ax, lows, highs, zero=False):
    lo, hi = min(lows), max(highs)
    if zero:
        lo, hi = min(0, lo), max(0, hi)
    pad = max((hi-lo)*.08, 1e-6)
    ax.set_ylim(lo-pad, hi+pad)


def export(fig, out, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    texts = list(fig.texts)
    for legend in fig.legends:
        texts.extend(legend.get_texts())
    for ax in fig.axes:
        texts.extend((ax.title, ax.xaxis.label, ax.yaxis.label))
        for axis in (ax.xaxis, ax.yaxis):
            lower, upper = sorted(axis.get_view_interval())
            texts.append(axis.get_offset_text())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                # Matplotlib retains invisible out-of-view tick Text objects.
                if lower <= tick.get_loc() <= upper:
                    texts.extend((tick.label1, tick.label2))
    for obj in texts:
        if obj.get_visible() and obj.get_text():
            box = obj.get_window_extent(renderer)
            require(box.x0 >= page.x0-1 and box.y0 >= page.y0-1 and box.x1 <= page.x1+1 and box.y1 <= page.y1+1,
                    'Clipped text in '+name+': '+obj.get_text())
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out/(name+'.'+ext), dpi=600)
    require(b'/Subtype /Image' not in (out/(name+'.pdf')).read_bytes(), 'PDF unexpectedly contains raster artwork')
    svg = (out/(name+'.svg')).read_text()
    require('<text' in svg and '<image' not in svg, 'Editable vector SVG required')
    plt.close(fig)


def metric_panels(data, methods, out, name, paired=False):
    fig, axs = plt.subplots(2, 2, figsize=(7, 5.3))
    fig.subplots_adjust(left=.11, right=.983, bottom=.11, top=.79, wspace=.43, hspace=.66)
    used = []
    for j, (metric, ax) in enumerate(zip(METRICS, axs.flat)):
        scale = 100 if metric == METRICS[-1] else 1
        low, high = [], []
        for method in methods:
            selected = sorted([r for r in data if r['method']==method and r['metric']==metric], key=lambda r:int(r['snr_db']))
            require([int(r['snr_db']) for r in selected] == list(SNRS), 'Incomplete/duplicate plot curve')
            y, a, b = [np.asarray([float(r[k])*scale for r in selected]) for k in ('mean', 'ci_low', 'ci_high')]
            label, color, marker, line = STYLE[method]
            ax.errorbar(SNRS, y, yerr=np.asarray([y-a,b-y]), fmt='none', ecolor=color,
                        elinewidth=.75, capsize=2.0, capthick=.75, alpha=.70, zorder=1)
            low.extend(a); high.extend(b)
            used.extend(dict(figure=name, **r, plotted_mean=yy, plotted_ci_low=aa, plotted_ci_high=bb,
                             plotted_unit=('percentage_points' if paired else 'percent') if scale==100 else 'original_units')
                        for r, yy, aa, bb in zip(selected, y, a, b))
        for method in methods:
            selected = sorted([r for r in data if r['method']==method and r['metric']==metric], key=lambda r:int(r['snr_db']))
            label, color, marker, line = STYLE[method]
            ax.plot(SNRS, [float(r['mean'])*scale for r in selected], color=color, marker=marker,
                    linestyle=line, linewidth=1.35, markersize=4, zorder=3)
        if paired:
            ax.axhline(0, color='#777777', linewidth=.8, linestyle=':', zorder=0)
        limits(ax, low, high, zero=paired)
        axes_style(ax)
        ax.set_title('('+chr(97+j)+') '+(DELTAS[j] if paired else LABELS[j]), pad=7, fontsize=8.6 if paired else 9)
    handles = [Line2D([], [], color=STYLE[m][1], marker=STYLE[m][2], linestyle=STYLE[m][3], linewidth=1.4,
                      markersize=4, label=STYLE[m][0]) for m in methods]
    if paired:
        fig.text(.5, .955, 'Partial-scale minus whole-scale prefix (entropy-coded)', ha='center', fontsize=9)
    else:
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.985), ncol=2,
                   handlelength=2.8, columnspacing=1.4, labelspacing=.6)
    export(fig, out, name)
    return used


def prefix_panels(data, out):
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.85))
    fig.subplots_adjust(left=.08, right=.985, top=.79, bottom=.21, wspace=.32)
    methods = ('EC_VAR_WHOLE', NEW)
    for snr, ax in zip(SNRS, axs):
        for method in methods:
            selected = sorted([r for r in data if r['method']==method and int(r['snr_db'])==snr], key=lambda r:int(r['actual_T']))
            label, color, marker, line = STYLE[method]
            x, y = zip(*[(int(r['actual_T']),100*float(r['source_fraction'])) for r in selected])
            ax.vlines(x, 0, y, color=color, linewidth=1.1, alpha=.8, linestyles='dashed' if method!=NEW else 'solid')
            ax.plot(x, y, linestyle='none', marker=marker, markersize=4, color=color)
        ax.set_title(str(snr)+' dB')
        ax.set_xlabel('Actually encoded tokens, T')
        ax.set_ylim(0, 105)
        axes_style(ax, snr=False)
    axs[0].set_ylabel('Sources (%)')
    handles = [Line2D([],[], color=STYLE[m][1], marker=STYLE[m][2], linestyle=STYLE[m][3], label=STYLE[m][0]) for m in methods]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,1), ncol=1)
    export(fig,out,'fig_prefix_endpoint_distribution')


def length_panels(data, out):
    fig, axs = plt.subplots(2,3,figsize=(7,5.2))
    fig.subplots_adjust(left=.09,right=.982,top=.90,bottom=.22,wspace=.33,hspace=.64)
    for j, snr in enumerate(SNRS):
        ax = axs[0,j]
        group = [r for r in data if r['method']==NEW and int(r['snr_db'])==snr and r['length_status']=='ACTUAL_TERMINATED_STREAM']
        for field, color, line, label in [('raw12_bits','#777777','--','12T: same token sequence'),
                                        ('arithmetic_bits',STYLE[NEW][1],'-','Terminated arithmetic stream')]:
            x=np.sort([float(r[field]) for r in group]);y=np.arange(1,len(x)+1)/500
            ax.step(x,y,where='post',color=color,linestyle=line,linewidth=1.2,label=label)
        ax.set_title(str(snr)+' dB')
        ax.set_xlabel('Source length (bits)')
        axes_style(ax,False)
        ax.set_ylim(0,1.04)
        ax=axs[1,j]
        for method in ('EC_VAR_WHOLE',NEW):
            group=[r for r in data if r['method']==method and int(r['snr_db'])==snr and r['source_capacity_utilization']!='']
            x=np.sort([float(r['source_capacity_utilization']) for r in group]);y=np.arange(1,len(x)+1)/500
            ax.step(x,y,where='post',color=STYLE[method][1],linestyle=STYLE[method][3],linewidth=1.2,label=STYLE[method][0])
        ax.set_xlabel('Source-capacity utilization')
        ax.set_xlim(0,1.02);ax.set_ylim(0,1.04)
        axes_style(ax,False)
    axs[0,0].set_ylabel('Fraction of all 500 sources')
    axs[1,0].set_ylabel('Fraction of all 500 sources')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=2,fontsize=7.3)
    handles,labels=axs[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.02),ncol=1,fontsize=7.3)
    export(fig,out,'fig_source_lengths_and_utilization')


def run(analysis, out):
    analysis,out=Path(analysis),Path(out)
    done=read(analysis/'completion.json')
    require(done['status']=='EMAIN_COMMON500_E1_E2_STATISTICS_COMPLETE','Actual completed statistics required')
    for name,digest in done['outputs'].items():
        require(sha(analysis/name)==digest,'Completed table changed: '+name)
    require(not out.exists(),'Fresh figure output required; do not overwrite a published version')
    out.mkdir(parents=True)
    setup()
    data=rows(analysis/'summary_all_methods.csv')
    used=metric_panels(data,MAIN,out,'fig_main_common500_entropy')
    used+=metric_panels(data,MECHANISM,out,'fig_same_prior_four_methods')
    paired=[r for r in rows(analysis/'paired.csv') if r['reference']=='EC_VAR_WHOLE']
    used+=metric_panels(paired,(NEW,),out,'fig_entropy_partial_minus_whole',paired=True)
    prefix_panels(rows(analysis/'prefix_distribution.csv'),out)
    length_panels(rows(analysis/'length_utilization.csv'),out)
    csv_write(out/'plot_data.csv',used)
    for name in ('prefix_distribution.csv','length_utilization.csv','length_summary.csv'):
        (out/name).write_bytes((analysis/name).read_bytes())
    captions=r'''\paragraph{Frozen common500 comparison.} N1024 results at the measured SNRs 4, 10, and 19 dB. Each source first averages three registered noise realizations. Markers show means and error bars show pointwise 95\% source bootstrap confidence intervals; connecting lines only guide the eye between measured points. Original reference intervals are reused unchanged. This is a frozen-policy extension on a previously used evaluation cohort. ConvNeXt agreement is consistency with the source-image prediction, not classification accuracy. All failure outcomes are retained. The adapted SwinJSCC model uses the final 200k checkpoint with the disclosed additional-training and optimizer-reset history.

\paragraph{Matched-prior mechanisms.} Raw and conditional-entropy representations each compare whole-scale and partial-scale transmission with the same frozen visual prior and image decoder. Both families permit fallback to complete scales. Cross-representation comparisons also change packet formats and failure rules and are not a pure two-factor ablation. The paired increment figure reports entropy-coded partial minus entropy-coded whole, using direct source-level difference intervals. Negative LPIPS differences indicate improvement; agreement differences are percentage points.

\paragraph{Actual source coding and prefixes.} Endpoint masses count each of the 500 sources once at each SNR, regardless of the three channel-noise replicates. T denotes the actual transmitted token count. Length distributions compare terminated arithmetic streams with 12T for exactly the same token sequence; the difference describes source representation, not wireless-symbol savings. Capacity utilization divides the actual arithmetic length by k minus 29 paid length/CRC bits. Empirical cumulative distributions retain the denominator of all 500 sources; missing or impossible streams are reported in the accompanying source table. Source fallback is reported separately from channel rejection. Whole-scale lengths are reused from the completed original experiment; no missing length is estimated from probabilities.
'''
    (out/'captions.tex').write_text(captions,encoding='utf8')
    write(out/'completion.json',dict(status='EMAIN_CSV_ONLY_PAPER_FIGURES_COMPLETE',figure_groups=5,
                                    source_statistics_sha256=sha(analysis/'completion.json'),
                                    original_intervals_unchanged=True,bootstrap_calls=0,inference_calls=0,
                                    png_dpi=600,pdf_vector_checked=True,svg_text_editable_checked=True,
                                    visual_review='PENDING_ACTUAL_PNG_OPEN',outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.analysis,a.out)
