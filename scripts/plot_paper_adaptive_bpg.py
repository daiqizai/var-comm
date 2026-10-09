#!/usr/bin/env python3
"""Plot completed adaptive-BPG CSV estimates only; no inference or statistics.

Inputs are the actual completed metrics_v1 comparison_summary.csv, paired.csv
and completion.json. Missing groups stop independently. Existing figure files
are never overwritten. No model, channel, bootstrap or policy module is loaded.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

COMMIT='252176e041758ecb2d3e81b6fde5b587e7e17bb7'
SNRS=(1,4,7,10,13,19)
ADAPTIVE='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024_SNR_{snr}'
NATIVE='BPG_LDPC_N1024_SNR_{snr}'
# First three identities/colors/styles/markers match plot_paper_mainraw64.py.
METHODS=(
    ('RAW64_PARTIAL_VAR_COMPLETION_SNR_{snr}','Proposed (raw + partial + VAR)','#0072B2','-','o'),
    ('P1024_SNR_{snr}','Latent continuous JSCC','#D55E00','--','s'),
    ('SWIN80K_N1024_SNR_{snr}','SwinJSCC-80k (adapted)','#009E73','-.','^'),
    (NATIVE,'BPG + LDPC (native 256)','#666666',':','D'),
    (ADAPTIVE,'Adaptive-resolution BPG + LDPC','#CC79A7',(0,(5,1,1,1)),'v'),
)
METRICS=(
    dict(key='psnr_db',name='PSNR',slug='psnr',scale=1,ylabel='PSNR (dB) ↑',delta='ΔPSNR (dB) ↑',unit='dB'),
    dict(key='lpips_alex',name='LPIPS',slug='lpips',scale=1,ylabel='LPIPS ↓',delta='ΔLPIPS ↓',unit='LPIPS'),
    dict(key='dinov2_vitl14_cosine',name='DINOv2-L',slug='dinov2_l',scale=1,ylabel='Cosine similarity ↑',delta='ΔDINOv2-L ↑',unit='cosine similarity'),
    dict(key='convnext_top1_source_prediction',name='ConvNeXt',slug='convnext_agreement',scale=100,
         ylabel='Prediction agreement (%) ↑',delta='ΔPrediction agreement\n(percentage points) ↑',unit='%'),
)
GROUPS=('fig_adaptive_bpg_main_N1024','fig_adaptive_minus_native_bpg')
CAPTIONS=r'''% Existing completed CSV estimates only; this script does not bootstrap.
\newcommand{\AdaptiveBPGMainCaption}{Comparison at $N=1024$ paid forward complex
channel uses on the common 500-source holdout, with three fixed noise realizations
per source and SNR. The proposed partial-scale digital system with VAR completion,
latent continuous JSCC, and adapted SwinJSCC-80k retain their original published
means and pointwise 95\% intervals. Native-resolution and adaptive-resolution BPG
plus LDPC are post-hoc supplementary baselines: their MCS policies are frozen
using the old 100-source calibration population and all-source per-image PSNR,
without selecting on holdout scores. The predefined adaptive source rule uses
sender-side image MSE and actual encoded lengths, never the current noise.
The adaptive codec includes non-learned
downsampling candidates and fixed bicubic restoration to the same 256-pixel
reference; it does not replace the native-resolution setting. All scheduled
outcomes, including source infeasibility, header/body rejection and image decode
failure with the prescribed gray fallback, remain in the reported means.
PSNR, LPIPS-Alex, DINOv2-ViT-L/14 cosine similarity, and ConvNeXt agreement with
the original source prediction are shown; agreement is not classification
accuracy. Higher is better except for LPIPS. Bands are the existing source-level
95\% intervals read from the completed CSV: each source first averages its three
noise realizations. No statistics are recomputed by the plotting code. Equal
sources and total symbol budget do not imply shared noisy observations or equal
training/search budgets. 19 dB is outside Swin's training and calibration range;
its final marker is hollow and the 13--19 dB segment is dashed. The 80k adaptation
is not claimed to be officially optimal or fully converged.}

\newcommand{\AdaptiveBPGPairedCaption}{Source-paired increments of the later-added
adaptive-resolution BPG plus LDPC baseline over the frozen native-resolution BPG
plus LDPC baseline on the same 500 sources and six SNRs: adaptive $-$ native.
Each arm first averages its three prescribed noise realizations per source.
Pairing is by source, not a claim of identical waveforms or noisy observations.
Points, connecting lines and error bars are the already completed paired means
and pointwise 95\% intervals; the zero line is retained and single-method
intervals are never subtracted to manufacture difference intervals. Positive
increments favor adaptive resolution for PSNR, DINOv2-L similarity and ConvNeXt
source-prediction agreement; negative increments favor it for LPIPS, whose sign
is unchanged. Agreement increments and intervals are percentage points, not
relative percentages. Both post-hoc BPG baselines use calibration-only MCS
selection with the stated old 100-source PSNR criterion; source-codec settings
are fixed separately and all failure outputs
are included. The contrast changes the source codec's resolution-selection
setting and possibly its selected MCS, so it is a complete-system comparison,
not a pure one-factor effect of downsampling. No holdout reselection or new
bootstrap is performed while plotting.}
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='')as f:
        return [dict(row,source_row=str(i))for i,row in enumerate(csv.DictReader(f),2)]
def require(ok,message):
    if not ok:raise ValueError(message)
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def select(group,rows):
    paired=group==GROUPS[1];selected=[];errors=[];indexed={}
    for row in rows:
        keys=('method','reference','snr_db','metric')if paired else('point_id','snr_db','metric')
        try:key=tuple(row[k]for k in keys)
        except KeyError as exc:errors.append('CSV missing field '+str(exc));continue
        indexed.setdefault(key,[]).append(row)
    templates=[ADAPTIVE]if paired else[x[0]for x in METHODS]
    for template in templates:
        for metric in METRICS:
            for snr in SNRS:
                point=template.format(snr=snr);reference=NATIVE.format(snr=snr)if paired else''
                key=(point,reference,str(snr),metric['key'])if paired else(point,str(snr),metric['key'])
                found=indexed.get(key,[]);identity=' | '.join(key)
                if len(found)!=1:
                    errors.append(identity+': expected one row, got '+str(len(found)));continue
                row=found[0]
                try:
                    require(int(row['source_count'])==500 and int(row['noise_count'])==3,'500x3 denominator')
                    estimates=[Decimal(row[k])for k in ('mean','ci_low','ci_high')]
                    require(all(x.is_finite()for x in estimates)and estimates[1]<=estimates[0]<=estimates[2],'CI/finite values')
                    if paired:
                        require(row['delta_definition']=='method minus reference','Paired direction')
                        require(row.get('frame_level_noise_pairing_claimed','False').lower()=='false','No shared-noisy-observation claim')
                    else:
                        expected='ADAPTIVE_BPG_NEW_BASELINE'if template==ADAPTIVE else'native256_BPG_UNCHANGED'if template==NATIVE else'published_common500_UNCHANGED'
                        require(row['provenance']==expected,'Exact original/new method provenance')
                except (ValueError,KeyError,ArithmeticError)as exc:
                    errors.append(identity+': '+str(exc));continue
                record=dict(row,figure=group,source_csv='paired.csv'if paired else'comparison_summary.csv',method=point,reference=reference,
                    display_scale=str(metric['scale']),display_unit='percentage points'if paired and metric['scale']==100 else metric['unit'])
                for name in ('mean','ci_low','ci_high'):record['plot_'+name]=str(Decimal(row[name])*metric['scale'])
                selected.append(record)
    return selected,errors


def render(group,rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator,ScalarFormatter
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,
        'axes.linewidth':.65,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.8,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':COMMIT+'adaptive-bpg','axes.unicode_minus':True,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','path.simplify':False})
    paired=group==GROUPS[1];styles=METHODS[-1:]if paired else METHODS;saved=[]
    handles=[Line2D([],[],color=c,linestyle=ls,marker=mk,linewidth=1.35,markersize=4,label=label)for _,label,c,ls,mk in styles]
    def panel(ax,metric,letter=None):
        lows=[];highs=[]
        for template,label,color,style,marker in styles:
            points={template.format(snr=s)for s in SNRS}
            rr=sorted((r for r in rows if r['metric']==metric['key']and r['method']in points),key=lambda r:int(r['snr_db']))
            require([int(r['snr_db'])for r in rr]==list(SNRS),'Validated six numeric points required')
            x=np.asarray(SNRS,dtype=float);y,lo,hi=[np.array([float(r['plot_'+k])for r in rr])for k in ('mean','ci_low','ci_high')]
            lows.extend(lo);highs.extend(hi)
            if paired:
                ax.errorbar(x,y,yerr=np.vstack((y-lo,hi-y)),color=color,linestyle=style,marker=marker,
                    markersize=4.4,linewidth=1.35,elinewidth=1.0,capsize=3,capthick=.9,zorder=4)
            else:
                ax.fill_between(x,lo,hi,color=color,alpha=.12,linewidth=0,zorder=1)
                if template.startswith('SWIN'):
                    ax.plot(x[:5],y[:5],color=color,linestyle=style,marker=marker,linewidth=1.35,markersize=4.4,zorder=3)
                    ax.plot(x[4:],y[4:],color=color,linestyle='--',linewidth=1.35,zorder=3)
                    ax.plot(x[-1:],y[-1:],color=color,linestyle='None',marker=marker,markerfacecolor='white',
                        markeredgewidth=1.2,markersize=5.1,zorder=5)
                else:
                    ax.plot(x,y,color=color,linestyle=style,marker=marker,linewidth=1.35,markersize=4.4,zorder=4)
        if paired:
            ax.axhline(0,color='#656565',linewidth=.8,linestyle=(0,(4,3)),zorder=2);lows.append(0);highs.append(0)
        low,high=min(lows),max(highs);span=high-low if high>low else max(abs(high)*.2,.01)
        ax.set_ylim(low-.09*span,high+.09*span);ax.set_xlim(.3,19.7);ax.set_xticks(SNRS);ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(metric['delta']if paired else metric['ylabel'],labelpad=5)
        ax.set_title((f'({letter}) 'if letter else'')+metric['name'],loc='left',fontweight='semibold',pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5));fmt=ScalarFormatter(useOffset=False);fmt.set_powerlimits((-3,4));ax.yaxis.set_major_formatter(fmt)
        ax.grid(color='#DFE3E6',linewidth=.55);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        for edge in ('left','bottom'):ax.spines[edge].set_color('#535353')
    def export(fig,stem):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;texts=[]
        for ax in fig.axes:texts.extend([ax.xaxis.label,ax.yaxis.label,ax._left_title,ax.yaxis.get_offset_text(),*ax.get_xticklabels(),*ax.get_yticklabels()])
        if fig._suptitle:texts.append(fig._suptitle)
        boxes=[t.get_window_extent(renderer)for t in texts if t.get_visible()and t.get_text()]
        boxes.extend(legend.get_window_extent(renderer)for legend in fig.legends)
        require(all(b.x0>=-.5 and b.y0>=-.5 and b.x1<=bounds.x1+.5 and b.y1<=bounds.y1+.5 for b in boxes),'Text/legend clipped: '+stem)
        for ext in ('pdf','svg','png'):
            path=out/(stem+'.'+ext);require(not path.exists(),'Refuse to overwrite '+str(path))
            kwargs={'dpi':600}if ext=='png'else{}
            if ext=='pdf':kwargs['metadata']={'Title':stem,'Author':'VAR_COMM','Subject':'Completed holdout CSV estimates; no recomputation'}
            fig.savefig(path,format=ext,**kwargs);saved.append(path.name)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7,5.2),dpi=150)
    fig.subplots_adjust(left=.105,right=.985,bottom=.10,top=.85,wspace=.36,hspace=.56)
    if paired:fig.suptitle('Adaptive-resolution BPG − Native-resolution BPG',fontsize=9,fontweight='semibold',y=.96)
    else:fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False,handlelength=2.5,columnspacing=1.1,labelspacing=.55)
    for ax,metric,letter in zip(axes.flat,METRICS,'abcd'):panel(ax,metric,letter)
    export(fig,group)
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(3.45,2.8 if paired else 3.6),dpi=150)
        fig.subplots_adjust(left=.23,right=.975,bottom=.18 if paired else .15,top=.77 if paired else .63)
        if paired:fig.suptitle('Adaptive BPG − Native BPG',fontsize=8.5,fontweight='semibold',y=.97)
        else:fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.99),ncol=1,frameon=False,handlelength=2.5,fontsize=7.8,labelspacing=.3)
        panel(ax,metric);export(fig,group+'_'+metric['slug'])
    return saved


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=root/'results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1')
    p.add_argument('--output-dir',type=Path,default=root/'paper/figures/adaptive_bpg_holdout500')
    p.add_argument('--validate-only',action='store_true',help='Read actual completed input only; create no figures or files')
    a=p.parse_args();data=a.data_dir.resolve();out=a.output_dir.resolve()
    require(out!=data and not out.is_relative_to(data),'Independent figure output required')
    cp=data/'completion.json';require(cp.is_file(),'Actual metrics completion is missing; figures NOT_RUN: '+str(cp))
    complete=read(cp)
    require(complete['status']=='ADAPTIVE_BPG_FOUR_METRICS_NEW_PAIRED_STATISTICS_COMPLETE_V1'
        and complete['source_count']==500 and complete['frame_count']==9000 and complete['summary_rows']==24
        and complete['paired_rows']==96 and complete['old_summary_recomputed']is False
        and complete['old_bootstrap_recomputed']is False,'Completed exact500x6x3 new baseline required')
    require(complete['metrics']==[m['key']for m in METRICS],'Exact four metrics; DINOv2-L required')
    grouped={};issues={};input_hashes={'completion.json':sha(cp)}
    for group,filename in zip(GROUPS,('comparison_summary.csv','paired.csv')):
        path=data/filename
        if not path.exists():issues[group]=['Missing file: '+str(path)];continue
        expected=[h for name,h in complete['outputs'].items()if Path(name).name==filename]
        require(len(expected)==1 and sha(path)==expected[0],'Changed/unbound completed CSV: '+filename)
        input_hashes[filename]=sha(path);selected,errors=select(group,read_csv(path))
        if errors:issues[group]=errors
        else:grouped[group]=selected
    if a.validate_only:
        print(json.dumps(dict(status='VALIDATED_NO_FIGURES'if not issues else'PARTIAL_INPUT',groups={k:len(v)for k,v in grouped.items()},issues=issues,inputs=input_hashes),indent=2));return
    require(not out.exists(),'Use a new output directory; existing scientific figures are not overwritten')
    out.mkdir(parents=True)
    files=[];used=[]
    for group,rows in grouped.items():files.extend(render(group,rows,out));used.extend(rows)
    fields=['figure','source_csv','source_row','point_id','method','reference','snr_db','metric','mean','ci_low','ci_high',
        'display_scale','display_unit','plot_mean','plot_ci_low','plot_ci_high','source_count','noise_count','delta_definition']
    fields+=sorted(set().union(*(r.keys()for r in used))-set(fields))if used else[]
    with(out/'plot_data.csv').open('w',encoding='utf8',newline='')as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(used)
    (out/'captions.tex').write_text(CAPTIONS,encoding='utf8')
    mapping='\n'.join('| '+label+' | `'+template+'` |'for template,label,*_ in METHODS)
    (out/'README.md').write_text(f'''# Adaptive BPG supplementary holdout figures

Original main result commit: `{COMMIT}`. New adaptive-BPG statistics and unchanged old-method estimates are read from `{data}`. This is a later-added baseline comparison, not a rewritten preregistration. Both BPG settings retain calibration-only selection; adaptive source resolution is explicitly part of its codec.

Inputs are completion-bound `comparison_summary.csv` and `paired.csv`. Main rows use500 sources×3 noises; per-source noise means and existing95% intervals are used at CSV precision. The plotting code does not train, infer, simulate a channel, recompute old statistics, bootstrap, or choose policy. Failure outcomes remain included. The paired contrast is adaptive−native, using paired intervals directly. Pairing by source is not a claim of shared noisy observations.

| Public label | Internal point template |
|---|---|
{mapping}

The first three colors/styles/markers match `plot_paper_mainraw64.py`; the continuous system is labelled Latent continuous JSCC publicly. Native BPG is gray/dotted/diamond and adaptive BPG purple/custom-dashed/down-triangle. Swin19 dB is hollow with a dashed13–19 segment; it remains outside training/calibration. LPIPS is not flipped. ConvNeXt agreement is a percentage; its paired increment is percentage points, not accuracy or relative percent.

Each complete group has a seven-inch2×2 composite plus four separately laid-out single-column panels. PDF/SVG curves and text are vector (SVG text editable); PNG is600 dpi. Real SNR spacing and all finite intervals, zero/negative differences are retained. No spline, broken axis, or per-point number labels are used.

Reproduce to a fresh directory:

```text
python scripts/plot_paper_adaptive_bpg.py --data-dir "{data}" --output-dir NEW_EMPTY_OUTPUT_DIRECTORY
```

`--validate-only` checks real completed inputs without making images. `validation.json` records exact missing rows and stopped groups. `plot_data.csv` retains the precise original CSV strings and only the required factor100 display conversions. Existing paper figures and scientific result files are not overwritten. Before paper use, open both combined PNGs and check layout, labels and failure/zero points.
''',encoding='utf8')
    require(all(sha(data/name)==h for name,h in input_hashes.items()),'Inputs changed during export')
    save_json(out/'validation.json',dict(status='COMPLETE'if not issues else'PARTIAL_COMPLETE_GROUPS_ONLY',
        input_hashes=input_hashes,rows_by_group={k:len(v)for k,v in grouped.items()},issues=issues,
        output_files={name:sha(out/name)for name in files},new_inference=0,new_channel=0,new_bootstrap=0,
        old_statistics_recomputed=False,visual_inspection='PENDING_ACTUAL_PNG_OPEN'))
    print(json.dumps(dict(status='COMPLETE'if not issues else'PARTIAL_COMPLETE_GROUPS_ONLY',figures=len(files),issues=issues,output=str(out)),indent=2))


if __name__=='__main__':main()
