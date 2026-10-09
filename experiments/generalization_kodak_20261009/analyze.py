"""Read new completed Kodak24 metric CSV; compute new source statistics and plots.

Only this new population receives bootstrap intervals. No old result, model,
channel, strategy or calibration file is changed or executed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS=('VAR_UNCONDITIONAL','P1024','BPG_ADAPTIVE','SWIN80K')
REFERENCES=METHODS[1:]
SNRS=(4,10,19)
SEED=2026100901
REPLICATES=10000
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
DISPLAY={
 'VAR_UNCONDITIONAL':('Partial-scale digital + unconditional VAR','#0072B2','-','o'),
 'P1024':('Latent continuous JSCC','#D55E00','--','s'),
 'BPG_ADAPTIVE':('Adaptive-resolution BPG + LDPC','#CC79A7',(0,(5,1,1,1)),'v'),
 'SWIN80K':('SwinJSCC-80k (adapted)','#009E73','-.','^'),
}
PANELS=(('PSNR','PSNR (dB) ↑','ΔPSNR (dB) ↑',1),('LPIPS','LPIPS ↓','ΔLPIPS ↓',1),
 ('DINOv2-L','Cosine similarity ↑','ΔDINOv2-L ↑',1),
 ('ConvNeXt','Source-prediction agreement (%) ↑','ΔSource-prediction agreement\n(percentage points) ↑',100))

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def csv_write(path,rows):
    with Path(path).open('w',encoding='utf8',newline='')as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def validate(rows,completion):
    require(completion['status']=='KODAK24_FOUR_METRICS_COMPLETE_V1'and completion['source_count']==24
        and completion['frame_count']==864 and completion['metric_rows']==3456
        and completion['metrics']==list(METRICS)and completion['N']==1024,'Actual completed Kodak four-metric input')
    ids=completion['source_ids'];require(len(ids)==len(set(ids))==24 and len(rows)==3456,'Exactly24 sources and3456 metric cells')
    by_point={};seen=set();references={};frames={};noise_grid={}
    for row in rows:
        method,metric,snr=row['method'],row['metric'],int(row['snr_db']);index=int(row['source_index']);noise=int(row['noise_seed'])
        require(method in METHODS and metric in METRICS and snr in SNRS and 0<=index<24
            and row['source_id']==ids[index]and int(row['N'])==1024 and row['population']=='Kodak24','Unexpected metric/source population')
        key=method,snr,index,noise,metric;require(key not in seen,'Duplicate metric row');seen.add(key)
        value=float(row['value'])
        require(np.isfinite(value),'Nonfinite metric: retain raw value and resolve unbounded PSNR explicitly;do not cap/drop it')
        if metric=='convnext_top1_source_prediction':
            require(value in (0,1)and value==int(int(row['source_prediction'])==int(row['reconstruction_prediction'])),
                'Per-frame agreement must equal the actual source/reconstruction prediction match')
        require(row['true_class_label_available'].lower()=='false','Kodak source prediction agreement needs no invented true label')
        reference=references.setdefault(index,(row['reference_sha256'],row['source_prediction']))
        require(reference==(row['reference_sha256'],row['source_prediction']),'Different source RGB/prediction across methods/noises')
        framekey=method,snr,index,noise
        metadata=(row['source_id'],row['status'],row['reconstruction_sha256'])
        require(framekey not in frames or frames[framekey]==metadata,'Four metrics must describe the same actual frame')
        frames[framekey]=metadata;by_point.setdefault((method,snr,metric),{}).setdefault(index,{})[noise]=value
        noise_grid.setdefault(method,set()).add(noise)
    require(len(frames)==864 and len(by_point)==48 and all(len(seeds)==3 for seeds in noise_grid.values()),'Complete4x3x24x3 frame grid')
    arrays={}
    for method in METHODS:
        for snr in SNRS:
            for metric in METRICS:
                group=by_point[method,snr,metric]
                require(set(group)==set(range(24))and all(set(noises)==noise_grid[method]for noises in group.values()),
                    'No source/noise intersection or success-only subset')
                arrays[method,snr,metric]=np.asarray([np.mean([group[i][n]for n in sorted(noise_grid[method])],dtype=np.float64)
                    for i in range(24)],dtype=np.float64)
    return ids,arrays,frames,{m:sorted(v)for m,v in noise_grid.items()}

def interval(values,draws):
    require(values.shape==(24,)and draws.shape==(REPLICATES,24)and np.isfinite(values).all(),'Source-bootstrap shapes/values')
    estimates=values[draws].mean(axis=1,dtype=np.float64);low,high=np.quantile(estimates,[.025,.975])
    return dict(mean=float(values.mean()),ci_low=float(low),ci_high=float(high),source_count=24,noise_count=3,
        frame_count=72,bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,bootstrap_unit='source after3-noise mean')

def statistics(ids,arrays):
    draws=np.random.default_rng(SEED).integers(0,24,size=(REPLICATES,24))  # One shared new draw array.
    summary=[];paired=[];source_means=[]
    for method in METHODS:
        for snr in SNRS:
            for metric in METRICS:
                vector=arrays[method,snr,metric]
                summary.append(dict(method=method,snr_db=snr,metric=metric,**interval(vector,draws)))
                source_means.extend(dict(method=method,snr_db=snr,metric=metric,source_index=i,source_id=sid,
                    mean=float(vector[i]),noise_count=3)for i,sid in enumerate(ids))
    for reference in REFERENCES:
        for snr in SNRS:
            for metric in METRICS:
                paired.append(dict(method='VAR_UNCONDITIONAL',reference=reference,snr_db=snr,metric=metric,
                    delta_definition='method minus reference',frame_level_noise_pairing_claimed=False,
                    **interval(arrays['VAR_UNCONDITIONAL',snr,metric]-arrays[reference,snr,metric],draws)))
    return summary,paired,source_means,hashlib.sha256(draws.tobytes()).hexdigest()

def plots(out,summary,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True,
        'figure.facecolor':'white','savefig.facecolor':'white','path.simplify':False})
    outputs=[];plot_data=[]
    for kind,table in [('main',summary),('paired',paired)]:
        is_pair=kind=='paired';fig,axes=plt.subplots(2,2,figsize=(7,5.2),dpi=150)
        fig.subplots_adjust(left=.12,right=.985,bottom=.10,top=.84,wspace=.40,hspace=.54)
        shown=REFERENCES if is_pair else METHODS
        handles=[]
        for method in shown:
            label,color,line,marker=DISPLAY[method]
            handles.append(Line2D([],[],label=('VAR − '+label)if is_pair else label,color=color,linestyle=line,marker=marker,linewidth=1.3,markersize=4))
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.99),ncol=1 if is_pair else 2,
            fontsize=7.5,frameon=False,handlelength=2.6,columnspacing=1.2,labelspacing=.35)
        for ax,metric,panel,letter in zip(axes.flat,METRICS,PANELS,'abcd'):
            title,ylabel,dlabel,scale=panel;lows=[];highs=[]
            for method in shown:
                key='reference'if is_pair else'method';group=sorted([r for r in table if r[key]==method and r['metric']==metric],key=lambda r:r['snr_db'])
                require([r['snr_db']for r in group]==list(SNRS),'Three real numerical SNR points')
                x=np.asarray(SNRS);y=np.asarray([r['mean']for r in group])*scale
                lo=np.asarray([r['ci_low']for r in group])*scale;hi=np.asarray([r['ci_high']for r in group])*scale
                lows.extend(lo);highs.extend(hi);_,color,line,marker=DISPLAY[method]
                if is_pair:
                    ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),color=color,linestyle=line,marker=marker,
                        markersize=4,linewidth=1.2,capsize=3,elinewidth=.9,zorder=3)
                else:
                    ax.fill_between(x,lo,hi,color=color,alpha=.12,linewidth=0,zorder=1)
                    if method=='SWIN80K':
                        ax.plot(x[:2],y[:2],color=color,linestyle=line,marker=marker,markersize=4,linewidth=1.3)
                        ax.plot(x[1:],y[1:],color=color,linestyle='--',linewidth=1.3)
                        ax.plot(x[-1:],y[-1:],color=color,linestyle='None',marker=marker,markersize=5,markerfacecolor='white',markeredgewidth=1.2)
                    else:ax.plot(x,y,color=color,linestyle=line,marker=marker,markersize=4,linewidth=1.3)
                plot_data.extend(dict(figure=kind,**r,display_scale=scale,plot_mean=r['mean']*scale,
                    plot_ci_low=r['ci_low']*scale,plot_ci_high=r['ci_high']*scale)for r in group)
            if is_pair:ax.axhline(0,color='#606060',linestyle='--',linewidth=.8);lows.append(0);highs.append(0)
            low,high=min(lows),max(highs);span=high-low if high>low else max(abs(high)*.1,.01)
            ax.set_ylim(low-.09*span,high+.09*span);ax.set_xlim(3.3,19.7);ax.set_xticks(SNRS);ax.set_xlabel('SNR (dB)')
            ax.set_ylabel(dlabel if is_pair else ylabel);ax.set_title(f'({letter}) {title}',loc='left',fontsize=9,pad=7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.grid(color='#DFE3E6',linewidth=.55);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
        texts=[t for ax in fig.axes for t in [ax.xaxis.label,ax.yaxis.label,ax._left_title,*ax.get_xticklabels(),*ax.get_yticklabels()]]
        boxes=[t.get_window_extent(renderer)for t in texts if t.get_visible()and t.get_text()]
        boxes.extend(l.get_window_extent(renderer)for l in fig.legends)
        require(all(b.x0>=-.5 and b.y0>=-.5 and b.x1<=bounds.x1+.5 and b.y1<=bounds.y1+.5 for b in boxes),'Figure text clipping: '+kind)
        for ext in ['pdf','svg','png']:
            path=out/f'kodak24_N1024_{kind}.{ext}';fig.savefig(path,dpi=600 if ext=='png'else 150);outputs.append(path)
        plt.close(fig)
    fields=list(dict.fromkeys(k for row in plot_data for k in row))
    with(out/'plot_data.csv').open('w',encoding='utf8',newline='')as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(plot_data)
    return outputs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=Path,required=True);parser.add_argument('--completion',type=Path)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();cp=args.completion or args.rows.parent/'completion.json';done=read(cp)
    matches=[h for p,h in done['outputs'].items()if Path(p).name==args.rows.name]
    require(len(matches)==1 and sha(args.rows)==matches[0],'Metric CSV must match actual completion')
    require(not args.out.exists(),'New independent output directory required')
    input_hashes={str(args.rows.resolve()):sha(args.rows),str(cp.resolve()):sha(cp)}
    rows=csv_rows(args.rows);ids,arrays,frames,noise=validate(rows,done)
    summary,paired,means,draw_sha=statistics(ids,arrays)
    require(len(summary)==48 and len(paired)==36 and len(means)==1152,'Exact new24-source statistic scope')
    args.out.mkdir(parents=True)
    for name,table in [('summary',summary),('paired',paired),('source_means',means)]:csv_write(args.out/(name+'.csv'),table)
    states=[]
    for method in METHODS:
        for snr in SNRS:
            selected=[v for k,v in frames.items()if k[0]==method and k[1]==snr]
            for state in sorted({v[1]for v in selected}):
                count=sum(v[1]==state for v in selected);states.append(dict(method=method,snr_db=snr,status=state,frames=count,denominator=72,fraction=count/72))
    csv_write(args.out/'status_counts.csv',states)
    plot_outputs=[]if args.no_plots else plots(args.out,summary,paired)
    caption=r'''\newcommand{\KodakGeneralizationCaption}{Frozen-method transfer to Kodak24 with one deterministic $256\times256$ center crop per image, $N=1024$, SNRs of 4, 10, and 19 dB, and three fixed noise realizations. All 24 source images and all failure outputs are retained. Curves and bands show source means and pointwise 95\% source-bootstrap intervals, using 10,000 replicates with seed 2026100901 after averaging three noises within each source. Methods retain their previously frozen source/channel policies without Kodak calibration or training; VAR completion uses the unconditional null class. Higher is better except LPIPS. ConvNeXt reports agreement with the source-image prediction, not ground-truth accuracy. Swin's 19 dB point lies outside its training/calibration range and is drawn hollow with a dashed 10--19 dB connecting segment. No 13 dB observation is invented. This 24-image center-crop experiment is limited cross-collection evidence; non-overlap with all pretrained-model training data is not established.}
\newcommand{\KodakPairedCaption}{Source-paired increments on the same Kodak24 crops: partial-scale digital transmission with unconditional VAR completion minus each frozen reference method. Each method first averages its own three prescribed noises per source; different waveforms are not claimed to share an observed received signal. Error bars are the new source-paired 95\% intervals from the same 10,000 resampling index rows used for all contrasts. Positive favors VAR for PSNR, DINOv2-L, and source-prediction agreement; negative favors VAR for LPIPS, whose sign is unchanged. Agreement differences are percentage points. Swin's 19 dB reference remains outside its training/calibration range. These pointwise intervals are not simultaneous or multiplicity-adjusted evidence of a universal winner.}
'''
    (args.out/'captions.tex').write_text(caption,encoding='utf8')
    report=['# Kodak24 frozen-method transfer','',
        'Actual population: 24 distinct source photographs, one deterministic 256x256 center crop each; N=1024; SNRs 4, 10, and 19 dB; three fixed noises per method/source. This is not an ImageNet-label accuracy experiment and does not establish absence from all pretrained training corpora.',
        '',f'New statistics only: 48 summaries, 36 VAR-minus-reference paired contrasts, and 1152 source means. One {REPLICATES}x24 resampling array uses fixed seed {SEED}. All 864 frames are included, including failures. Old main/supplementary results and intervals are untouched.',
        '', 'No model/PHY/noise or policy evaluation runs in this script. The input CSV is bound to its actual scoring completion. Ground-truth labels are unavailable; ConvNeXt 0/1 source-prediction agreement is used. DINO-L is dinov2_vitl14_cosine. PSNR averages per-image PSNR values, not PSNR of average MSE.',
        '', 'Swin at 19 dB remains outside training/calibration; the 10--19 dB main-curve segment is dashed because 13 dB is not a tested point. All results, negative effects, and confidence intervals remain visible. LPIPS signs are unchanged; agreement differences scale by 100 to percentage points.',
        '', 'Pairing is by source after three-noise means, not by received waveform. The 24-image benchmark and center crop restrict generality; all CIs are pointwise, not multiplicity-adjusted. No quality ranking chooses sources, noise realizations, or a new strategy.',
        '', '| SNR | Reference | Metric | VAR - reference | 95% CI |','|---:|---|---|---:|---|']
    for row in paired:
        scale=100 if row['metric']=='convnext_top1_source_prediction'else 1
        report.append(f"| {row['snr_db']} | {DISPLAY[row['reference']][0]} | {row['metric']} | {row['mean']*scale:.6g} | [{row['ci_low']*scale:.6g}, {row['ci_high']*scale:.6g}] |")
    report+=['','Reproduce into a new output directory:', '',f'`python experiments/generalization_kodak_20261009/analyze.py --rows "{args.rows}" --completion "{cp}" --out NEW_DIRECTORY`',
        '', 'Actual PNG visual inspection is required after export; programmatic text bounds are checked during plotting. The per-frame CSV and fixed dataset/protocol request retain the original method identities and source/noise mapping.']
    (args.out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    require(all(sha(p)==h for p,h in input_hashes.items()),'Input changed during analysis')
    outputs={str(p.resolve()):sha(p)for p in args.out.iterdir()if p.is_file()}
    write_json(args.out/'completion.json',dict(status='KODAK24_SOURCE_STATISTICS_AND_PLOTS_COMPLETE_V1',source_count=24,noise_count=3,
        frame_count=864,point_frame_count=72,summary_rows=48,paired_rows=36,source_mean_rows=1152,
        bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,draws_calls=1,shared_draws_sha256=draw_sha,
        source_ids=ids,noise_seeds_by_method=noise,inputs=input_hashes,outputs=outputs,script_sha256=sha(__file__),
        vector_and_PNG_figures=len(plot_outputs),actual_PNG_visual_review='PENDING',new_model_calls=0,new_PHY_calls=0,
        new_channel_draws=0,old_statistics_modified=False,policy_selection=False))
    print(json.dumps(dict(status='COMPLETE',summary_rows=48,paired_rows=36,figure_formats=len(plot_outputs))))

if __name__=='__main__':main()
