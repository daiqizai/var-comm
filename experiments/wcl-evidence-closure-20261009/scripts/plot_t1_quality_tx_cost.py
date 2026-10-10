#!/usr/bin/env python3
"""Join closed quality statistics and actual online timing; plotting only."""
import argparse
from decimal import Decimal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import plot_t5_entropy_quality as q

TIMING_SHA='2ddc567174e11ac8720b481f91652fe33389c74158c8383b75cc50a9ab3ce373'
WAIT_SHA='4540c60c1c94eddc1d020ea8604012025ae530d8f5b934b2f8d5d8c44937a6bb'
WHOLE='RAW64_WHOLE_VAR_COMPLETION'
METHODS=((WHOLE,'Complete-scale raw + VAR','#555555','--','s'),)+q.METHODS
TIMING_IDS={WHOLE:'RAW64_WHOLE',q.PARTIAL:'RAW64_PARTIAL',
            'EC_STATIC_WHOLE':'EC_STATIC_WHOLE','EC_VAR_WHOLE':'EC_VAR_WHOLE'}


def timing_data(path):
    q.require(q.sha(path/'completion.json')==TIMING_SHA and q.sha(path/'actual_wait_verification.json')==WAIT_SHA,
              'Exact observed successful timing run required')
    done=q.read(path/'completion.json');wait=q.read(path/'actual_wait_verification.json')
    q.require(wait['status']=='ACTUAL_T4_CHILD_EXIT_AND_SEALS_VERIFIED' and wait['actual_child_waited'] is True
              and wait['exit_code']==0 and wait['completion_sha256']==TIMING_SHA
              and wait['request_sha256']==done['request_sha256'],'Actual timing exit0 receipt mismatch')
    q.require(done['status']=='T4_FIXED16_ONLINE_TIMING_COMPLETE' and done['source_count']==16
              and done['total_frames']==1152 and done['measured_frames']==done['warmup_frames']==576
              and done['packet_budget']['complete_frames']==1152 and done['packet_budget']['unresolved_frames']==0
              and done['no_online_tokens_or_output_cache'] is True and done['snrs']==list(q.SNRS), 'Timing scope mismatch')
    inputs={str(path/n):q.sha(path/n) for n in ('completion.json','actual_wait_verification.json')}
    for name in ('timing_summary.csv','timing_per_call.csv'):
        expected=q.one([h for p,h in done['outputs'].items() if p.endswith('/'+name)])
        q.require(q.sha(path/name)==expected,'Timing output seal mismatch: '+name);inputs[str(path/name)]=expected
    summary=q.readcsv(path/'timing_summary.csv');calls=q.readcsv(path/'timing_per_call.csv')
    q.require(len(calls)==1152 and {r['phase'] for r in calls}=={'warmup','measured'},'All actual timing rows required')
    grid={(r['method'],int(r['snr_db']),r['source_id'],r['phase'],int(r['repetition'])) for r in calls}
    q.require(len(grid)==1152 and all(r['request_sha256']==done['request_sha256']
              and r['source_role']=='development_timing' and r['online_source_cache']==r['online_output_cache']=='False'
              and int(r['batch_size'])==1 for r in calls),'Timing identity/cache policy differs')
    selected=[]
    for family,*_ in METHODS:
        method=TIMING_IDS[family]
        for snr in q.SNRS:
            row=q.one([r for r in summary if r['method']==method and int(r['snr_db'])==snr
                       and r['metric']=='TX_seconds' and r['condition']=='all'])
            actual=[r for r in calls if r['method']==method and int(r['snr_db'])==snr and r['phase']=='measured']
            ids={r['source_id'] for r in actual}
            q.require(len(actual)==int(row['sample_count'])==48 and len(ids)==int(row['source_count'])==16
                      and all({int(r['repetition']) for r in actual if r['source_id']==sid}=={0,1,2} for sid in ids),
                      'Every timing point must have fixed16 x three measured repetitions')
            values=np.asarray([float(r['TX_seconds']) for r in actual])
            q.require(np.isfinite(values).all() and (values>0).all() and np.isclose(values.mean(),float(row['mean']),rtol=1e-13)
                      and np.isclose(np.median(values),float(row['median']),rtol=1e-13),'Actual all-frame TX summary differs')
            selected.append(dict(row,family=family,tx_mean_ms=str(Decimal(row['mean'])*1000),
                                 tx_median_ms=str(Decimal(row['median'])*1000)))
    return selected,inputs,done


def gather(args):
    rows,inputs,quality=q.gather(args.statistics.resolve(),args.original_summary.resolve(),args.freeze.resolve())
    rows=[r for r in rows if r['figure']=='quality']
    original=q.readcsv(args.original_summary);copied=q.read(args.statistics/'raw_reference_summary_unchanged.json')
    for metric,*rest in q.METRICS:
        scale=rest[-2]
        for snr in q.SNRS:
            point=WHOLE+'_SNR_'+str(snr)
            row=q.one([r for r in original if r['point_id']==point and r['metric']==metric])
            prior=q.one([r for r in copied if r['point_id']==point and r['metric']==metric])
            q.require(all(Decimal(row[k])==Decimal(str(prior[k])) for k in ('mean','ci_low','ci_high')),'Original complete-scale interval changed')
            rows.append(dict(row,family=WHOLE,public_method=METHODS[0][1],source_csv='original_published/summary.csv',
                             display_scale=scale,**{f'plot_{k}':str(Decimal(row[k])*scale) for k in ('mean','ci_low','ci_high')}))
    costs,bindings,timing=timing_data(args.timing.resolve());inputs.update(bindings)
    for r in rows:
        c=q.one([c for c in costs if c['family']==r['family'] and c['snr_db']==r['snr_db']])
        r.update(figure='quality_vs_tx_cost',timing_method=c['method'],cost_population='fixed16 development timing',
                 quality_population='original500 posthoc same-source holdout',cost_source_count=16,
                 cost_repetitions_per_source=3,cost_sample_count=48,cost_phase='measured only',
                 tx_mean_seconds=c['mean'],tx_median_seconds=c['median'],tx_p95_seconds=c['p95'],
                 tx_mean_ms=c['tx_mean_ms'],tx_median_ms=c['tx_median_ms'],cost_ci='not estimated')
    q.require(len(rows)==48,'Four methods x three SNRs x four metrics required')
    return rows,inputs,timing


def panel(ax,rows,metric,snr,statistic):
    key,title,ylabel,_,_,_=metric;bounds=[]
    for family,name,color,linestyle,marker in METHODS:
        row=q.one([r for r in rows if r['family']==family and int(r['snr_db'])==snr and r['metric']==key])
        mean,lo,hi=[float(row['plot_'+k]) for k in ('mean','ci_low','ci_high')]
        face='white' if family in (WHOLE,'EC_VAR_WHOLE') else color
        ax.errorbar([float(row['tx_'+statistic+'_ms'])],[mean],yerr=[[mean-lo],[hi-mean]],
                    color=color,marker=marker,markerfacecolor=face,ms=5,lw=1.05,capsize=3,zorder=3)
        bounds.extend((lo,hi))
    low,high=min(bounds),max(bounds);pad=max((high-low)*.14,1e-5)
    ax.set_ylim(low-pad,high+pad);ax.set_xscale('log');ax.set_xlim(10,500)
    ax.set_xticks([10,20,50,100,200,500]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.minorticks_off()
    ax.set_xlabel(statistic.title()+' TX time (ms; log scale)');ax.set_ylabel(ylabel);ax.set_title(title,pad=7)
    ax.grid(True);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)


def figures(rows,out):
    handles=[Line2D([],[],color=c,marker=m,markerfacecolor='white' if f in (WHOLE,'EC_VAR_WHOLE') else c,
                    linestyle='none',ms=5,label=p) for f,p,c,l,m in METHODS]
    for statistic in ('mean','median'):
        for snr in q.SNRS:
            stem=f'fig_t1_quality_vs_tx_{statistic}_N1024_{snr}dB'
            fig,axes=plt.subplots(2,2,figsize=(7,5.8))
            for ax,metric in zip(axes.flat,q.METRICS):panel(ax,rows,metric,snr,statistic)
            fig.suptitle(f'Quality and online transmitter cost · N = 1024 · {snr} dB',y=.985,fontsize=10)
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=1,frameon=False,labelspacing=.35)
            fig.subplots_adjust(left=.113,right=.985,bottom=.135,top=.755,wspace=.4,hspace=.65)
            fig.text(.5,.042,'Quality: 500 sources × 3 noises, existing 95% CIs\nTX timing: 16 development sources × 3 measured repeats; no timing CI',
                     ha='center',va='center',fontsize=7.2)
            q.save(fig,out,stem);plt.close(fig)


CAPTION=r'''\newcommand{\ToneQualityTxCostCaption}{Quality and online transmitter software time at $N=1024$, separately at4,10 and19~dB.
The horizontal axis reports the arithmetic mean (primary panels) or median (companion panels) of actual measured TX time in milliseconds on a logarithmic scale.
TX spans the input image tensor to the full I/Q frame, including visual encoding, necessary probability computation, entropy encoding and length-based fallback attempts, FEC and modulation; model loading, file I/O, metrics, queueing and channel propagation are excluded.
The same NVIDIA GeForce RTX4090D environment, batch size1, frozen numerical settings and GPU synchronization protocol are used for all four systems. Each point aggregates16 fixed development sources with three measured repetitions after the registered warm-ups (48 measured calls); these small-sample timings are descriptive, with no timing confidence interval or universal deployment/tail-latency claim.
Quality means and vertical error bars instead come from the previously examined500-source holdout, three noises per source, with existing pointwise95\% source-bootstrap intervals and no multiple-comparison correction.
This is a post-hoc comparison of method/workpoint aggregates across two different populations, not an image-paired quality-versus-cost correlation. All scheduled failure outcomes remain included. ConvNeXt reports prediction agreement, not accuracy.
The original raw complete-scale and partial-scale policies and intervals are unchanged; the complete-scale entropy policies were selected on calibration data. The plotted protocols retain their distinct raw KEEP versus entropy rejection/fallback rules. No quality value, confidence interval or timing is synthesized for an unmeasured setting.}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('statistics','original-summary','freeze','timing','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();q.require(not a.out.exists(),'Fresh independent figure directory required')
    rows,inputs,timing=gather(a);a.out.mkdir(parents=True);q.style();figures(rows,a.out)
    q.csvout(a.out/'quality_vs_tx_cost.csv',rows);q.csvout(a.out/'plot_data.csv',rows)
    (a.out/'captions.tex').write_text(CAPTION,encoding='utf-8')
    (a.out/'README.md').write_text('# Actual quality and TX cost\n\n'
        'Six completed2x2 figure groups: mean TX(primary) and median TX(companion), each at4/10/19dB. '
        'PDF/SVG contain vector points/error bars/text; PNG600dpi. Logarithmic time axis is explicitly labelled. '
        'Quality uses500 sources x3 noises; TX uses16 development sources x3 measured repeats. '
        'Warm-ups are excluded. No cross-population per-image correlation or paired interval is constructed. '
        'Vertical intervals are existing quality CIs. Timing has no inferred CI.\n\n'
        'The actual1152-frame timing owner completed with observed exit0;576 measured rows and576 warm-ups are sealed. '
        'The plot validates all-frame TX means and medians against timing_per_call.csv and verifies the original raw quality values unchanged. '
        'All four methods, three registered SNRs and four metrics are retained, including VAR-conditional entropy at19dB with higher quality and greater TX time than NeST-Com.\n\n'
        'Reproduce with Python, numpy and matplotlib into a fresh output directory:\n\n```text\n'
        'python experiments/wcl-evidence-closure-20261009/scripts/plot_t1_quality_tx_cost.py '
        f'--statistics "{a.statistics}" --original-summary "{a.original_summary}" --freeze "{a.freeze}" '
        f'--timing "{a.timing}" --out "<fresh-output-directory>"\n```\n',encoding='utf-8')
    done=dict(status='T1_QUALITY_TX_COST_ACTUAL_EXISTING_DATA_PLOTS_COMPLETE',N=1024,snrs=q.SNRS,plot_rows=48,
              quality_source_count=500,quality_noise_count=3,timing_source_count=16,timing_measured_repeats=3,
              timing_samples_per_point=48,quality_cost_same_population=False,post_hoc_same_source_quality=True,
              timing_statistics=['mean','median'],cost_units='milliseconds',cost_axis='logarithmic',timing_CI_estimated=False,
              actual_timing_runtime=timing['runtime'],input_bindings=inputs,
              source_bindings={str(Path(__file__).resolve()):q.sha(__file__),str(Path(q.__file__).resolve()):q.sha(q.__file__)},
              new_model_calls=0,new_packet_calls=0,new_bootstrap_calls=0,
              output_hashes={p.name:q.sha(p) for p in sorted(a.out.iterdir()) if p.is_file()})
    q.write(a.out/'completion.json',done);print(done['status'])


if __name__=='__main__':main()
