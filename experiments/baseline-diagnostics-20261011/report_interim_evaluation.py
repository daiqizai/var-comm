"""Validate and present a completed fixed calibration diagnostic; no inference."""
from pathlib import Path
import csv, json, math
import numpy as np

ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
OUT=ROOT/'results/baseline_strength_20261011/interim_cal100_v1'
TEMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/interim_cal100_v1')
METHODS=[('P40k','P_continue',40000),('P80k','P_continue',80000),('Swin80k','Swin_1to19',80000),('Swin150k','Swin_1to19',150000)]
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
SNRS=[1,4,7,10,13,19]

def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    done=json.loads((OUT/'completion.json').read_text());wait=json.loads((TEMP/'actual_wait.json').read_text())
    assert done['status']=='COMPLETE_CALIBRATION_ONLY_INTERIM' and done['frames']==7200
    assert wait['exit_code']==0 and wait['actual_child_waited']
    reg=json.loads((OUT/'registration.json').read_text());comparisons={}
    for method,branch,step in METHODS:
        measured=rows(OUT/(method+'_per_frame.csv'));assert len(measured)==1800
        old=rows(ROOT/'results/baseline_strength_20261011'/branch/'calibration'/f'{step:06d}.csv')
        lookup={(x['source_id'],int(x['N']),int(x['snr_db']),int(x['seed'])):x for x in old}
        maximum=0.;count=0;lpmax=0.
        for x in measured:
            key=(x['source_id'],int(x['N']),int(x['snr_db']),int(x['noise_seed']))
            assert x['source_id']==reg['source_ids'][int(x['source_index'])]
            assert all(math.isfinite(float(x[m])) for m in METRICS)
            if key not in lookup:continue
            y=lookup[key];a=10**(-float(x['psnr_db'])/10);difference=abs(a-float(y['mse']));maximum=max(maximum,difference)
            assert math.isclose(a,float(y['mse']),rel_tol=2e-5,abs_tol=2e-7),(method,key,a,y['mse'])
            assert x['header_accepted']==y['header_accepted'],(method,key,'header differs')
            if branch=='P_continue':
                lpmax=max(lpmax,abs(float(x['lpips_alex'])-float(y['lpips'])))
                assert math.isclose(float(x['lpips_alex']),float(y['lpips']),rel_tol=5e-5,abs_tol=2e-5)
            count+=1
        assert count==(1500 if branch=='P_continue' else 600)
        comparisons[method]={'matched_original_calibration_frames':count,'maximum_mse_absolute_difference':maximum,'maximum_lpips_absolute_difference':lpmax if branch=='P_continue' else None}
    validation={'status':'PASS','original_calibration_overlap':comparisons,'actual_wait':wait,
                'all_frames_and_failures_retained':True,'source_count':100,'noise_count':3,'bootstrap_calls':0}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    summary=rows(OUT/'summary.csv');paired=rows(OUT/'paired.csv');lookup={(x['method'],int(x['snr_db']),x['metric']):float(x['mean']) for x in summary}
    delta={(x['method'],int(x['snr_db']),x['metric']):float(x['mean']) for x in paired}
    lines=['# Interim calibration-only baseline evaluation','',
           'Completed while both training branches continued on separate GPUs. These fixed checkpoint snapshots are not the final selected models. No holdout image was evaluated and the new scores do not enter model selection, learning-rate decisions or stopping.','',
           'Population: the first100 sources in the original frozen calibration order, shared by all four methods. N1024; SNR1/4/7/10/13/19dB; noises4101/4102/4103;7200 retained method frames. Average the three repetitions within each source, then average100 sources. These descriptive means have no bootstrap intervals. This small diagnostic does not replace the complete1000-source milestone diagnostics or final500-source evaluation.','',
           'P40k and P80k use the same frozen Dc, original continuous1024-complex-symbol interface and noise construction. Swin80k and Swin150k regenerate their own channel masks, normalization powers, paid metadata and receive observations. Metadata failures retain the original fixed-gray output. Old Swin80k at19dB remains outside its training/calibration range. Swin150k includes the explicitly approved fresh Adam/order/RNG initialization, expanded1–19dB coverage and microbatch4/effective16; changes cannot be attributed solely to training duration.','',
           'The four metrics are float64-MSE PSNR, LPIPS-Alex, DINOv2 ViT-L/14 cosine and ConvNeXt agreement with the source-image prediction. Agreement is not classification accuracy. Frozen metric code, weights and preprocessing are reused. Source labels are not used. The diagnostic was checked against all overlapping original calibration MSE/header outcomes and P LPIPS values.','',
           '## Fixed checkpoint means','',
           '| Method | SNR (dB) | PSNR (dB) | LPIPS | DINOv2-L | ConvNeXt agreement (%) |','|---|---:|---:|---:|---:|---:|']
    for method,_,_ in METHODS:
        for snr in SNRS:
            v=[lookup[method,snr,m] for m in METRICS]
            lines.append(f'| {method} | {snr} | {v[0]:.4f} | {v[1]:.5f} | {v[2]:.5f} | {100*v[3]:.2f} |')
    lines+=['','## Source-paired mean differences','',
            'New checkpoint minus its own parent. Positive PSNR/DINO/agreement and negative LPIPS are favorable. Agreement differences are percentage points; no relative-percent conversion.','',
            '| Comparison | SNR (dB) | Delta PSNR (dB) | Delta LPIPS | Delta DINOv2-L | Delta agreement (pp) |','|---|---:|---:|---:|---:|---:|']
    for method,old in [('P80k','P40k'),('Swin150k','Swin80k')]:
        for snr in SNRS:
            v=[delta[method,snr,m] for m in METRICS]
            lines.append(f'| {method} - {old} | {snr} | {v[0]:+.4f} | {v[1]:+.5f} | {v[2]:+.5f} | {100*v[3]:+.2f} |')
    lines+=['','Full precision is retained in summary.csv, paired.csv and the four per-frame CSVs. No historical scientific result was overwritten. GPU0 performs only this finite diagnostic; P and Swin training continue on GPUs1 and2. This shared-host inference is not a unified latency benchmark.','',
            'Reproduce only into a fresh versioned output directory after inspecting active jobs. The inference entry is experiments/baseline-diagnostics-20261011/run_interim_evaluation.py; it requires the existing external assets and fixed checkpoint hashes. The reporting entry is report_interim_evaluation.py. Neither should modify the bound live training sources.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
    labels=['PSNR (dB)','LPIPS (lower is better)','DINOv2-L cosine similarity','ConvNeXt prediction agreement (%)']
    styles=[('P40k','#0072B2','--','o'),('P80k','#0072B2','-','o'),('Swin80k','#D55E00','--','s'),('Swin150k','#D55E00','-','s')]
    for ax,metric,label in zip(axes.flat,METRICS,labels):
        for name,color,ls,marker in styles:
            scale=100 if metric==METRICS[-1] else 1;values=[lookup[name,snr,metric]*scale for snr in SNRS]
            display={'P40k':'Latent JSCC 40k','P80k':'Latent JSCC 80k','Swin80k':'SwinJSCC 80k','Swin150k':'SwinJSCC 150k'}[name]
            ax.plot(SNRS,values,color=color,linestyle=ls,marker=marker,markersize=4,linewidth=1.3,label=display,markerfacecolor='white' if name in ['P40k','Swin80k'] else color)
        ax.set(xlabel='SNR (dB)',ylabel=label,xticks=SNRS);ax.grid(True,alpha=.22);ax.margins(y=.12)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Fixed calibration snapshots: 100 sources, 3 noises\nInterim diagnostic; SwinJSCC 80k at 19 dB is out of training range',fontsize=9)
    fig.savefig(OUT/'interim_comparison.png',dpi=220);fig.savefig(OUT/'interim_comparison.svg');plt.close(fig)
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
