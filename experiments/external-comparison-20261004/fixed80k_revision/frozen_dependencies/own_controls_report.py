"""CPU-only, matched five-method report. Missing receiver cost keeps the full plan open."""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.util
import itertools
import json
import os
from pathlib import Path
import signal
import sys
import textwrap
import time
import uuid
import numpy as np
import external_eval_common as c
import external_eval_report as external

HERE=Path(__file__).resolve().parent
FAMILIES=('P','D_U','M1','SwinJSCC','HiFiDiffCom')
METHODS={1024:('P1024','D_U_QPSK','M1_entropy_N1024_frozen',*c.METHODS),
         2048:('P2048','D_U_whole_N2048_v1','M1_entropy_N2048_v1',*c.METHODS)}
LABELS={'P':'Continuous P','D_U':'Digital VAR (unconditional)','M1':'M1 calibrated policy',
        'SwinJSCC':'SwinJSCC','HiFiDiffCom':'HiFi-DiffCom + Swin'}
METRICS=external.METRIC_LABELS
LOWER={'lpips_alex','dists','dreamsim','semantic_error','confidently_wrong'}
REFERENCE_CONDITIONS=['identity','unrelated','same_class','gaussian_blur_sigma1','gaussian_noise_sigma2_255','jpeg_quality90']


def require(value,message):
    if not value:raise RuntimeError(message)


def family(n,method):
    require(int(n) in METHODS and method in METHODS[int(n)],'Unregistered five-method member')
    return FAMILIES[METHODS[int(n)].index(method)]


def row_key(row):
    return int(row['source_index']),int(row['N']),int(row['snr_db']),int(row['noise_seed']),family(row['N'],row['method'])


def admitted_json(path,bindings):
    path=Path(path)
    require(bindings.get(str(path))==c.sha(path),'Metadata is not an authenticated scorer input: '+str(path))
    return c.read(path)


def policy_table(indexed,digital1024,m1_original,new2048):
    """Disclose calibration choices; never infer a new winner from development."""
    require(digital1024.get('development_read') is False and m1_original.get('development_read') is False
            and new2048.get('development_read') is False,'Policy choice used development')
    selected={}
    for s in c.SNRS:
        selected[1024,s,'D_U']=dict(m=int(digital1024['levels'][str(s)]['QPSK']['U']['action_m']),q=0,order='whole')
        matches=[x for x in m1_original['cells'] if (int(x['N']),x['phy_family'],int(x['snr_db']),x['method'])==(1024,'QPSK',s,'entropy')]
        require(len(matches)==1,'Frozen N1024 entropy policy is missing or duplicated')
        selected[1024,s,'M1']=matches[0]['action']
        for method_family,method in zip(('D_U','M1'),METHODS[2048][1:3]):
            matches=[x for x in new2048['cells'] if (int(x['snr_db']),x['method'])==(s,method)]
            require(len(matches)==1,'New N2048 calibrated policy is missing or duplicated')
            selected[2048,s,method_family]=matches[0]['action']
    answer=[]
    for (n,s,m),action in sorted(selected.items()):
        wanted=(int(action['m']),int(action['q']),action['order'])
        require(wanted[1]>=0 and (m!='D_U' or wanted[1]==0),'Whole control cannot have partial tokens')
        frames=[indexed[i,n,s,seed,m] for i in range(100) for seed in c.SEEDS]
        for row in frames:
            original=json.loads(row['original_row_json'])
            require(c.identity(original)==row['original_row_sha256'],'Original per-frame provenance changed')
            native=original['original_scientific_row'] if n==1024 else original
            actual=(int(native['action_m']),0,'whole') if n==1024 and m=='D_U' else (int(native['m']),int(native['q']),native['order'])
            require(actual==wanted,'Scored development action differs from its frozen calibration policy')
        same_rgb=sum(indexed[i,n,s,seed,'M1']['image_sha256']==indexed[i,n,s,seed,'D_U']['image_sha256']
                     for i in range(100) for seed in c.SEEDS)
        if n==2048 and m=='M1' and wanted[1]==0:
            other=selected[n,s,'D_U'];same_action=wanted==(int(other['m']),int(other['q']),other['order'])
            require(not same_action or same_rgb==300,'Identical N2048 whole policy did not retain identical outputs')
        answer.append(dict(N=n,snr_db=s,comparison_family=m,method=METHODS[n][FAMILIES.index(m)],m=wanted[0],
            K=wanted[1],order=wanted[2],partial_scale_active=wanted[1]>0,
            operating_mode='partial_scale' if wanted[1]>0 else 'whole_scale_K0',
            policy_origin='original_frozen_N1024' if n==1024 else 'new_N2048_calibration_only',
            header_N=68,body_N=n-68,paid_class_bits=10 if n==1024 and m=='D_U' else 0,
            class_condition_used=False,all_300_actions_match=True,
            same_rgb_as_D_U_frames=same_rgb if m=='M1' else '',comparison_frames=300))
    return answer


def protocol_metadata(root,indexed,own_registration,external_completion):
    root=Path(root);own_inputs=dict(own_registration['input_bindings']);bindings={}
    # P512 is context only and is not constructed by the selected N1024 replay.
    # Its exact completion remains authenticated by the original completed
    # unified evaluation registration, which the own scorer already admits.
    old_registration=root/'results/unified_metrics_20261002/metrics_registration.json'
    published_original=admitted_json(old_registration,own_inputs)
    for path,digest in published_original['frozen_policy_bindings'].items():
        require(path not in own_inputs or own_inputs[path]==digest,'Conflicting original model/policy identity')
        own_inputs[path]=digest
    bindings[str(old_registration)]=c.sha(old_registration)
    old_dir=root/'results/extreme_bandwidth_20261001_R1_N1024'
    paths=[old_dir/'digital_selected_policy.json',root/'results/scale_causal_partial_residual_20261002/m1_policy.json',
           root/'results/external_comparison_20261004/own_controls/N2048_policy.json']
    policies=[admitted_json(p,own_inputs) for p in paths];bindings.update({str(p):c.sha(p) for p in paths})
    require(policies[2]==own_registration['policy'],'Score and report N2048 policy differ')
    policy_rows=policy_table(indexed,*policies)
    training=[]
    for n,name in ((512,'EXTREME-BW-20260930-R1'),(1024,'EXTREME-BW-20261001-R1-N1024')):
        folder=root/'outputs'/name/'training'/f'p{n}_2026093001'
        dp,rp=folder/'completion.json',folder/'registration.json'
        done,reg=admitted_json(dp,own_inputs),admitted_json(rp,own_inputs)
        require(done.get('status')=='CALIBRATION_ONLY_SELECTION_COMPLETE' and done.get('synthetic') is False
                and done.get('development_read') is False and done['registration_sha256']==c.sha(rp)
                and done['state']['finished'] is True,'Original continuous training is not complete')
        chosen=done['selected'][f'P{n}'];decision=done['state']['decisions'][-1]
        require(type(done.get('budget_truncated')) is bool and done['budget_truncated']==decision['budget_truncated']
                and chosen['training_seed']==2026093001,'Original budget/seed disclosure differs')
        training.append(dict(method=f'P{n}',N=n,main_comparison=n==1024,training_seed=chosen['training_seed'],
            selected_step=int(chosen['step']),completed_step=int(done['state']['step']),
            maximum_steps=int(reg['protocol']['maximum_updates']),budget_truncated=done['budget_truncated'],
            stop_reason=decision['reason'],convergence_claimed=False,checkpoint_sha256=chosen['checkpoint_sha256'],
            loss='RGB MSE + 0.1 LPIPS-Alex + 0.01 normalized latent error'))
        bindings.update({str(p):c.sha(p) for p in (dp,rp)})
    # The already authenticated P2048 row carries its original selected model,
    # actual initialization seed and completed training length. No new fit.
    p_metadata=[]
    for i in range(100):
        for s in c.SNRS:
            for seed in c.SEEDS:
                source=json.loads(indexed[i,2048,s,seed,'P']['original_row_json'])
                meta=json.loads(source['history_metadata_json'])
                p_metadata.append({k:meta[k] for k in ('training_seed','initialization_seed','selected_step','training_completed_step','checkpoint_sha256')})
    require(all(x==p_metadata[0] for x in p_metadata) and p_metadata[0]['training_seed']==2026092304,
            'P2048 rows use different preselected training runs')
    p=p_metadata[0]
    training.append(dict(method='P2048',N=2048,main_comparison=True,training_seed=p['training_seed'],
        initialization_seed=p['initialization_seed'],selected_step=p['selected_step'],completed_step=p['training_completed_step'],
        maximum_steps='',budget_truncated='not_reclassified_original_completed_run',stop_reason='original_selected_checkpoint',
        convergence_claimed=False,checkpoint_sha256=p['checkpoint_sha256'],
        loss='RGB MSE + 0.1 LPIPS-Alex + 0.01 normalized latent error'))
    folder=root/'outputs/EXTERNAL-COMPARISON-20261004/swin_training'
    rp,sp,dp=folder/'registration.json',folder/'selected_swin.json',folder/'completion.json'
    sr,selected,done=[admitted_json(p,external_completion['bindings']) for p in (rp,sp,dp)]
    config=sr['config']
    require(done.get('status')=='SWIN_TRAINING_COMPLETE' and done.get('synthetic') is False
            and selected['registration_sha256']==done['registration_sha256']==c.sha(rp)
            and selected.get('development_read') is False and selected.get('holdout_read') is False
            and done['budget_truncated']==selected['budget_truncated'] and type(done['budget_truncated']) is bool,
            'Selected shared Swin training disclosure differs')
    require(config['total_N']==[1024,2048] and config['data_N']==[768,1664] and config['header_N']==[256,384]
            and config['maximum_steps']==240000,'External paid-header budget or registered cap changed')
    training.append(dict(method='SwinJSCC shared; also frozen HiFi likelihood',N='1024+2048',main_comparison=True,
        training_seed=config['seed'],selected_step=selected['step'],completed_step=done['completed_step'],
        maximum_steps=config['maximum_steps'],budget_truncated=done['budget_truncated'],
        stop_reason='registered_budget_or_calibration_plateau_stop',convergence_claimed=False,
        checkpoint_sha256=selected['checkpoint_sha256'],loss='RGB MSE only'))
    bindings.update({str(p):c.sha(p) for p in (rp,sp,dp)})
    resources=[dict(N=n,comparison_family=m,method=METHODS[n][FAMILIES.index(m)],
        body_N=n if m=='P' else n-68 if m in ('D_U','M1') else 768 if n==1024 else 1664,
        header_N=0 if m=='P' else 68 if m in ('D_U','M1') else 256 if n==1024 else 384,
        E=2*n,metadata=('none' if m=='P' else 'paid 10-bit class ignored by unconditional receiver; mode; CRC/tail/rate-matching'
            if m=='D_U' and n==1024 else 'm/K/order; CRC/tail/rate-matching; no class' if m in ('D_U','M1')
            else 'paid mask rank/power; CRC/tail/rate-matching')) for n in c.BUDGETS for m in FAMILIES]
    return dict(policies=policy_rows,training=training,resources=resources),bindings


def policy_title(n,s,method,policies):
    if method not in ('D_U','M1'):return LABELS[method]
    rows=[x for x in policies if (x['N'],x['snr_db'],x['comparison_family'])==(n,s,method)]
    require(len(rows)==1,'Fixed figure policy annotation missing')
    row=rows[0];base='Digital VAR' if method=='D_U' else 'M1 policy'
    mode=f"whole-scale K=0, m{row['m']}" if row['K']==0 else f"entropy m{row['m']} + K={row['K']}"
    return base+'\n'+mode


def validate_rows(own_rows,external_rows,records,manifest_sha):
    external.validate_inventory(external_rows)
    rows=[*own_rows,*external_rows];index={row_key(r):r for r in rows}
    wanted={(i,n,s,seed,m) for i in range(100) for n in c.BUDGETS for s in c.SNRS for seed in c.SEEDS for m in FAMILIES}
    require(len(own_rows)==5400 and len(external_rows)==3600 and len(index)==len(rows)==9000 and set(index)==wanted,
            'Five-method report needs all 9000 unique paired rows')
    require(len(records)==100 and len({r['image_id'] for r in records})==100,'Incomplete source population')
    baselines={}
    for key,row in index.items():
        i,n,s,seed,m=key;record=records[i]
        require(row['source_id']==record['image_id'] and row['preprocessing_id']==record['preprocessing_id']
                and int(row['true_class_index'])==int(record['class_index']), 'Five-method source/preprocessing/class mismatch')
        require(not external.boolean(row['synthetic']) and not external.boolean(row['label_conditioned'])
                and not external.boolean(row.get('semantic_side_information',False))
                and row.get('decoder_id')!='D0', 'Conditioned or D0 output cannot enter the main comparison')
        require(row['modelmanifest_sha256']==manifest_sha,'Different metric weights or preprocessing cannot be merged')
        require(float(row['E'])==2*n,'Five-method energy or N accounting differs')
        if row.get('actual_energy') not in (None,''):
            require(abs(float(row['actual_energy'])-2*n)<.02,'Measured per-frame energy differs')
        require(all(np.isfinite(float(row[k])) for k in (*METRICS,'resnet50_top1_probability','dino_mismatched')),
                'A required metric is missing or nonfinite')
        prediction=int(row['resnet50_prediction']);source_prediction=int(row['resnet50_source_prediction'])
        probability=float(row['resnet50_top1_probability'])
        require(0<=prediction<1000 and 0<=source_prediction<1000 and 0<=probability<=1,'Invalid independent classifier output')
        error=int(prediction!=source_prediction)
        require(float(row['resnet50_top1_label'])==int(prediction==int(record['class_index']))
                and float(row['resnet50_top1_source_prediction'])==1-error
                and float(row['semantic_error'])==error
                and float(row['confidently_wrong'])==int(probability>=.5 and error), 'Classifier identities/semantic error disagree')
        require(abs(float(row['dino_specificity'])-(float(row['dino_cosine'])-float(row['dino_mismatched'])))<1e-10,
                'Mismatch-specificity metric differs from its measured components')
        baseline=(row['reference_sha256'],source_prediction)
        require(i not in baselines or baselines[i]==baseline,'Methods did not score the same exact original float RGB')
        baselines[i]=baseline
        if n==2048 and m=='D_U':
            require(row['method']=='D_U_whole_N2048_v1','Paid-class historical N2048 is not an unconditional replacement')
    return index


def load_analysis(root):
    path=Path(root)/'experiments/unified-metrics-20261002/analysis.py'
    registered=c.read(Path(root)/'outputs/UNIFIED-METRICS-20261002/supervisor_registration.json')
    require(registered['source_bindings'].get(str(path))==c.sha(path),'Original paired bootstrap source changed')
    spec=importlib.util.spec_from_file_location('_own_controls_report_original_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(module.SEED==20261002 and module.REPLICATES==10000,'Original source bootstrap definition differs')
    return module,{str(path):c.sha(path)}


def aggregate(indexed,analysis,stopped=lambda:False,progress=lambda *a,**k:None):
    """Use original Bootstrap directly, preserving its draw seed and source unit."""
    bootstrap=analysis.Bootstrap();summary=[];source_rows=[];paired=[];means={}
    for n in c.BUDGETS:
        for snr in c.SNRS:
            if stopped():raise c.PauseRequested('Stop requested at a completed statistical group')
            for method in FAMILIES:
                for metric in METRICS:
                    values=np.asarray([np.mean([float(indexed[i,n,snr,seed,method][metric]) for seed in c.SEEDS])
                                       for i in range(100)],dtype=np.float64)
                    means[n,snr,method,metric]=values
                    meta=dict(N=n,snr_db=snr,method=METHODS[n][FAMILIES.index(method)],comparison_family=method,
                              metric=metric,direction='lower' if metric in LOWER else 'higher')
                    summary.append(dict(**meta,**bootstrap.interval(values),n_frames=300,
                        aggregation='mean_noise_within_source_then_equal_source_mean'))
                    source_rows.extend(dict(**meta,source_index=i,source_id=indexed[i,n,snr,2001,method]['source_id'],
                                            n_noise=3,value=float(v)) for i,v in enumerate(values))
            # Every pair is declared by method order, before looking at metric values.
            for b,a in itertools.combinations(FAMILIES,2):
                for metric in METRICS:
                    paired.append(dict(N=n,snr_db=snr,family_A=a,family_B=b,
                        method_A=METHODS[n][FAMILIES.index(a)],method_B=METHODS[n][FAMILIES.index(b)],
                        metric=metric,direction='lower' if metric in LOWER else 'higher',
                        **bootstrap.interval(means[n,snr,a,metric]-means[n,snr,b,metric]),delta='A_minus_B',
                        comparison_scope=('same_source_noise_seed_and_identical_measured_full_waveform' if
                            (a,b)==('HiFiDiffCom','SwinJSCC') else 'same_source_means; physical_noise_and_waveform_identity_not_assumed'),
                        aggregation='paired_difference_after_mean_noise_within_source',n_frames_per_method=300))
            progress('STATISTICS',groups_complete=len(summary)//len(METRICS),groups_expected=30)
    return summary,source_rows,paired


def reference_gate(root):
    folder=Path(root)/'outputs/EXTERNAL-COMPARISON-20261004/reference_metrics_cpu_full'
    path=folder/'completion.json';done=c.read(path);rp=folder/'registration.json';reg=c.read(rp)
    require(done.get('status')=='SIX_REFERENCE_METRICS_COMPLETE' and done.get('sources')==100
            and done.get('rows')==600 and done.get('conditions')==REFERENCE_CONDITIONS
            and done.get('bootstrap_seed')==20261002 and done.get('bootstrap_replicates')==10000
            and done.get('noise_triplication') is False and done.get('holdout_access') is False
            and done.get('training_updates')==done.get('policy_selection_updates')==0,
            'Required six reference-level conditions are incomplete')
    require(done['registration_sha256']==c.sha(rp) and reg['conditions']==REFERENCE_CONDITIONS
            and reg['reference_only'] is True,'Metric-reference registration differs')
    c.verify(done['outputs']);c.verify(reg['model_bindings']);c.verify(reg['source_bindings'])
    return done,{str(path):c.sha(path),str(rp):c.sha(rp),**done['outputs'],**reg['model_bindings'],**reg['source_bindings']}


def step0_gate(root):
    folder=Path(root)/'outputs/EXTERNAL-COMPARISON-20261004/step0_statistics'
    path=folder/'completion.json';value=c.read(path)
    require(value.get('status')=='STEP0_STATISTICS_COMPLETE' and value.get('rows')==6600
            and value.get('cells')==22 and value.get('bootstrap')==10000 and value.get('source_unit') is True
            and value.get('model_inference') is False,'Previously required large-bandwidth diagnostics are incomplete')
    outputs={str(Path(p) if Path(p).is_absolute() else folder/p):h for p,h in value['outputs'].items()}
    c.verify(value['input_bindings']);c.verify(outputs)
    return {str(path):c.sha(path),**value['input_bindings'],**outputs}


def cost_gate(path,external_timings):
    """Verify the complete all-five cal4 benchmark; retain full-dev timings separately."""
    external_timings=[dict(r,frames=int(r['n_frames']),source_population='development100_x3_noise',
        timing_scope='actual_development_execution',same_population_all5=False,batch_size=1,
        warmup_per_method_N_per_invocation='not_preregistered_for_main_development') for r in external_timings]
    if path is None:
        missing=[dict(N=n,snr_db=s,method=METHODS[n][j],comparison_family=FAMILIES[j],status='NOT_MEASURED_COMPARABLY',
                      reason='Source replay wall time includes TX, cache reuse and metrics; not a receiver measurement')
                 for n in c.BUDGETS for s in c.SNRS for j in range(3)]
        return external_timings+missing,{},False
    from own_controls_cost_common import verify_report
    rows,bindings=verify_report(path)
    expected={(n,s,m) for n in c.BUDGETS for s in c.SNRS for m in METHODS[n]}
    require(len(rows)==30 and {(int(r['N']),int(r['snr_db']),r['method']) for r in rows}==expected,
            'All-five same-population receiver grid differs')
    rows=[dict(r,comparison_family=family(r['N'],r['method']),timing_scope='fixed_cal4_comparable_benchmark') for r in rows]
    return rows+external_timings,bindings,True


def completion_claim(row_count,figure_cells,reference_complete,cost_complete):
    require(row_count==9000 and figure_cells==480,'Incomplete matched comparison cannot produce a final receipt')
    missing=[]
    if not reference_complete:missing.append('six_metric_reference_conditions')
    if not cost_complete:missing.append('matched_own_receiver_cost')
    return not missing,missing


def verify_figure_population(indexed,population):
    require(set(population)==set(external.FIXED),'Figure source selection differs')
    for i,(target,images) in population.items():
        require(c.rgb_sha(target)==indexed[i,1024,1,2001,'P']['reference_sha256'],'Figure original pixels differ')
        expected={(n,s,seed,m) for n in c.BUDGETS for s in c.SNRS for seed in (2001,) for m in FAMILIES}
        require(set(images)==expected,'Fixed example misses a method/N/SNR')
        for (n,s,seed,m),image in images.items():
            require(c.rgb_sha(image)==indexed[i,n,s,seed,m]['image_sha256'],'Figure differs from scored float image')


def plot_fixed(output,indexed,population,labels,stopped,progress,policies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    verify_figure_population(indexed,population)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    def class_label(value):
        v=int(value);return f'{labels[v]} [{v}]' if labels else f'class {v}'
    figures=[];cells=[]
    for n in c.BUDGETS:
        for snr in c.SNRS:
            pdf_path=output/f'five_methods_N{n}_SNR{snr}.pdf';pages=[]
            with PdfPages(pdf_path,metadata={'Title':f'Five methods N{n} SNR{snr}', 'CreationDate':None,'ModDate':None}) as pdf:
                for page in range(4):
                    if stopped():raise c.PauseRequested('Stop requested at completed fixed-example page')
                    indices=external.FIXED[page*4:(page+1)*4]
                    fig,axes=plt.subplots(4,6,figsize=(21.6,15.8),dpi=120)
                    fig.subplots_adjust(left=.025,right=.985,bottom=.15,top=.895,wspace=.10,hspace=.58)
                    titles=['Original',*[policy_title(n,snr,m,policies) for m in FAMILIES]]
                    for j,title in enumerate(titles):axes[0,j].set_title(title,fontsize=13,pad=13,weight='bold')
                    for pos,i in enumerate(indices):
                        target,images=population[i];first=indexed[i,n,snr,2001,'P']
                        for j,image in enumerate([target]+[images[n,snr,2001,m] for m in FAMILIES]):
                            ax=axes[pos,j];ax.imshow(image.transpose(1,2,0),interpolation='nearest',vmin=0,vmax=1)
                            ax.set_xticks([]);ax.set_yticks([])
                            for spine in ax.spines.values():spine.set_visible(False)
                            if j==0:
                                caption=f"Source {i:02d} | true: {class_label(first['true_class_index'])}\nR50: {class_label(first['resnet50_source_prediction'])}"
                            else:
                                m=FAMILIES[j-1];row=indexed[i,n,snr,2001,m]
                                caption=(f"PSNR {float(row['psnr_db']):.2f} | LPIPS {float(row['lpips_alex']):.3f}\n"
                                    f"DINOv2-L {float(row['dinov2_vitl14_cosine']):.3f}\n"
                                    f"R50: {class_label(row['resnet50_prediction'])}\n"
                                    f"{'same' if float(row['resnet50_top1_source_prediction']) else 'different'} | p={float(row['resnet50_top1_probability']):.3f}")
                                cells.append(dict(source_index=i,N=n,snr_db=snr,noise_seed=2001,comparison_family=m,
                                    method=row['method'],image_sha256=row['image_sha256'],reference_sha256=row['reference_sha256'],
                                    replay_row_id=row['replay_row_id'],page=page+1,psnr_db=float(row['psnr_db']),
                                    lpips_alex=float(row['lpips_alex']),dinov2_vitl14_cosine=float(row['dinov2_vitl14_cosine']),
                                    resnet50_prediction=int(row['resnet50_prediction'])))
                            ax.set_xlabel('\n'.join(textwrap.fill(line,36) for line in caption.splitlines()),fontsize=9,labelpad=7)
                    fig.suptitle(f'N = {n} | SNR = {snr} dB | fixed sources {page*4+1}–{page*4+4} / 16',fontsize=18,y=.974,weight='bold')
                    fig.text(.5,.941,'Same source, absolute complex-symbol budget and E = 2N; original source-float metric target; no class-conditioned generation',ha='center',fontsize=11)
                    fig.text(.5,.030,'Noise seed 2001; each method retains its registered noise namespace. Swin and HiFi alone share an identical received waveform.\n'
                        'DINOv2-L = ViT-L/14. ResNet-50 agreement refers to its prediction on the original. Samples were fixed before evaluation.\n'
                        'N1024 D_U pays an ignored class field; N2048 D_U is the newly registered M1-header q=0 policy. Different external decoders are explicit.',ha='center',fontsize=10,linespacing=1.35)
                    png=output/f'five_methods_N{n}_SNR{snr}_page{page+1}.png'
                    fig.savefig(png,dpi=120,facecolor='white');pdf.savefig(fig,facecolor='white');plt.close(fig)
                    pages.append(dict(path=str(png),sha256=c.sha(png),source_indices=list(indices)))
                    progress('FIGURES',pages_complete=len(figures)*4+page+1,pages_expected=24)
            figures.append(dict(N=n,snr_db=snr,pdf=str(pdf_path),pdf_sha256=c.sha(pdf_path),png_pages=pages))
    return figures,cells


def plot_resource(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected={(r['N'],r['snr_db'],r['comparison_family'],r['metric']):r for r in summary}
    specs=(('semantic_error','Semantic error'),('confidently_wrong','Confidently wrong'),
           ('dinov2_vitl14_cosine','DINOv2-L/14'),('lpips_alex','LPIPS-Alex'))
    fig,axes=plt.subplots(4,3,figsize=(14,14),sharex=True)
    for j,snr in enumerate(c.SNRS):
        for i,(metric,title) in enumerate(specs):
            ax=axes[i,j]
            for offset,method in enumerate(FAMILIES):
                rr=[selected[n,snr,method,metric] for n in c.BUDGETS]
                y=np.asarray([r['mean'] for r in rr]);low=np.asarray([r['ci_low'] for r in rr]);high=np.asarray([r['ci_high'] for r in rr])
                # Only measured points are drawn; no interpolation of a crossing.
                ax.errorbar(np.asarray(c.BUDGETS)+(offset-2)*12,y,yerr=[np.maximum(0,y-low),np.maximum(0,high-y)],
                            marker='o',linestyle='none',capsize=3,label=LABELS[method],ms=5)
            ax.set_title(f'{title} | {snr} dB');ax.grid(alpha=.2);ax.set_xticks(c.BUDGETS)
            if i==3:ax.set_xlabel('Total complex symbols N (points slightly offset for readability)')
    handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,ncol=3,loc='upper center',bbox_to_anchor=(.5,.99))
    fig.subplots_adjust(left=.07,right=.98,bottom=.075,top=.92,hspace=.40,wspace=.28)
    fig.text(.5,.021,'100 sources × 3 noise seeds; original source-bootstrap 95% intervals. No unmeasured N is filled.\n'
        'Digital VAR at N2048 uses the new no-class q=0 PHY; differences across N do not isolate bandwidth alone.',ha='center',fontsize=10)
    outputs={}
    for suffix in ('png','pdf'):
        path=output/('quality_resource_measured_points.'+suffix);fig.savefig(path,dpi=150,facecolor='white');outputs[str(path)]=c.sha(path)
    plt.close(fig);return outputs


def text_report(summary,paired,costs,cost_complete,figures,source_info):
    lookup={(r['N'],r['snr_db'],r['comparison_family'],r['metric']):r for r in summary}
    lines=['# Matched external comparison: P, digital VAR, M1, SwinJSCC and HiFi-DiffCom','',
        'The quality comparison contains all five registered methods at N1024 and N2048, SNR 1/7/13 dB: '
        '100 sources × three noise seeds, 9,000 actual reconstruction rows. '+
        ('A separately registered receiver benchmark is also complete.' if cost_complete else
         '**The full plan remains open: comparable receiver timing for P, digital VAR and M1 is still missing.**'),'',
        '## Comparison contract','',
        'All methods use the same 256×256 development originals and exact float RGB metric targets, N complex symbols and E=2N. '
        'Noise seeds are 2001/2002/2003. Each method retains its registered physical noise namespace. '
        'Only Swin and HiFi use the identical measured received waveform; cross-family pairs compare source means.', '',
        'P/D_U/M1 share the original frozen visual representation and Dc. External methods keep their own trained encoder/decoder. '
        'Swin is the newly trained shared MSE model; HiFi uses its received paid mask/power and a frozen unconditional ADM prior with the complete author-derived schedule. '
        'No external output is described as a same-Dc mechanism result.', '',
        '**Protocol distinction:** historical N1024 D_U pays ten class bits but unconditional completion ignores the decoded class. '
        'N2048 `D_U_whole_N2048_v1` is a new calibration-selected q=0 policy with the M1 no-class header and m4–m8 grid. '
        'Its 68-symbol header, CRC/tails and rate matching are charged. It is not the old paid-class N2048 chain. '
        'M1 N1024 keeps the frozen entropy policy; N2048 uses the newly registered entropy calibration. No development value selected any policy.', '',
        'Old native metrics on uint8/255 targets are preserved in provenance fields. This main table recomputes every metric on the shared inherited P2048 source-float target. '
        'The original DINO is DINOv2 ViT-S/14; the added DINO column is DINOv2 ViT-L/14. CLIP is ViT-L/14. '
        'Independent ImageNet ResNet-50 gives both true-label accuracy and agreement with its original-image prediction. '
        'Neither DINO nor CLIP is the transmission backbone. LPIPS is not independent of historical training and selection: '
        'P and the currently frozen stage-A Dc were trained/selected with MSE and LPIPS. The current Dc selection is '
        'stage_A_v1, step 38,000, using mixture image utility; the earlier m8/m9 decoder experiment that used DINO is a different model. '
        'Current M1/whole policy selection uses PSNR feasibility and LPIPS ranking, with failure-rate tie rules; DINO-S is not its ranking objective. '
        'DINO-S did enter feasibility gates in other historical experiments, including M2, so it is not described as never used for project selection. '
        'CLIP-L/14, DINOv2-L/14 and ResNet-50 were not used to train or select the communication models and policies in this comparison. '
        'DINO-S and DINO-L share a model family, and DreamSim also contains DINO/CLIP-family components; these evaluation spaces are not all statistically independent.', '',
        'Intervals average the three noise outcomes within each source, then use the unchanged 10,000-draw source bootstrap (seed 20261002). '
        'Semantic error is classifier disagreement with the original; confidently wrong additionally requires top-1 probability ≥0.5. '
        'These are classifier diagnostics, not human labels of hallucination. Holdout is unused; KID/FID remain deferred.', '',
        '## Frozen operating policies and paid resources','',
        'The following m/K/order values come from authenticated calibration policy files and are checked against every original per-frame action. '
        'N1024 is not reselected. K=0 transmits whole scales only: it is not evidence of a partial-scale or entropy-order advantage. '
        'At N2048 the D_U and M1 policies share the same PHY; if their m/K/order match, all 300 output hashes must match. '
        'N1024 D_U and M1 use different header protocols even when both have K=0.','']
    metadata=source_info['protocol_metadata']
    lines+=external.markdown_table(['N','SNR','Policy','m','K','Order','Operating mode','Same RGB as D_U / 300'],
        [[r['N'],r['snr_db'],r['comparison_family'],r['m'],r['K'],r['order'],r['operating_mode'],r['same_rgb_as_D_U_frames']]
         for r in metadata['policies']])+['','All complex-symbol counts include the complete paid header and data protection.','']
    lines+=external.markdown_table(['N','Method','Body N','Header N','Total E','Charged metadata'],
        [[r['N'],LABELS[r['comparison_family']],r['body_N'],r['header_N'],r['E'],r['metadata']]
         for r in metadata['resources']])+['','## Training scope and stopping limits','',
        'P1024 and P2048 retain different original training seeds and selected checkpoints. Their cross-budget difference is not a pure bandwidth intervention. '
        'P512 is listed only to preserve the earlier result and stopping qualification; it is not added to this N1024/N2048 main table. '
        'A true budget_truncated flag remains explicit and does not establish convergence. The original P512/P1024 40k results are unchanged. '
        'Swin uses one random-initialized MSE-only shared model. Its 20k milestone is diagnostic, with calibration every 2,500 updates; '
        'the registered earliest plateau stop is 80k after two learning-rate reductions, and the cap is 240k. '
        'HiFi adds no generative-model training.','']
    lines+=external.markdown_table(['Model','N','Training seed','Selected step','Completed step','Registered cap','budget_truncated','Stopping record'],
        [[r['method'],r['N'],r['training_seed'],r['selected_step'],r['completed_step'],r['maximum_steps'],r['budget_truncated'],r['stop_reason']]
         for r in metadata['training']])+['',
        '[Policy ledger](selected_operating_policies.csv) · [resource ledger](paid_resource_ledger.csv) · '
        '[training provenance](training_scope.csv) · [complete protocol metadata](protocol_metadata.json)','']
    for title,names in [('Quality and semantic similarity',('psnr_db','lpips_alex','dinov2_vitl14_cosine','clip_image_cosine')),
                        ('Perception and mismatch specificity',('dists','dreamsim','ms_ssim','dino_cosine','dino_specificity')),
                        ('Classification and semantic errors',('resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong'))]:
        lines += ['## '+title,'','Mean [95% source-bootstrap interval].','']
        table=[[n,s,LABELS[m]]+[external.interval(lookup[n,s,m,k]) for k in names]
               for n in c.BUDGETS for s in c.SNRS for m in FAMILIES]
        lines+=external.markdown_table(['N','SNR','Method']+[METRICS[k] for k in names],table)+['']
    lines+=['## Paired comparisons','',
        'All ten unordered method pairs were fixed by method identity before scoring. Deltas are A minus B. '
        'Positive favors A for similarity/accuracy; negative favors A for error/distance. '
        'The [complete paired table](metrics_paired_intervals.csv) contains all 780 metric comparisons. '
        'No multiplicity-adjusted or universal superiority claim is inferred from these exploratory development intervals.','']
    focus=[r for r in paired if r['family_A'] in ('SwinJSCC','HiFiDiffCom') and r['family_B']=='M1'
           and r['metric'] in ('psnr_db','lpips_alex','dinov2_vitl14_cosine','semantic_error')]
    lines+=external.markdown_table(['N','SNR','A minus M1','Metric','Delta [95% CI]'],
        [[r['N'],r['snr_db'],LABELS[r['family_A']],METRICS[r['metric']],external.interval(r)] for r in focus])+['',
        'A DINO advantage must be interpreted with PSNR, LPIPS, semantic error and receiver cost. '
        'The results do not turn a single winning metric or example into a claim of overall superiority.','',
        '## Quality and resource','',
        '![Measured quality-resource points](figures/quality_resource_measured_points.png)','',
        'Only measured N1024/N2048 points are plotted. A zero-crossing or threshold between them is not estimated. '
        'Digital-PHY changes and separately trained P checkpoints prevent attributing their cross-N difference solely to bandwidth. '
        'Earlier large-bandwidth Step0 figures retain their original paid-class/D0/Dc and legacy-protocol labels and remain a separate diagnostic.','',
        '## Receiver cost','',
        'The [receiver-cost table](receiver_cost.csv) keeps actual measured boundaries and populations explicit. '
        'External measurements include paid-header decoding and RGB reconstruction, GPU synchronization, and failed-header outputs; they exclude TX/model loading/scoring. '
        'Own source replay wall time includes caching and scoring and is never substituted for receiver cost. '+
        ('All five receivers were additionally timed on the same four preregistered calibration sources (indices 0,1,2,3), '
         'N1024/N2048, SNR1/7/13, noise seed4101: 120 outputs, batch1, one measured call per output. '
         'Each process first ran one complete warmup per method and N at source0/SNR13; HiFi warmups and measurements use the full sampler. '
         'The 30 calibration groups share the physical GPU UUID and driver; the original VAR and Swin software environments remain labeled. '
         'Four observations per group provide a small cost benchmark, not a population-level speed estimate. '
         'The separate 100-source development timings are retained as execution evidence, without treating their means as paired speed ratios against calibration. '
         'No minimum-deployment GPU-memory claim is made.' if cost_complete else
         'The missing own-method rows are explicitly marked NOT_MEASURED_COMPARABLY; full_plan_complete remains false.'),'',
        '## Fixed 16 examples','',
        'Original sources 0,25,50,75 and the twelve preregistered additional sources are unchanged. '
        'Every panel uses seed 2001. SNR7 is the main view; SNR1 and SNR13 are supplements. '
        'Images come directly from authenticated float caches, and captions come from the same scored rows.','']
    for f in sorted(figures,key=lambda x:(x['N'],{7:0,1:1,13:2}[x['snr_db']])):
        lines += [f"### N{f['N']} / {f['snr_db']} dB",'',
            f"[Four-page PDF](figures/{Path(f['pdf']).name}) · "+' · '.join(f"[PNG {i+1}](figures/{Path(p['path']).name})" for i,p in enumerate(f['png_pages'])), '',
            f"![Fixed examples page 1](figures/{Path(f['png_pages'][0]['path']).name})",'']
    lines+=['## Metric reference levels and provenance','',
        '[Six preregistered reference conditions](metric_reference_levels.csv) give identity, unrelated-image, same-class and fixed mild-degradation levels. '
        'Their original CPU backend remains labeled separately from CUDA main scoring; they are not decision thresholds. '
        'Identity PSNR is explicitly infinite, never an invented finite number.','',
        '[All per-frame scores](metrics_per_frame.csv) · [source means](metrics_per_source.csv) · '
        '[summary intervals](metrics_summary.csv) · [row provenance](row_provenance.json) · [figure proofs](figures_manifest.json)','',
        'Optional class-prediction heads, larger VAR, new SGD/DiffJSCC/ADJSCC/DeepJSCC runs and LPIPS fine-tuning are outside the narrowed mandatory comparison. '
        'DiT-JSCC remains related work where executable code/weights were unavailable.','']
    return '\n'.join(lines)


def report_locations(root,cost_complete):
    root=Path(root)
    return (root/'outputs/EXTERNAL-COMPARISON-20261004/final_comparison'/('final_stage' if cost_complete else 'quality_stage'),
            root/'results/external_comparison_20261004'/('final_comparison' if cost_complete else 'five_method_quality'))


def load_figure_population(root,config,indexed,reconstruction,own_registration):
    # Public scorer helper authenticates originals and inherited P2048 before
    # returning any image. This path is CPU-only and never reconstructs an image.
    from own_controls_score import load_source_images
    population={};bindings={};out=Path(config['output'])
    for i in external.FIXED:
        target,images,proof=load_source_images(root,i,own_registration);bindings.update(proof)
        pairs={(n,s,seed,family(n,m)):image for (n,s,seed,m),image in images.items() if seed==2001}
        path=out/'source_checkpoints'/f'{i:04d}.json'
        require(reconstruction['outputs'].get(str(path))==c.sha(path),'Figure external source checkpoint changed')
        checkpoint=c.validate_source(c.read(path),reconstruction['binding'],i)
        receipt=checkpoint['float_reconstructions'];bindings[str(path)]=c.sha(path)
        with np.load(receipt['path'],allow_pickle=False) as archive:
            ext_target=c.pixels(archive['source_rgb']);ext_images=archive['images'];slots=archive['image_slots'].tolist()
            require(np.array_equal(target,ext_target),'Own/external figure reference targets differ')
            for row,slot in zip(checkpoint['rows'],slots):
                if int(row['noise_seed'])==2001:
                    key=(int(row['N']),int(row['snr_db']),2001,family(row['N'],row['method']))
                    pairs[key]=c.pixels(ext_images[slot]).copy()
        bindings[receipt['path']]=receipt['sha256'];population[i]=(c.pixels(target),pairs)
    verify_figure_population(indexed,population)
    return population,bindings


def generate(config_path,fixed_path,class_names,cost_path,progress,stopped):
    config=c.read(config_path)
    from external_eval import validate_config
    validate_config(config);root=Path(config['root']);ext_out=Path(config['output']);ext_result=Path(config['result'])
    own_out=root/'outputs/EXTERNAL-COMPARISON-20261004/own_controls';own_result=root/'results/external_comparison_20261004/own_controls'
    ext_done_path=ext_out/'completion.json';own_done_path=own_out/'score_completion.json'
    ext_done=c.read(ext_done_path);own_done=c.read(own_done_path)
    require(ext_done.get('status')=='EXTERNAL_EVALUATION_COMPLETE' and ext_done.get('rows')==3600
            and ext_done.get('sources')==100 and ext_done.get('physical_frames')==1800
            and ext_done.get('synthetic') is False and ext_done.get('sampler_step_limit') is None
            and ext_done.get('selection_uses_development') is False and ext_done.get('holdout_access') is False,
            'Both full external receivers must finish before comparison')
    require(own_done.get('status')=='OWN_CONTROLS_METRICS_COMPLETE' and own_done.get('rows')==5400
            and own_done.get('sources')==100 and own_done.get('synthetic') is False
            and own_done.get('selection_uses_development') is False and own_done.get('holdout_access') is False,
            'All matched P, unconditional digital and M1 metrics must finish')
    progress('VERIFYING_COMPLETED_INPUTS')
    c.verify(ext_done['bindings']);c.verify(ext_done['outputs']);c.verify(own_done['bindings']);c.verify(own_done['outputs'])
    own_reg_path=own_result/'metrics_registration.json';ext_reg_path=ext_result/'metrics_registration.json'
    own_reg=c.read(own_reg_path);ext_reg=c.read(ext_reg_path)
    require(own_done['metrics_registration_sha256']==c.sha(own_reg_path)
            and ext_done['outputs'].get(str(ext_reg_path))==c.sha(ext_reg_path),'Scoring registration lacks completed proof')
    require(own_reg['metric_evaluator_identity']==ext_reg['metric_evaluator_identity']==own_done['metric_evaluator_identity']==ext_done['metric_evaluator_identity']
            and own_reg['modelmanifest_sha256']==ext_reg['modelmanifest_sha256']
            and own_reg['numerical_runtime']==ext_reg['numerical_runtime']==own_done['numerical_runtime']==ext_done['numerical_runtime'],
            'Scoring assets, evaluator or numerical flags differ across methods')
    reconstruction_path=ext_out/'reconstruction_completion.json';reconstruction=c.read(reconstruction_path)
    require(c.sha(reconstruction_path)==ext_done['reconstruction_completion_sha256'],'External reconstruction identity changed')
    c.verify(reconstruction['bindings']);c.verify(reconstruction['outputs'])
    source_registration_path=ext_out/'reconstruction_registration.json';source_registration=c.read(source_registration_path)
    require(c.identity(source_registration)==reconstruction['binding'],'External reconstruction registration differs')
    records=source_registration['source_identity'];external.fixed_selection(c.read(fixed_path),records)
    require(len(own_reg['source_identity'])==100 and all(
        {k:r[k] for k in ('image_id','preprocessing_id','class_index')}==
        {k:records[i][k] for k in ('image_id','preprocessing_id','class_index')}
        for i,r in enumerate(own_reg['source_identity'])),'Scorer population mappings differ')
    labels=c.read(class_names) if class_names else None
    require(labels is None or isinstance(labels,list) and len(labels)==1000 and all(isinstance(x,str) for x in labels),
            'Class names require the ordered 1000-label ImageNet mapping')
    own_rows=external.csv_rows(own_result/'metrics_per_frame.csv');ext_rows=external.csv_rows(ext_result/'metrics_per_frame.csv')
    indexed=validate_rows(own_rows,ext_rows,records,own_reg['modelmanifest_sha256'])
    metadata,metadata_bindings=protocol_metadata(root,indexed,own_reg,reconstruction)
    analysis,analysis_bindings=load_analysis(root)
    _,reference_bindings=reference_gate(root);step0_bindings=step0_gate(root)
    cost_rows,cost_bindings,cost_complete=cost_gate(cost_path,external.timing_table(ext_rows))
    out,result=report_locations(root,cost_complete);out.mkdir(parents=True,exist_ok=True);result.mkdir(parents=True,exist_ok=True)
    deps=['own_controls_report.py','own_controls_report_tests.py','external_eval_report.py','external_eval_common.py',
          'own_controls_score.py','own_controls_score_common.py','own_controls_cost_common.py']
    inputs={str(HERE/name):c.sha(HERE/name) for name in deps}
    inputs.update({str(p):c.sha(p) for p in (config_path,fixed_path,ext_done_path,own_done_path,own_reg_path,ext_reg_path,
                                         reconstruction_path,source_registration_path)})
    inputs.update({**ext_done['outputs'],**own_done['outputs'],**analysis_bindings,**reference_bindings,**step0_bindings,**cost_bindings,**metadata_bindings})
    if class_names:inputs[str(class_names)]=c.sha(class_names)
    registration=dict(status='MATCHED_FIVE_METHOD_REPORT_REGISTERED',input_bindings=inputs,source_indices=list(external.FIXED),
        source_identity=records,methods_by_N={str(n):list(METHODS[n]) for n in c.BUDGETS},
        budgets=list(c.BUDGETS),snrs=list(c.SNRS),noise_seeds=list(c.SEEDS),figure_noise_seed=2001,
        bootstrap_seed=analysis.SEED,bootstrap_replicates=analysis.REPLICATES,planned_pairs='all10unordered_method_pairs_per_N_SNR',
        sample_selection_uses_quality=False,holdout_access=False,synthetic=False,receiver_cost_complete=cost_complete,
        metric_evaluator_identity=own_reg['metric_evaluator_identity'],numerical_runtime=own_reg['numerical_runtime'],
        protocol_metadata=metadata)
    reg_path=out/'registration.json';c.seal(reg_path,registration)
    completion_path=out/'completion.json'
    if completion_path.exists():
        done=c.read(completion_path)
        require(done['registration_sha256']==c.sha(reg_path),'Existing comparison registration differs')
        c.verify(done['inputs']);c.verify(done['outputs']);progress('COMPLETE',full_plan_complete=done['full_plan_complete']);return done
    if stopped():raise c.PauseRequested('Stop requested before comparison aggregation')
    summary,source_values,paired=aggregate(indexed,analysis,stopped,progress)
    population,figure_bindings=load_figure_population(root,config,indexed,reconstruction,own_reg)
    figures_dir=result/'figures';figures_dir.mkdir(exist_ok=True)
    figures,cells=plot_fixed(figures_dir,indexed,population,labels,stopped,progress,metadata['policies'])
    resource_outputs=plot_resource(figures_dir,summary)
    all_rows=[dict(indexed[key],comparison_family=key[-1]) for key in sorted(indexed)]
    tables=[('metrics_per_frame.csv',all_rows),('metrics_summary.csv',summary),('metrics_per_source.csv',source_values),
            ('metrics_paired_intervals.csv',paired),('receiver_cost.csv',cost_rows),
            ('selected_operating_policies.csv',metadata['policies']),('paid_resource_ledger.csv',metadata['resources']),
            ('training_scope.csv',metadata['training'])]
    references=external.csv_rows(root/'outputs/EXTERNAL-COMPARISON-20261004/reference_metrics_cpu_full/summary.csv')
    tables.append(('metric_reference_levels.csv',references))
    outputs={}
    for name,rows in tables:
        path=result/name;c.write_csv(path,rows);outputs[str(path)]=c.sha(path)
    provenances=[]
    for key in sorted(indexed):
        row=indexed[key];origin={k:row[k] for k in ('image_archive','image_archive_sha256','image_array_key','image_slot',
                'original_replay_row_id','original_scientific_row_sha256','source_origin','original_row_json') if k in row}
        provenances.append(dict(source_index=key[0],N=key[1],snr_db=key[2],noise_seed=key[3],comparison_family=key[4],
            method=row['method'],image_sha256=row['image_sha256'],reference_sha256=row['reference_sha256'],
            replay_row_id=row['replay_row_id'],metric_registration_sha256=c.sha(ext_reg_path if key[4] in FAMILIES[3:] else own_reg_path),
            original_identity=origin))
    proof_path=result/'row_provenance.json';c.seal(proof_path,dict(status='ALL9000_ROWS_PROVENANCE',rows=provenances,
                native_metric_target='kept as original fields; not reused for main common-target scores',input_bindings=inputs))
    metadata_path=result/'protocol_metadata.json';c.seal(metadata_path,dict(metadata,input_bindings=metadata_bindings,
        policy_inferred_from_development=False,all_3600_digital_actions_match_frozen_policy=True))
    manifest_path=result/'figures_manifest.json';c.seal(manifest_path,dict(status='VERIFIED_FIXED16_FIVE_METHOD_FIGURES',
        figures=figures,cells=cells,source_indices=list(external.FIXED),noise_seed=2001,input_bindings=figure_bindings,
        original_float_pixels=True,no_metric_based_example_selection=True))
    full_complete,missing=completion_claim(len(indexed),len(cells),True,cost_complete)
    report=result/'MATCHED_COMPARISON_REPORT.md';temporary=report.with_suffix('.md.tmp')
    temporary.write_text(text_report(summary,paired,cost_rows,cost_complete,figures,registration),encoding='utf-8');os.replace(temporary,report)
    for path in (reg_path,proof_path,metadata_path,manifest_path,report):outputs[str(path)]=c.sha(path)
    outputs.update(resource_outputs)
    for f in figures:
        outputs[f['pdf']]=f['pdf_sha256'];outputs.update({p['path']:p['sha256'] for p in f['png_pages']})
    c.verify(inputs);c.verify(figure_bindings)
    done=dict(status='MATCHED_FIVE_METHOD_REPORT_COMPLETE',synthetic=False,sources=100,rows=9000,method_groups=30,
        metrics_per_group=13,paired_comparisons=780,source_bootstrap_replicates=10000,full_plan_complete=full_complete,
        remaining_requirements=missing,receiver_cost_complete=cost_complete,reference_conditions_complete=True,
        fixed_sources=16,png_figures=24,pdf_figures=6,figure_cells=480,resource_figures=2,
        sample_selection_uses_quality=False,holdout_access=False,registration_sha256=c.sha(reg_path),
        inputs={**inputs,**figure_bindings},outputs=outputs)
    c.seal(completion_path,done);progress('COMPLETE',full_plan_complete=full_complete,remaining_requirements=missing)
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--fixed-examples',type=Path,required=True)
    parser.add_argument('--class-names',type=Path)
    parser.add_argument('--receiver-cost',type=Path)
    parser.add_argument('--launch-id',default=None)
    args=parser.parse_args();config=c.read(args.config);root=Path(config['root'])
    out,_=report_locations(root,args.receiver_cost is not None);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'report.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    token=args.launch_id or str(uuid.uuid4())
    def progress(status,**values):
        c.write(out/'status.json',dict(status=status,pid=os.getpid(),launch_id=token,safe_pause_handler_installed=True,updated=time.time(),**values))
    progress('STARTING');failure=out/'failure.json'
    try:
        require(not failure.exists(),'Previous report failure requires review')
        generate(args.config.resolve(),args.fixed_examples.resolve(),args.class_names.resolve() if args.class_names else None,
                 args.receiver_cost.resolve() if args.receiver_cost else None,progress,lambda:stop[0])
    except c.PauseRequested as exc:progress('PAUSED',reason=str(exc));raise SystemExit(75)
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        if not failure.exists():c.write(failure,dict(status='FAILED_REQUIRES_REVIEW',error=repr(exc),automatic_restart_allowed=False))
        progress('FAILED_REQUIRES_REVIEW',error=repr(exc));raise


if __name__=='__main__':main()
