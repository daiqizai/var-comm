"""Source-paired C statistics, calibration-only diagnostic and separate seed variation."""
import csv
from pathlib import Path
import numpy as np
from latent_enhancement.runtime import digest,write_json,verify_snapshot
from latent_enhancement_eval.runner import write_rows
from .common import ROOT,OUT,read,config
from .statistics import FrameTable,METRICS,paired
from .resource_report import read_grid,export_compact
from .C_lifecycle import SHORT

DEST=ROOT/'results/token_channel_efficiency_20260923/C_v1'

def grouped_pair(table,left,right,snrs):
    a=np.mean([table.source_values(left,s) for s in snrs],axis=0);b=np.mean([table.source_values(right,s) for s in snrs],axis=0)
    # Add image utility before resampling: MSE and LPIPS covariance is preserved.
    a=np.column_stack([a,a[:,0]+.1*a[:,2]]);b=np.column_stack([b,b[:,0]+.1*b[:,2]]);delta=a-b
    rng=np.random.default_rng(20260923);draws=[]
    for _ in range(40):draws.append(delta[rng.integers(0,len(a),(250,len(a)))].mean(1))
    ci=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    return {'left':left,'right':right,'snrs':list(snrs),'sources':len(a),'unit':'source after equally averaging registered SNR and noise','training_seed_variation_included':False,'table_sha256':table.sha256,'metrics':{k:{'mean_left':float(a[:,i].mean()),'mean_right':float(b[:,i].mean()),'difference':float(delta[:,i].mean()),'ci95':ci[:,i].tolist()} for i,k in enumerate((*METRICS,'U_image'))}}

def cross_noise_diagnostic(utilities):
    """Shape actions x source x SNR x 3 noise seeds; no deployed/oracle claim."""
    u=np.asarray(utilities,dtype=float)
    if u.ndim!=4 or u.shape[-1]!=3 or not np.isfinite(u).all():raise ValueError('complete finite action/source/SNR/three-noise grid')
    rows=[]
    for held in range(3):
        fit=u[:,:,:,np.arange(3)!=held].mean(-1);test=u[:,:,:,held]
        fixed=int(np.argmin(fit.mean((1,2))));snr=np.argmin(fit.mean(1),axis=0);source=np.argmin(fit,axis=0)
        for si in range(u.shape[2]):
            for image in range(u.shape[1]):
                rows.append({'source_index':image,'snr_index':si,'held_noise_index':held,'fixed_action':fixed,'snr_action':int(snr[si]),'source_action':int(source[image,si]),'fixed_utility':float(test[fixed,image,si]),'snr_utility':float(test[snr[si],image,si]),'source_utility':float(test[source[image,si],image,si])})
    return rows

def calibration_diagnostic():
    arrays=[];evidence={};keys=None
    for m in (6,7,8):
        folder=SHORT/'training'/f'm{m}_seed2026092304';rec=read(folder/f'selected_H{m}-V.json');p=folder/'calibration'/f"full_{rec['step']:05d}.csv";meta=read(p.with_suffix('.json'))
        if digest(p)!=meta['sha256']:raise RuntimeError('calibration diagnostic selected receipt')
        with p.open() as f:rows=[r for r in csv.DictReader(f) if r['method']==f'H{m}-V']
        mapping={(r['image_id'],float(r['snr_db']),int(r['seed'])):float(r['U_image']) for r in rows}
        ids=sorted({k[0] for k in mapping});snrs=[1,4,7,13,19];seeds=[4101,4102,4103]
        expected=[(i,s,n) for i in ids for s in snrs for n in seeds]
        if len(ids)!=1000 or len(rows)!=15000 or set(mapping)!=set(expected) or (keys is not None and expected!=keys):raise RuntimeError('diagnostic paired complete source/noise grid')
        keys=expected;arrays.append(np.array([mapping[k] for k in keys]).reshape(1000,5,3));evidence[str(p)]=digest(p)
    rows=cross_noise_diagnostic(arrays)
    for row in rows:row.update(source_id=ids[row['source_index']],snr_db=snrs[row['snr_index']],held_noise_seed=seeds[row['held_noise_index']])
    return rows,{'status':'CALIBRATION_SOURCE_INFORMED_CROSS_NOISE_DIAGNOSTIC_ONLY','deployed_selector':False,'new_holdout':False,'calibration_already_used_for_codec_selection':True,'unbiased_generalization_or_theoretical_bound_claimed':False,'actions':['H6-V','H7-V','H8-V'],'evidence_bindings':evidence}

def historical_references(sources):
    base=ROOT/'results/external_baselines';manifest=read(base/'development_manifest.json')
    # These are preserved contextual records until metric/PHY compatibility is verified.
    ids={r['image_id'] for r in manifest};same_ids=ids==set(sources)
    paths=[base/'analysis/summary.csv',base/'reference_per_frame.csv',base/'digital_development.csv',ROOT/'results/review_20260923_phase2/comparison/per_frame.csv']
    return {'status':'PRESERVED_HISTORICAL_REFERENCES_CONTEXT_ONLY','same_source_ids':same_ids,'files':{str(p):digest(p) for p in paths if p.exists()},'manifest_sha256':digest(base/'development_manifest.json'),'strict_current_scope_reuse':'NOT_RUN_PENDING_FULL_PREPROCESSING_METRIC_PHY_IDENTITY_AUDIT','methods':['perceptual_deepjscc','wetok_r3','historical digital VAR','historical N4084 mixed and resource lookup','ADJSCC/Swin natural resource points'],'HDA_DeepSC_reproduction':'NOT_RUN_NOT_CLAIMED'}

def main():
    cfg=config();folder=OUT/'C_followups/selected_grid';rows,timing,reg,done,previews=read_grid(folder)
    table=FrameTable(rows,'development',reg['sources'],reg['snrs'],reg['seeds'],sorted({r['method'] for r in rows}))
    if done['table_sha256']!=table.sha256:raise RuntimeError('C grid table identity')
    contexts={reg['context_sha256']:reg['context']};original_rows=list(rows);metadata=reg['context']['models'];choice=read(OUT/'C_followups/candidate.json');m=choice['m']
    # Reuse exactly the B evaluation objects; no duplicate P3060 training/evaluation.
    for source in [OUT/'continuous_grid_v1',OUT/'digital_grid_v1/QPSK/development']:
        other,more,oreg,odone,opreview=read_grid(source)
        if oreg['sources']!=reg['sources'] or oreg['context']['decoder_sha256']!=reg['context']['decoder_sha256']:raise RuntimeError('C/reference source or decoder mismatch')
        rows.extend(other);timing.extend(more);contexts[oreg['context_sha256']]=oreg['context']
    table=FrameTable(rows,'development',reg['sources'],reg['snrs'],reg['seeds'],sorted({r['method'] for r in rows}));pairs=set();seed=2026092304
    def name(arm,N=4084,s=seed):return f'{arm}_N{N}_seed{s}'
    for arm in ('H6-V','H7-V','H8-V'):pairs.add((name(arm),name('P4084')))
    for a in (6,7):pairs.add((name(f'H{a}-V'),name(f'H{a}-P')))
    for family in ('V','P'):pairs.add((name(f'H{m}-{family}',3060),'P3060'))
    pairs.add((name(f'H{m}-V',3060),name(f'H{m}-P',3060)))
    for s in choice['additional_seeds']:
        for other in (f'H{m}-P','H8-V','P4084'):pairs.add((name(f'H{m}-V',s=s),name(other,s=s)))
    policy=read(OUT/'digital_grid_v1/QPSK/calibration/policy.json');comparisons=[]
    for left,right in sorted(pairs):
        for scope in ([1],[4],[7],[13],[19],[1,4,7],[13,19],[1,4,7,13,19]):comparisons.append(grouped_pair(table,left,right,scope))
    for c in policy['choices']:
        if c['N']==4084:
            for arm in ('H6-V','H7-V','H8-V'):comparisons.append(grouped_pair(table,name(arm),c['method'],[c['snr_db']]))
        elif c['N']==3060:comparisons.append(grouped_pair(table,name(f'H{m}-V',3060),c['method'],[c['snr_db']]))
    DEST.mkdir(parents=True,exist_ok=True);write_json(DEST/'execution_contexts.json',contexts);export_compact(table,DEST/'development');write_json(DEST/'paired.json',comparisons)
    summary=table.summary()
    for r in summary:r['U_image']=r['mse']+.1*r['lpips_alex']
    write_rows(DEST/'summary.csv',summary);write_rows(DEST/'timing.csv',timing);write_rows(DEST/'local_reconstruction_index.csv',previews);write_json(DEST/'selected.json',metadata)
    variation=[];lookup={(r['method'],r['snr_db']):r for r in summary}
    for arm in (f'H{m}-V',f'H{m}-P','H8-V','P4084'):
        for snr in cfg['snrs_db']:
            for metric in (*METRICS,'U_image'):
                values=[lookup[(name(arm,s=s),snr)][metric] for s in [seed,*choice['additional_seeds']]]
                variation.append({'arm':arm,'snr_db':snr,'metric':metric,'training_seeds':3,'mean':float(np.mean(values)),'sample_std':float(np.std(values,ddof=1)),'min':min(values),'max':max(values),'image_bootstrap_interval':False})
    write_rows(DEST/'training_seed_variation.csv',variation)
    diag,diagmeta=calibration_diagnostic();write_rows(DEST/'source_choice_diagnostic.csv',diag);write_json(DEST/'source_choice_diagnostic.json',diagmeta)
    base_rows=[]
    for cell in done['cell_sha256']:base_rows.extend(read(folder/'cells'/cell)['base_rows'])
    for start in range(0,len(base_rows),20000):write_rows(DEST/f'received_base_without_enhancement_{start//20000:03d}.csv',base_rows[start:start+20000])
    history=historical_references(reg['sources']);write_json(DEST/'historical_references.json',history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('psnr_db','lpips_alex','dino_cosine','U_image'):
        fig,ax=plt.subplots(figsize=(7,4))
        for arm in ('H6-V','H6-P','H7-V','H7-P','H8-V','P4084'):
            ax.plot(cfg['snrs_db'],[lookup[(name(arm),s)][metric] for s in cfg['snrs_db']],'o-',label=arm)
        ax.set(xlabel='SNR (dB)',ylabel=metric,title='Selected N4084, initial training seed');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(DEST/f'quality_snr_{metric}.png',dpi=160);plt.close(fig)
    time_lookup={}
    for row in timing:time_lookup.setdefault((row['method'],row['snr_db']),[]).append(row['total_ms'])
    for metric in ('psnr_db','lpips_alex','dino_cosine','U_image'):
        for axis in ('N','time'):
            fig,axes=plt.subplots(1,5,figsize=(19,4))
            for ax,snr in zip(axes,cfg['snrs_db']):
                for label,methods in [('continuous',['P2048','P3060',name('P4084')]),(f'H{m}-V',[name(f'H{m}-V',3060),name(f'H{m}-V')]),(f'H{m}-P',[name(f'H{m}-P',3060),name(f'H{m}-P')])]:
                    x=[table.context[n]['N'] if axis=='N' else float(np.mean(time_lookup[(n,snr)])) for n in methods]
                    ax.plot(x,[lookup[(n,snr)][metric] for n in methods],'o-',label=label)
                ax.set(title=f'{snr} dB',xlabel='Complex channel uses' if axis=='N' else 'Full online TX+RX (ms)',ylabel=metric);ax.grid(alpha=.25);ax.legend(fontsize=7)
            fig.tight_layout();fig.savefig(DEST/f'C_quality_{axis}_{metric}.png',dpi=140);plt.close(fig)
    calibration=[]
    for desc in metadata.values():
        training=Path(desc['training'])
        for p in sorted((training/'calibration').glob('full_*.json')):
            meta=read(p);calibration.append({'method':desc['method'],'step':meta['step'],'utility':meta['summary'][desc['arm']],'calibration_csv_sha256':meta['sha256']})
    write_rows(DEST/'calibration_curves.csv',calibration)
    fig,ax=plt.subplots(figsize=(8,5))
    for method in sorted({r['method'] for r in calibration}):
        rr=[r for r in calibration if r['method']==method];ax.plot([r['step'] for r in rr],[r['utility'] for r in rr],label=method,alpha=.8)
    ax.set(xlabel='Total training updates',ylabel='Complete calibration training utility');ax.grid(alpha=.25);ax.legend(fontsize=5);fig.tight_layout();fig.savefig(DEST/'calibration_curves.png',dpi=160);plt.close(fig)
    report='''# Merged short-prefix selected results\n\nActual frozen selected checkpoints, complete100-source original development, five SNRs and three noise seeds. All failures remain. N3060 uses the supplemental P3060 control; no duplicate training. Source-image bootstrap and three-training-seed standard deviations are separate. P4084 first seed continues its verified10k parent while other seeds begin fresh; selected/total training identities are disclosed.\n\nReceiver-only B/C outputs remove the paid enhancement observations and retain digital failures; they are branch-contribution diagnostics, not competing full-budget systems. Cross-noise source-informed selection is a calibration diagnostic only, not an implementable selector or an unbiased theoretical bound. No new holdout or selector was opened.\n\nHistorical external/natural-resource references are preserved with hashes. Full current-scope preprocessing/metric/PHY compatibility audit is pending and they are not silently inserted into strict paired tables. HDA-DeepSC reproduction was not run and no such claim is made. The actual registered grids do not automatically imply a superior method; all signed comparisons remain.\n'''
    reportpath=ROOT/'reports/token_channel_efficiency_C_20260924.md';reportpath.write_text(report)
    write_json(DEST/'completion.json',{'status':'REAL_C_GRIDS_COMPLETE_REFERENCE_AUDIT_PENDING','synthetic':False,'new_holdout':False,'remaining':['full historical reference compatibility audit and any required reevaluation','review and publish exact remote commit'],'files':{str(p.relative_to(DEST)):digest(p) for p in DEST.rglob('*') if p.is_file() and p.name!='completion.json'},'report_sha256':digest(reportpath),'table_sha256':table.sha256})
if __name__=='__main__':main()
