#!/usr/bin/env python3
"""Join only frozen same-source development outputs; preserve Decoder/evidence roles."""
import csv,json,hashlib
from pathlib import Path
from summarize_review_runs import summarize,write_csv
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/followup';OUT=BASE/'research_20260923_comparison'
def read(p):return list(csv.DictReader(p.open()))
def require_grid(rows,method_count):
    expected={(i,s,n) for i in range(100) for s in (1.,4.,7.,13.,19.) for n in (2001,2002,2003)}
    methods={r['method'] for r in rows}
    if len(methods)!=method_count:raise RuntimeError('unexpected method coverage')
    for m in methods:
        keys=[(int(r['source_index']),float(r['snr_db']),int(r['seed'])) for r in rows if r['method']==m]
        if len(keys)!=len(expected) or set(keys)!=expected:raise RuntimeError('incomplete development grid: '+m)

def canonical_adaptation_rows(rows):
    aliases={'m8_Dc_adapted':'raw_N4084_m8_Dc_adapted'};resolved={}
    key=lambda r:(int(r['source_index']),float(r['snr_db']),int(r['seed']))
    for alias,canonical in aliases.items():
        a={key(r):r for r in rows if r['method']==alias};b={key(r):r for r in rows if r['method']==canonical}
        if not a:continue
        if len(a)!=1500 or set(a)!=set(b):raise RuntimeError('historical alias grid mismatch')
        for k in a:
            if any(a[k][f]!=b[k][f] for f in ('image_id','N','E')):raise RuntimeError('historical alias scope mismatch')
        differences={m:max(abs(float(a[k][m])-float(b[k][m])) for k in a) for m in ('psnr_db','lpips_alex','dino_cosine')}
        if max(differences.values())>1e-6:raise RuntimeError('historical alias metrics differ materially')
        resolved[alias]={'canonical':canonical,'rows':1500,'maximum_metric_differences':differences}
    return [r for r in rows if r['method'] not in resolved],resolved

def main():
    OUT.mkdir(parents=True,exist_ok=False);rows=[];lineage=[]
    training=BASE/'research_20260923_training_v3'
    common_decoder=json.loads((training/'registration.json').read_text())['decoder_state_sha256']
    digital_reg=json.loads((BASE/'research_20260923_digital_strict/registration.json').read_text())
    policy_path=BASE/'research_20260923_system_policy_v2/frozen_policy.json';policy=json.loads(policy_path.read_text())
    if digital_reg['decoder_state_sha256']!=common_decoder or policy['decoder_state_sha256']!=common_decoder:raise RuntimeError('comparison Decoder mismatch')
    if digital_reg['precision']!={'matmul_tf32':False,'cudnn_tf32':False} or policy['precision']!=digital_reg['precision']:raise RuntimeError('comparison numerical protocol mismatch')
    selected=json.loads((training/'completion.json').read_text())['selected']
    model_contexts={n:r['checkpoint_sha256'] for n,r in selected.items()}
    model_contexts['original_1024_Dc']=json.loads((BASE.parent/'stage_B_v1/training/selected_enhancement1024.json').read_text())['checkpoint_sha256']
    model_contexts['calibration_frozen_resource_lookup']=hashlib.sha256(policy_path.read_bytes()).hexdigest()
    adaptive_sha=hashlib.sha256((policy_path.parent/'digital_adaptive_policies.json').read_bytes()).hexdigest()
    model_contexts.update({family+'_adaptive_m789_Dc':adaptive_sha for family in ('raw','arithmetic')})
    files=[(BASE/'research_20260923_evaluation/per_frame.csv',lambda r:True,lambda r:r['method'],'new_selected_model'),
           (BASE/'research_20260923_diagnostics/per_frame.csv',lambda r:r['method']=='actual_RX_AWGN',lambda r:'original_1024_Dc','fixed_parent_real_reevaluation'),
           (BASE/'research_20260923_system_policy_v2/per_frame.csv',lambda r:True,lambda r:r['method'],'frozen_existing_endpoint_lookup'),
           (BASE/'research_20260923_system_policy_v2/digital_adaptive_per_frame.csv',lambda r:True,lambda r:r['method'],'fresh_strict_precision_digital_same_Dc_frozen_policy')]
    for path,keep,name,evidence in files:
        lineage.append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'evidence':evidence})
        for r in read(path):
            if not keep(r):continue
            if r.get('decoder_sha256',common_decoder)!=common_decoder:raise RuntimeError('row Decoder differs from registered context')
            rows.append({'method':name(r),'source_index':int(r['source_index']),'image_id':r['image_id'],'snr_db':float(r['snr_db']),'seed':int(r['seed']),'N':4084,'E':8168,'decoder':'frozen_stageA_Dc','decoder_sha256':common_decoder,'protocol_id':'N4084:E8168:matmul_tf32_off:cudnn_tf32_off','model_context_sha256':model_contexts[name(r)],'evidence':evidence,**{k:float(r[k]) for k in ('psnr_db','lpips_alex','dino_cosine')}})
    require_grid(rows,7)
    write_csv(OUT/'per_frame.csv',rows);summarize(OUT/'per_frame.csv',OUT/'analysis');(OUT/'lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
    # Adaptation has a different Decoder and incomplete historical execution identity.
    # Keep it as explicit context, outside the same-Dc paired superiority table.
    p=BASE/'adaptation_development_v1/per_frame.csv'
    if p.exists():
        (OUT/'adaptation_context.json').write_text(json.dumps({'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'HISTORICAL_DIFFERENT_DECODER_CONTEXT_NOT_REQUALIFIED','note':'Not a complete digital policy under the adapted Decoder; no new claim of matched same-Dc superiority.'},indent=2)+'\n')
    if p.exists():
        old,aliases=canonical_adaptation_rows(read(p))
        (OUT/'adaptation_aliases.json').write_text(json.dumps(aliases,indent=2)+'\n')
        keys=[(r['method'],int(r['source_index']),float(r['snr_db']),int(r['seed'])) for r in old]
        if len(keys)!=len(set(keys)):raise RuntimeError('duplicate historical adaptation keys')
        base_ids={int(r['source_index']):r['image_id'] for r in rows}
        if any(base_ids[int(r['source_index'])]!=r['image_id'] for r in old):raise RuntimeError('adaptation source mismatch')
        context=[]
        for m in sorted({r['method'] for r in old}):
            for snr in [1.,4.,7.,13.,19.]:
                selected=[r for r in old if r['method']==m and float(r['snr_db'])==snr]
                if len(selected)!=300:raise RuntimeError('historical adaptation coverage')
                budgets={(r['N'],r['E']) for r in selected}
                if len(budgets)!=1:raise RuntimeError('historical adaptation mixes budgets')
                N,E=next(iter(budgets))
                context.append({'method':m,'N':N,'E':E,'snr_db':snr,'evidence':'historical_adaptation_not_requalified','decoder':'different_adapted_Dc' if 'adapt' in m else 'original_Dc',**{k:sum(float(r[k]) for r in selected)/len(selected) for k in ('psnr_db','lpips_alex','dino_cosine')}})
        write_csv(OUT/'adaptation_context_by_snr.csv',context)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read(OUT/'analysis/per_source_snr.csv');methods=sorted({r['method'] for r in summary});fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,metric,label in zip(axes,['psnr_db','lpips_alex','dino_cosine'],['PSNR (dB; higher better)','LPIPS (lower better)','DINO cosine (higher better)']):
        for m in methods:
            snrs=sorted({float(r['snr_db']) for r in summary});means=[sum(float(r[metric]) for r in summary if r['method']==m and float(r['snr_db'])==s)/100 for s in snrs];ax.plot(snrs,means,marker='o',label=m)
        ax.set(xlabel='SNR (dB)',ylabel=label);ax.grid(alpha=.3)
    fig.legend(*axes[-1].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=7);fig.suptitle('N4084 / E8168 / same frozen Dc — existing development, not new holdout');fig.tight_layout(rect=(0,.15,1,.94));fig.savefig(OUT/'quality_by_snr.png',dpi=180);fig.savefig(OUT/'quality_by_snr.pdf');plt.close(fig)

    import numpy as np
    diag=read(BASE/'research_20260923_diagnostics/per_frame.csv');diagnostic=[]
    require_grid(diag,5)
    for m in sorted({r['method'] for r in diag}):
        for snr in (1.,4.,7.,13.,19.):
            part=[r for r in diag if r['method']==m and float(r['snr_db'])==snr]
            if len(part)!=300:raise RuntimeError('diagnostic coverage')
            diagnostic.append({'method':m,'snr_db':snr,**{k:float(np.mean([float(r[k]) for r in part])) for k in ('psnr_db','lpips_alex','dino_cosine','normalized_latent')}})
    write_csv(OUT/'diagnostic_by_snr.csv',diagnostic)
    projection=[]
    for prefix,dirname in [('random','review_20260923_linear'),('PCA','research_20260923_pca_evaluation')]:
        path=BASE/dirname/'per_frame.csv'
        projection_context=hashlib.sha256((BASE/dirname/'projection.json').read_bytes()).hexdigest()
        lineage.append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'evidence':prefix+'_projection'})
        for r in read(path):
            projection.append({**r,'method':prefix+'_'+r['method'],'decoder_sha256':common_decoder,'protocol_id':'N4084:E8168:32_control_992_measurement:strict','model_context_sha256':projection_context})
    require_grid(projection,4)
    write_csv(OUT/'projection_per_frame.csv',projection);summarize(OUT/'projection_per_frame.csv',OUT/'projection_analysis')
    timing=read(BASE/'research_20260923_evaluation/timing.csv');cost=[]
    names=sorted({r['method'] for r in timing})
    for n in names:
        part=[r for r in timing if r['method']==n]
        if len(part)!=100:raise RuntimeError('timing coverage')
        cost.append({'method':n,'calls':len(part),**{k+'_'+stat:float(fn([float(r[k]) for r in part])) for k in ('tx_ms','rx_ms','total_ms') for stat,fn in [('mean',np.mean),('median',np.median),('p95',lambda v:np.quantile(v,.95))]}})
    write_csv(OUT/'latency_summary.csv',cost)
    per_snr=[]
    for n in names:
        for snr in (1.,4.,7.,13.,19.):
            part=[r for r in timing if r['method']==n and float(r['snr_db'])==snr]
            if len(part)!=20:raise RuntimeError('per-SNR timing coverage')
            per_snr.append({'method':n,'snr_db':snr,'calls':len(part),**{k:float(np.mean([float(r[k]) for r in part])) for k in ('tx_ms','rx_ms','total_ms')}})
    write_csv(OUT/'latency_by_snr.csv',per_snr)
    fig,ax=plt.subplots(figsize=(8,4))
    tx=[r['tx_ms_mean'] for r in cost];rx=[r['rx_ms_mean'] for r in cost]
    ax.barh(names,tx,label='TX: CPU RGB to CPU waveform');ax.barh(names,rx,left=tx,label='RX: CPU waveform to CPU RGB')
    ax.set(xlabel='Mean latency (ms)',title='Shared quality/timing execution; noise outside RX');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'endpoint_latency.png',dpi=180);fig.savefig(OUT/'endpoint_latency.pdf');plt.close(fig)
    (OUT/'lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')


    training=BASE/'research_20260923_training_v3';done=json.loads((training/'completion.json').read_text())
    curves=[]
    for path in sorted((training/'calibration').glob('*.json')):
        rec=json.loads(path.read_text())
        for name,utility in rec['summary'].items():
            curves.append({'method':name,'step':rec['step'],'utility':utility,'full_calibration_rows':rec['rows'],'calibration_seconds':rec['seconds'],'source_sha256':rec['sha256']})
    write_csv(OUT/'training_curve.csv',curves)
    parent_record=json.loads((BASE.parent/'stage_B_v1/training/selected_enhancement1024.json').read_text())
    lineage_rows=[]
    for name,rec in done['selected'].items():
        parent=0 if name=='pure_continuous' else int(parent_record['step'])
        lineage_rows.append({'method':name,'parent_updates':parent,'executed_new_updates':done['state']['updates'][name],
            'selected_new_updates':rec['step'],'selected_total_updates':parent+rec['step'],'checkpoint_sha256':rec['checkpoint_sha256'],
            'parent_checkpoint_sha256':'' if name=='pure_continuous' else parent_record['checkpoint_sha256'],'arm_key':rec['arm_key'],'registration_sha256':rec['registration_sha256'],'calibration_utility':rec['utility']})
    write_csv(OUT/'selected_lineage.csv',lineage_rows)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in sorted({r['method'] for r in curves}):
        part=[r for r in curves if r['method']==name]
        ax.plot([r['step'] for r in part],[r['utility'] for r in part],marker='o',label=name)
    ax.set(xlabel='New updates',ylabel='Full-calibration utility (log scale)',yscale='log',title='Finite training opportunity; not a convergence claim')
    ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(OUT/'training_curve.png',dpi=180);fig.savefig(OUT/'training_curve.pdf');plt.close(fig)
    reg=json.loads((training/'registration.json').read_text());cache=Path(reg['prefix_cache_root'])
    source_roles={}
    for role,count in [('train',20000),('calibration',1000)]:
        ids=[]
        for receipt in sorted((cache/'prefix_cache'/role).glob('*.json')):ids.extend(json.loads(receipt.read_text())['identity']['image_ids'])
        if len(ids)!=count or len(set(ids))!=count:raise RuntimeError('source identity coverage')
        source_roles[role]=ids
    source_roles['development']=[next(r['image_id'] for r in rows if r['source_index']==i) for i in range(100)]
    (OUT/'source_identity.json').write_text(json.dumps(source_roles,indent=2)+'\n')

if __name__=='__main__':main()
