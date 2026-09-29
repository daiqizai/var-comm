"""CPU publication of actual v3 B data, paired decisions and diagnostic plots."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,gzip,csv,shutil,time,io
from pathlib import Path
import numpy as np
from rx_v3_archive import pack,unpack,original_bytes
ROOT=Path(__file__).resolve().parents[2];RAW=ROOT/'outputs/RX-POSTERIOR-STEP1-20260929/revision_v3'
DEST=ROOT/'results/rx_posterior_step1_20260929/revision_v3_B'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sealed(p):
    assert Path(str(p)+'.sha256').read_text().strip()==sha(p),str(p)
    return read(p)
def table(path,rows):
    assert rows
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert not DEST.exists(),'immutable publication exists'
    done=sealed(RAW/'B_completion.json');assert done['status']=='REAL_V3_B_DEVELOPMENT_COMPLETE'
    cfg=sealed(RAW/'B_frozen_config.json');algorithm=cfg['algorithm']
    assert done['frozen_config_sha256']==sha(RAW/'B_frozen_config.json')
    assert read(RAW/'B_evaluation_supervisor_completion.json')['status']=='RX_STAGE_COMPLETE_ORIGINAL_TRAINING_REMAINS_STOPPED'
    reg=read(RAW/'B_evaluation_registration.json')
    for p,d in reg['bindings'].items():assert sha(p)==d,p
    for p,d in cfg['rho_cells'].items():assert sha(p)==d;sealed(p)
    sel=sealed(RAW/'B_selection.json')
    for p,d in sel['grid_cells'].items():assert sha(p)==d;sealed(p)
    pop=sealed(RAW/'B_population.json')['population'];frames=[];source_rows=[];bins={};archive=[];lookup={}
    profiles=['A1_decision','A2_decision','A2_fixed','V_decision','V_fixed']
    for i in range(100):
        p=RAW/'B_cells'/f'{i:03d}.json';d=sealed(p)
        assert sha(p)==done['cells'][str(p)] and d['registration_sha256']==sha(RAW/'B_evaluation_registration.json')
        assert d['source_identity']==pop[i] and d['frozen_config_sha256']==sha(RAW/'B_frozen_config.json')
        seen=set()
        for row in d['rows']:
            key=(row['level'],row['noise_seed'],row['mode'],row['profile'])
            assert key not in seen;seen.add(key)
            assert row['source_id']==pop[i]['source_id'] and row['preprocessing_sha256']==pop[i]['rgb_sha256']
            level=next(x for x in cfg['levels'] if x['name']==row['level'])
            assert row['eta']==level['eta'] and row['snr_equiv_db']==level['snr_db']
            assert len(row['scales'])==10
            for metrics in [row['scales'],row['probability_only']]:
                assert len(metrics) in [0,10]
                for scale,m in enumerate(metrics):
                    n=[1,4,9,16,25,36,64,100,169,256][scale]
                    assert len(m['bins'])==15 and 0<=m['acc']<=1 and np.isfinite(m['logp_true']) and m['logp_true']<=1e-9
                    assert np.isfinite(m['entropy']) and m['entropy']>=-1e-9
                    assert sum(b['count'] for b in m['bins'])==n
                    assert abs(sum(b['correct_sum'] for b in m['bins'])/n-m['acc'])<1e-12
                    for b in m['bins']:
                        assert 0<=b['correct_sum']<=b['count'] and 0<=b['confidence_sum']<=b['count']+1e-5
                    if 'path_accuracy' in m:assert 0<=m['path_accuracy']<=1

            vals=dict(source_index=i,source_id=row['source_id'],level=row['level'],snr_db=row['snr_equiv_db'],noise_seed=row['noise_seed'],
                      mode=row['mode'],profile=row['profile'],lambda_=row['lambda_'],beta=0,preprocessing_sha256=row['preprocessing_sha256'])
            for k,m in enumerate(row['scales'],1):
                for metric in ['acc','logp_true','entropy','path_accuracy']:vals[f'{metric}_k{k}']=m.get(metric,'')
            for metric in ['latent_squared_error','fused_latent_squared_error']:vals[metric]=row.get(metric,'')
            for output in ['token_image','fused_image']:
                for metric in ['psnr_db','lpips_alex','dino_cosine']:vals[output+'_'+metric]=row.get(output,{}).get(metric,'')
            frames.append(vals)
            stat={k:float(v) for k,v in vals.items() if k not in ['source_index','snr_db','noise_seed','lambda_','beta'] and isinstance(v,(int,float))}
            band=[k-1 for k in level['ambiguity_scales_1based']]
            stat['TF_band_accuracy']=float(np.mean([row['scales'][k]['acc'] for k in band])) if band and row['mode']=='TF' else None
            stat['TF_band_logp']=float(np.mean([row['scales'][k]['logp_true'] for k in band])) if band and row['mode']=='TF' else None
            if row['mode']=='CL':
                stat['path_accuracy']=float(np.mean([m['path_accuracy'] for m in row['scales']]))
                stat['path_accuracy_token_weighted']=float(np.average([m['path_accuracy'] for m in row['scales']],weights=[1,4,9,16,25,36,64,100,169,256]))
            lookup[(i,*key)]=stat
            for variant,metrics in [('decision_probability',row['scales']),('beta_positive_probability_only',row['probability_only'])]:
                for k,m in enumerate(metrics,1):
                    bk=(row['level'],row['mode'],row['profile'],variant,k)
                    arr=bins.setdefault(bk,np.zeros((15,3)))
                    for j,bin in enumerate(m['bins']):arr[j]+=np.array([bin['count'],bin['confidence_sum'],bin['correct_sum']])
        expected={(x['name'],n,mode,pr) for x in cfg['levels'] for n in [2001,2002,2003] for mode in (['TF','CL'] if x['role']=='decision' else ['TF']) for pr in profiles}
        assert seen==expected
        archive.append((p,sha(p)))
    assert len(frames)==22500==done['rows']
    arrays={}
    for level in cfg['levels']:
        name=level['name']
        for mode in (['TF','CL'] if level['role']=='decision' else ['TF']):
            for profile in profiles:
                sample=lookup[(0,name,2001,mode,profile)]
                for metric,val in sample.items():
                    if val is None:continue
                    v=np.array([np.mean([lookup[(i,name,n,mode,profile)][metric] for n in [2001,2002,2003]]) for i in range(100)])
                    assert np.isfinite(v).all()
                    arrays[(name,mode,profile,metric)]=v
                    for i,x in enumerate(v):source_rows.append(dict(source_index=i,source_id=pop[i]['source_id'],level=name,snr_db=level['snr_db'],mode=mode,profile=profile,metric=metric,value=float(x)))
    rng=np.random.default_rng(20260929);weights=rng.multinomial(100,[.01]*100,size=10000)/100
    paired=[];decisions=[];summary=[]
    for key,v in arrays.items():
        name,mode,profile,metric=key
        summary.append(dict(level=name,mode=mode,profile=profile,metric=metric,mean=float(v.mean())))
    for level in cfg['levels']:
        if level['role']!='decision':continue
        name=level['name'];record=dict(level=name,snr_db=level['snr_db'],ambiguity_scales=level['ambiguity_scales_1based'],comparisons={})
        for control in ['A1_decision','A2_decision']:
            comparisons={}
            for mode,metrics in [('TF',['TF_band_accuracy','TF_band_logp']),('CL',['path_accuracy','path_accuracy_token_weighted','latent_squared_error','fused_latent_squared_error','fused_image_psnr_db','fused_image_lpips_alex','fused_image_dino_cosine'])]:
                for metric in metrics:
                    k=(name,mode,'V_decision',metric)
                    if k not in arrays:continue
                    delta=arrays[k]-arrays[(name,mode,control,metric)];lo,hi=np.quantile(weights@delta,[.025,.975])
                    q=dict(mean=float(delta.mean()),lo=float(lo),hi=float(hi));comparisons[metric]=q
                    paired.append(dict(level=name,snr_db=level['snr_db'],mode=mode,contrast='V_decision-'+control,metric=metric,**q))
            record['comparisons'][control]=comparisons
        comp=list(record['comparisons'].values())
        record['M1_accuracy_pass']=bool(level['ambiguity_scales_1based']) and all(x['TF_band_accuracy']['mean']>=.02 and x['TF_band_accuracy']['lo']>0 for x in comp)
        record['M1_original_joint_pass']=record['M1_accuracy_pass'] and all(x['TF_band_logp']['lo']>0 for x in comp)
        record['M2_pass']=all(x['path_accuracy']['lo']>0 and x['latent_squared_error']['hi']<0 for x in comp)
        record['M3_pass']=all(((x['fused_image_lpips_alex']['mean']<=-.01 and x['fused_image_lpips_alex']['hi']<0) or (x['fused_image_dino_cosine']['mean']>=.01 and x['fused_image_dino_cosine']['lo']>0)) and x['fused_image_psnr_db']['mean']>=-.3 for x in comp)
        decisions.append(record)
    result=dict(status='REAL_FROZEN_V3_B_PAIRED_ANALYSIS',levels=decisions,
        M1_accuracy_passed=sum(x['M1_accuracy_pass'] for x in decisions)>=2,
        M1_original_joint_passed=sum(x['M1_original_joint_pass'] for x in decisions)>=2,
        M2_passed=sum(x['M2_pass'] for x in decisions)>=2,M3_passed=sum(x['M3_pass'] for x in decisions)>=2,
        M1_rule_note='Both predeclared readings reported; no choice based on development outcomes.',
        interpretation='M3 is practical image utility; token accuracy supports mechanism only; no automatic next step or training restoration.',
        bootstrap=dict(resamples=10000,seed=20260929,unit='source image after 3-noise averaging',interval='pointwise percentile 95%',training_seed_variation=False))
    DEST.mkdir(parents=True);(DEST/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
    table(DEST/'summary.csv',summary);table(DEST/'paired.csv',paired)
    # Pre-existing A shares exactly the same original sources and Gaussian draws.
    # These low-SNR baseline contrasts are secondary and cannot alter M1-M3.
    acells=[sealed(RAW/'A_cells'/f'{i:03d}.json') for i in range(100)]
    baseline_pairs=[];baseline_means=[]
    for level in cfg['levels']:
        if not level['name'].startswith('target_'):continue
        name=level['name'];snr=level['snr_db']
        for baseline in ['B1','O1']:
            for metric in ['psnr_db','lpips_alex','dino_cosine']:
                ref=np.asarray([np.mean([x[metric] for x in d['rows'] if x['method']==baseline and x['snr_equiv_db']==snr]) for d in acells])
                assert all(acells[i]['source_identity']==pop[i] for i in range(100))
                v=arrays[(name,'CL','V_decision','fused_image_'+metric)];delta=v-ref;lo,hi=np.quantile(weights@delta,[.025,.975])
                baseline_pairs.append(dict(level=name,snr_db=snr,contrast='V_fuse-'+baseline,metric=metric,mean=float(delta.mean()),lo=float(lo),hi=float(hi),role='secondary_not_M1_M2_M3'))
                baseline_means.append(dict(level=name,snr_db=snr,method=baseline,metric=metric,mean=float(ref.mean())))
    table(DEST/'low_SNR_A_baseline_pairs.csv',baseline_pairs);table(DEST/'low_SNR_A_baseline_means.csv',baseline_means)
    (DEST/'A_reference_identity.json').write_text(json.dumps(dict(A_index_sha256=sha(ROOT/'results/rx_posterior_step1_20260929/revision_v3_A/index.json'),A_completion_sha256=sha(RAW/'A_completion.json'),same_source_and_noise=True,role='secondary_only_not_parameter_selection'),indent=2)+'\n')

    # Plain JSON, sparse zero bins and compact spacing; exact original byte reconstruction.
    am=[]
    for p,d in archive:
        target=DEST/('source_'+p.stem+'.json');packed=pack(read(p))
        target.write_text(json.dumps(packed,separators=(',',':'),allow_nan=False)+'\n')
        restored=original_bytes(unpack(read(target)));assert hashlib.sha256(restored).hexdigest()==d
        am.append(dict(file=target.name,original_sha256=d,original_bytes=p.stat().st_size,encoding='plain_JSON_sparse_bins_v1'))
    (DEST/'source_archive_manifest.json').write_text(json.dumps(am,indent=2)+'\n')
    calibration_archive=[]
    for kind,bound in [('grid',sel['grid_cells']),('rho',cfg['rho_cells'])]:
        for original,digest in sorted(bound.items()):
            original=Path(original);assert sha(original)==digest;target=DEST/(kind+'_'+original.stem+'.json')
            target.write_text(json.dumps(pack(read(original)),separators=(',',':'),allow_nan=False)+'\n')
            restored=original_bytes(unpack(read(target)));assert hashlib.sha256(restored).hexdigest()==digest
            calibration_archive.append(dict(file=target.name,kind=kind,original_sha256=digest,original_bytes=original.stat().st_size,encoding='plain_JSON_sparse_bins_v1'))
    (DEST/'calibration_archive_manifest.json').write_text(json.dumps(calibration_archive,indent=2)+'\n')
    for label,data in [('frames',frames),('source_means',source_rows)]:
        for start in range(0,len(data),5000):table(DEST/f'{label}_{start//5000:03d}.csv',data[start:start+5000])
    reliability=[];ece=[]
    for key,arr in bins.items():
        level,mode,profile,variant,k=key;total=arr[:,0].sum();e=0.
        for j,(count,confidence,correct) in enumerate(arr):
            e+=abs(confidence-correct)/total
            reliability.append(dict(level=level,mode=mode,profile=profile,variant=variant,scale=k,bin=j,count=int(count),
                                    confidence_sum=float(confidence),correct_sum=int(correct),
                                    mean_confidence=float(confidence/count) if count else '',accuracy=float(correct/count) if count else ''))
        ece.append(dict(level=level,mode=mode,profile=profile,variant=variant,scale=k,ECE=e))
    table(DEST/'reliability.csv',reliability);table(DEST/'ECE.csv',ece)
    for name in ['B_completion.json','B_evaluation_registration.json','B_calibration_completion.json','B_calibration_registration.json','B_frozen_config.json','B_selection.json','B_population.json','limit_check.json']:
        shutil.copyfile(RAW/name,DEST/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'A1_decision':'tab:blue','A2_decision':'tab:orange','V_decision':'tab:green'}
    for level in cfg['levels']:
        name=level['name'];fig,axes=plt.subplots(1,2 if level['role']=='decision' else 1,figsize=(10 if level['role']=='decision' else 6,4));axes=np.atleast_1d(axes)
        for profile,color in colors.items():
            axes[0].plot(range(1,11),[arrays[(name,'TF',profile,f'acc_k{k}')].mean() for k in range(1,11)],'o-',color=color,label=profile.split('_')[0])
            if len(axes)>1:axes[1].plot(range(1,11),[arrays[(name,'CL',profile,f'path_accuracy_k{k}')].mean() for k in range(1,11)],'o-',color=color,label=profile.split('_')[0])
        for ax,title in zip(axes,['TF true-token accuracy','CL path-conditioned accuracy']):
            ax.set(xlabel='Scale',ylabel='Accuracy',title=title,ylim=(0,1));ax.grid(alpha=.25);ax.legend()
        for k in level['ambiguity_scales_1based']:axes[0].axvspan(k-.35,k+.35,alpha=.08,color='green')
        fig.suptitle(f"{name}: {level['snr_db']:g} dB; calibration-frozen decision parameters");fig.tight_layout()
        fig.savefig(DEST/f'{name}_accuracy.svg');fig.savefig(RAW/f'{name}_accuracy.png',dpi=140);plt.close(fig)
    levels=[x for x in cfg['levels'] if x['role']=='decision'];levels=sorted(levels,key=lambda x:x['snr_db'])
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,title in zip(axes,['fused_image_psnr_db','fused_image_lpips_alex','fused_image_dino_cosine'],['Fused PSNR','Fused LPIPS','Fused DINO']):
        for profile,color in colors.items():ax.plot([x['snr_db'] for x in levels],[arrays[(x['name'],'CL',profile,metric)].mean() for x in levels],'o-',color=color,label=profile.split('_')[0])
        ax.set(xlabel='Equivalent SNR (dB)',title=title);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(DEST/'M3_fused_quality.svg');fig.savefig(RAW/'M3_fused_quality.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,level in zip(axes.flat,cfg['levels']):
        for variant,style in [('decision_probability','o-'),('beta_positive_probability_only','s--')]:
            a=sum((bins[(level['name'],'TF','V_decision',variant,k)] for k in range(1,11)),np.zeros((15,3)));ok=a[:,0]>0
            ax.plot(a[ok,1]/a[ok,0],a[ok,2]/a[ok,0],style,label=variant.replace('_probability',''))
        ax.plot([0,1],[0,1],'k:',alpha=.4);ax.set(title=f"{level['snr_db']:g} dB",xlim=(0,1),ylim=(0,1),xlabel='Confidence',ylabel='Accuracy');ax.legend(fontsize=6)
    fig.suptitle('V TF reliability: beta>0 is diagnostic only, same prefixes');fig.tight_layout()
    fig.savefig(DEST/'V_reliability.svg');fig.savefig(RAW/'V_reliability.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for profile,color in colors.items():
        axes[0].plot([x['snr_db'] for x in levels],[arrays[(x['name'],'CL',profile,'path_accuracy')].mean() for x in levels],'o-',color=color,label=profile.split('_')[0])
        axes[1].plot([x['snr_db'] for x in levels],[arrays[(x['name'],'CL',profile,'latent_squared_error')].mean() for x in levels],'o-',color=color,label=profile.split('_')[0])
    for ax,title in zip(axes,['CL path accuracy (equal scale mean)','Raw latent squared error']):
        ax.set(xlabel='Equivalent SNR (dB)',title=title);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(DEST/'M2_path_and_latent.svg');fig.savefig(RAW/'M2_path_and_latent.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,level in zip(axes.flat,levels):
        for variant,style in [('decision_probability','o-'),('beta_positive_probability_only','s--')]:
            a=sum((bins[(level['name'],'CL','V_decision',variant,k)] for k in range(1,11)),np.zeros((15,3)));ok=a[:,0]>0
            ax.plot(a[ok,1]/a[ok,0],a[ok,2]/a[ok,0],style,label=variant.replace('_probability',''))
        ax.plot([0,1],[0,1],'k:',alpha=.4);ax.set(title=f"{level['snr_db']:g} dB",xlim=(0,1),ylim=(0,1),xlabel='Confidence',ylabel='Original-token accuracy');ax.legend(fontsize=6)
    axes.flat[-1].set_axis_off()
    fig.suptitle('V CL probability reliability against original encoding tokens (not M2 path target)')
    fig.tight_layout();fig.savefig(DEST/'V_CL_probability_reliability.svg');fig.savefig(RAW/'V_CL_probability_reliability.png',dpi=140);plt.close(fig)
    for family,metrics,thresholds in [
        ('M1',[('TF_band_accuracy','TF accuracy gain'),('TF_band_logp','TF true-token logp gain')],[.02,0]),
        ('M2',[('path_accuracy','CL path accuracy gain'),('latent_squared_error','CL latent squared-error change')],[0,0]),
        ('M3',[('fused_image_psnr_db','Fused PSNR change'),('fused_image_lpips_alex','Fused LPIPS change'),('fused_image_dino_cosine','Fused DINO change')],[-.3,-.01,.01])]:
        fig,axes=plt.subplots(1,len(metrics),figsize=(4.5*len(metrics),4));axes=np.atleast_1d(axes)
        for ax,(metric,title),threshold in zip(axes,metrics,thresholds):
            for control,color,offset in [('A1_decision','tab:blue',-.13),('A2_decision','tab:orange',.13)]:
                data=sorted([x for x in paired if x['metric']==metric and x['contrast']=='V_decision-'+control],key=lambda x:x['snr_db'])
                xs=np.array([x['snr_db'] for x in data])+offset
                ax.vlines(xs,[x['lo'] for x in data],[x['hi'] for x in data],color=color)
                ax.plot(xs,[x['mean'] for x in data],'o',color=color,label='V - '+control.split('_')[0],markersize=4)
            ax.axhline(0,color='black',linestyle=':',linewidth=.8)
            if threshold!=0:ax.axhline(threshold,color='green',linestyle='--',linewidth=.8,label='mean threshold')
            ax.set(xlabel='Equivalent SNR (dB)',title=title);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f'{family}: paired source-bootstrap 95% intervals (10000 resamples)');fig.tight_layout()
        fig.savefig(DEST/f'{family}_paired_intervals.svg');fig.savefig(RAW/f'{family}_paired_intervals.png',dpi=140);plt.close(fig)
    files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(DEST.iterdir()) if p.is_file()}
    assert all(x['bytes']<10_000_000 for x in files.values())
    (DEST/'index.json').write_text(json.dumps(dict(status='REAL_V3_B_PUBLICATION',source_cells=100,rows=len(frames),files=files,frozen_config_sha256=sha(RAW/'B_frozen_config.json'),publisher_sha256=sha(__file__),archive_codec_sha256=sha(Path(__file__).with_name('rx_v3_archive.py'))),indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='levels'}))
if __name__=='__main__':main()
