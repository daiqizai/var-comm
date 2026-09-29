"""Publish the immutable calibration-only revision outcome; never run a GPU model."""
from pathlib import Path
import json,csv,hashlib,shutil,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'outputs/RX-POSTERIOR-STEP1-20260929/revision_v2'
DEST=ROOT/'results/rx_posterior_step1_20260929/revision_v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvwrite(name,rows):
    assert rows
    with (DEST/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def checked(p):
    assert sha(p)==Path(str(p)+'.sha256').read_text().strip()
    return read(p)
def main():
    assert not DEST.exists()
    reg=read(SRC/'registration.json')
    for p,h in reg['bindings'].items():assert sha(p)==h
    cfg=checked(SRC/'frozen_config.json');noise=checked(SRC/'noise_levels.json')
    done=checked(SRC/'calibration_completion.json')
    assert done['status']=='REVISED_HARD_GATE_FAILED_NO_DEVELOPMENT' and done['passed'] is False
    assert not (SRC/'failure.json').exists() and not (SRC/'supervisor_failure.json').exists()
    score_rows=[];score_means=np.zeros((4,3,20),dtype=np.float64)
    source_ids=[]
    for i in range(200):
        p=SRC/'proper_score_cells'/f'{i:04d}.json';c=checked(p)
        assert cfg['proper_score_cell_sha256'][str(p)]==sha(p) and c['source_index']==i
        scores=np.asarray(c['scores']);assert scores.shape==(4,3,20) and np.isfinite(scores).all()
        score_means+=scores/200;source_ids.append(c['source_id'])
        for level in range(4):
            for j,prior in enumerate(['A1','A2','V']):
                for g,(beta,lam) in enumerate(cfg['grid']):
                    score_rows.append(dict(source_id=c['source_id'],source_index=i,noise_seeds='4101|4102|4103',
                        level=level,eta=cfg['levels'][level]['eta'],snr_equiv_db=cfg['levels'][level]['snr_db'],
                        prior=prior,beta=beta,lambda_=lam,TF_true_logp_scales7_10=float(scores[level,j,g]),
                        aggregation='equal scales, then three calibration seeds'))
    assert len(set(source_ids))==200
    np.testing.assert_array_equal(score_means,np.asarray(cfg['complete_grid_scores']))
    for level in range(4):
        for j,prior in enumerate(['A1','A2','V']):
            best=int(np.argmax(score_means[level,j]))
            assert best==cfg['levels'][level]['parameters'][prior]['grid_index']
    difficulty_rows=[]
    for phase in ['coarse','refined']:
        aggregate=np.zeros(len(noise[phase]['snrs_db']),dtype=np.float64)
        for path,h in noise[phase]['cells'].items():
            assert sha(path)==h;c=checked(path)
            assert c['source_id']==source_ids[c['source_index']]
            aggregate+=np.asarray(c['accuracy'])/200
            for snr,acc in zip(c['snrs_db'],c['accuracy']):
                difficulty_rows.append(dict(phase=phase,source_id=c['source_id'],source_index=c['source_index'],
                    noise_seeds='4101|4102|4103',snr_equiv_db=snr,eta=10**(-snr/20),A1_TF_accuracy_scales8_10=acc))
        np.testing.assert_array_equal(aggregate,np.asarray(noise[phase]['accuracy']))
    gate_rows=[];scale_rows=[];cells=[]
    for path,h in done['cells'].items():
        assert sha(path)==h;c=checked(path);cells.append(c)
        assert c['zero_noise_nearest_tokens_exact'] and c['zero_noise_Fq_bitwise_equal']
        assert c['official_TF_logits_max_abs']<=cfg['design']['official_forward_absolute_tolerance']
        for row in c['checks']:
            flat={k:row[k] for k in ['version','mode','seed','beta','lambda_','token_correct','token_total']}
            flat.update(source_id=c['source_id'],source_index=c['source_index'],snr_db=30,token_accuracy=row['token_correct']/row['token_total'])
            for k,metric in enumerate(row['scale_metrics'],1):
                for key in ['acc','logp_true','entropy']:flat[f'{key}_k{k}']=metric[key]
                scale_rows.append(dict(source_id=c['source_id'],source_index=c['source_index'],version=row['version'],mode=row['mode'],
                    seed=row['seed'],scale=k,token_count=[1,4,9,16,25,36,64,100,169,256][k-1],
                    acc=metric['acc'],logp_true=metric['logp_true'],entropy=metric['entropy']))
            gate_rows.append(flat)
    assert len(gate_rows)==120 and len(scale_rows)==1200
    for row in done['summary']:
        rows=[r for r in gate_rows if r['version']==row['version'] and r['mode']==row['mode']]
        acc=sum(r['token_correct'] for r in rows)/sum(r['token_total'] for r in rows)
        assert acc==row['token_accuracy'] and row['passed']==(acc>=.99)
    DEST.mkdir(parents=True)
    for name in ['registration.json','frozen_config.json','noise_levels.json','calibration_completion.json','supervisor_completion.json']:
        shutil.copyfile(SRC/name,DEST/name)
    for c,path in zip(cells,done['cells']):
        target=DEST/'engineering_cells'/Path(path).name;target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
    shutil.copyfile(SRC/'queue_restoration/completion.json',DEST/'original_queue_restoration.json')
    shutil.copyfile(SRC/'queue_restoration/failure.json',DEST/'first_restoration_registration_refusal.json')
    shutil.copyfile(SRC/'queue_restoration/n3060_manifest_adapter/registration.json',DEST/'n3060_manifest_adapter_registration.json')
    shutil.copyfile(SRC/'queue_restoration/n3060_manifest_adapter/runtime_receipt.json',DEST/'n3060_manifest_adapter_runtime.json')
    csvwrite('calibration_proper_scores_by_source.csv',score_rows)
    csvwrite('calibration_difficulty_by_source.csv',difficulty_rows)
    csvwrite('engineering_per_frame.csv',gate_rows);csvwrite('engineering_per_scale.csv',scale_rows)
    selected=[]
    for level,x in enumerate(cfg['levels']):
        for prior,param in x['parameters'].items():
            selected.append(dict(level=level,target_accuracy=x['target_accuracy'],achieved_accuracy=x['achieved_accuracy'],
                snr_equiv_db=x['snr_db'],eta=x['eta'],prior=prior,**param))
    csvwrite('selected_calibration_parameters.csv',selected)
    csvwrite('hard_gate_summary.csv',done['summary'])
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'rx-v2','axes.grid':True,'grid.alpha':.2})
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    ax=axs[0];ax.plot(noise['coarse']['snrs_db'],noise['coarse']['accuracy'],color='#245b9b')
    for x in cfg['levels']:
        ax.scatter([x['snr_db']],[x['achieved_accuracy']],color='#b24335')
        ax.annotate(f"{x['snr_db']:.1f} dB / {100*x['achieved_accuracy']:.2f}%",(x['snr_db'],x['achieved_accuracy']),xytext=(4,8),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Equivalent SNR (dB)',ylabel='A1 TF token accuracy (scales 8-10)',xlim=(0,45),ylim=(0,1.05),title='200 calibration sources x 3 seeds')
    ax=axs[1];names=['fixed']+[f'calibrated_level_{i}' for i in range(4)]
    for j,mode in enumerate(['TF','CL']):
        ys=[next(r['token_accuracy'] for r in done['summary'] if r['version']==n and r['mode']==mode) for n in names]
        ax.bar(np.arange(5)+(j-.5)*.32,ys,width=.32,label=mode,color=['#245b9b','#da8e28'][j])
    ax.axhline(.99,color='#ad2435',ls='--',label='required 99%')
    ax.set_xticks(range(5),['fixed','cal 90%','cal 70%','cal 50%','cal 30%'])
    ax.set(ylim=(0,1.08),ylabel='V token agreement with clean official tokens',title='Hard check: 30 dB, 4 calibration sources x 3 seeds')
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(DEST/'calibration_and_hard_gate.svg',metadata={'Date':None});fig.savefig(SRC/'calibration_and_hard_gate.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,mode in zip(axs,['TF','CL']):
        for name in names:
            ys=[np.mean([r['acc'] for r in scale_rows if r['version']==name and r['mode']==mode and r['scale']==k]) for k in range(1,11)]
            ax.plot(range(1,11),ys,marker='.',label=name.replace('calibrated_level_','cal level '))
        ax.axhline(.99,color='#ad2435',ls='--')
        ax.set(xlabel='Scale',ylabel='Token agreement',ylim=(0,1.04),title=f'30 dB engineering only: {mode}',xticks=range(1,11))
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(DEST/'engineering_scale_accuracy.svg',metadata={'Date':None});fig.savefig(SRC/'engineering_scale_accuracy.png',dpi=150);plt.close(fig)
    audit=dict(status='REAL_CALIBRATION_AND_FAILED_30DB_GATE_PUBLICATION_VERIFIED',calibration_sources=200,calibration_noise_seeds=3,
        grid_parameter_pairs=20,source_averaged_proper_score_rows=len(score_rows),source_averaged_difficulty_rows=len(difficulty_rows),
        real_engineering_frames=len(gate_rows),engineering_scale_rows=len(scale_rows),development_accessed=False,holdout_accessed=False,
        original_bindings_verified=reg['original_bindings_verified'],C='NOT_RUN',M1='NOT_RUN',M2='NOT_RUN',M3='NOT_RUN',
        limitation='30 dB was fixed before results; failure here is not a claim that every higher SNR fails. Calibrated positive beta retains a clean residual variance floor.')
    (DEST/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    files={str(p.relative_to(DEST)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(DEST.rglob('*')) if p.is_file()}
    assert all(x['bytes']<10_000_000 for x in files.values())
    (DEST/'index.json').write_text(json.dumps(dict(status=audit['status'],files=files,source_registration_sha256=sha(SRC/'registration.json')),indent=2)+'\n')
    print(json.dumps(audit));print('INDEX',len(files),'bytes',sum(v['bytes'] for v in files.values()))
if __name__=='__main__':main()
