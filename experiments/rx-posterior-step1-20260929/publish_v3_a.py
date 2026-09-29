"""Publish actual completed task A and exact-limit check without claiming B completion."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,shutil,csv,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RAW=ROOT/'outputs/RX-POSTERIOR-STEP1-20260929/revision_v3'
DEST=ROOT/'results/rx_posterior_step1_20260929/revision_v3_A'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sealed(p):
    assert Path(str(p)+'.sha256').read_text().strip()==sha(p)
    return read(p)
def main():
    assert not DEST.exists(),'immutable publication exists'
    done=sealed(RAW/'A_completion.json');assert done['status']=='REAL_ORACLE_AND_LMMSE_TASK_A_COMPLETE'
    assert done['source_A_approximate_check_passed'] and sealed(RAW/'limit_check.json')['passed']
    assert read(RAW/'A_supervisor_completion.json')['status']=='RX_STAGE_COMPLETE_ORIGINAL_TRAINING_REMAINS_STOPPED'
    reg=read(RAW/'A_registration.json')
    for p,d in reg['bindings'].items():assert sha(p)==d,p
    pop=sealed(RAW/'A_population.json')['records'];rows=[];cells=[]
    for i in range(100):
        p=RAW/'A_cells'/f'{i:03d}.json';d=sealed(p)
        assert sha(p)==done['cells'][str(p)] and d['registration_sha256']==sha(RAW/'A_registration.json')
        assert d['source_identity']==pop[i] and d['source_index']==i
        seen=set()
        for row in d['rows']:
            k=(row['method'],row['snr_equiv_db'],row['noise_seed'])
            assert k not in seen;seen.add(k)
            assert row['source_id']==pop[i]['source_id'] and row['preprocessing_sha256']==pop[i]['rgb_sha256']
            assert abs(row['eta']-10**(-row['snr_equiv_db']/20))<1e-14
            assert all(np.isfinite(row[k]) for k in ['psnr_db','lpips_alex','dino_cosine'])
        assert seen=={(m,s,n) for m in ['B1','O1'] for s in [-5,-2,0,1,4,7,13] for n in [2001,2002,2003]}
        rows.extend(d['rows']);cells.append(d)
    assert len(rows)==4200
    excerpt=read(RAW.parent/'existing_results_excerpt.json')
    for p,d in excerpt['source_files'].items():assert sha(p)==d
    p4084=[x for x in excerpt['summary'] if x['method']=='P4084_N4084_seed2026092304']
    assert len(p4084)==2
    weights=np.random.default_rng(20260929).multinomial(100,[.01]*100,size=10000)/100
    means=[];paired=[];sources=[];C_levels=[]
    for snr in [-5,-2,0,1,4,7,13]:
        z={m:{k:np.array([np.mean([x[k] for x in d['rows'] if x['method']==m and x['snr_equiv_db']==snr]) for d in cells]) for k in ['psnr_db','lpips_alex','dino_cosine','latent_squared_error']} for m in ['B1','O1']}
        for m in z:
            means.append(dict(method=m,snr_db=snr,**{k:float(v.mean()) for k,v in z[m].items()}))
            for i in range(100):sources.append(dict(source_id=pop[i]['source_id'],source_index=i,method=m,snr_db=snr,**{k:float(v[i]) for k,v in z[m].items()}))
        contrast={}
        for k in z['B1']:
            d=z['O1'][k]-z['B1'][k];lo,hi=np.quantile(weights@d,[.025,.975])
            contrast[k]=dict(mean=float(d.mean()),lo=float(lo),hi=float(hi))
            paired.append(dict(snr_db=snr,contrast='O1-B1',metric=k,**contrast[k]))
        if snr<=1:
            lp=contrast['lpips_alex'];di=contrast['dino_cosine']
            C_levels.append(dict(snr_db=snr,passed=lp['mean']<=-.02 and lp['hi']<0 and di['mean']>=.02 and di['lo']>0))
    O1=next(x for x in means if x['method']=='O1' and x['snr_db']==1);P=next(x for x in p4084 if float(x['snr_db'])==1)
    pbetter=O1['lpips_alex']<float(P['lpips_alex']) or O1['dino_cosine']>float(P['dino_cosine'])
    decision=dict(C_passed=sum(x['passed'] for x in C_levels)>=2 and pbetter,low_SNR_points=C_levels,
                  approximate_P4084_1dB_better=pbetter,P4084_reference=P,
                  B_status='NOT_COMPLETE_IN_THIS_PUBLICATION',exact_limit_passed=True,source_A_reproduction=done['source_A_reproduction'],
                  caveats=['Oracle uses true Fq unavailable at RX.','Equivalent N4096 latent probe versus N4084 actual PHY is approximate.','Calibration standardization gives ensemble energy 2N, not per-frame exact normalization.','No original training resumed.'])
    DEST.mkdir(parents=True)
    def table(name,records):
        with (DEST/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    table('frames.csv',rows);table('source_means.csv',sources);table('summary.csv',means);table('paired.csv',paired)
    for name in ['A_completion.json','A_registration.json','A_population.json','A_supervisor_completion.json','limit_check.json','quality_identity.json']:
        shutil.copyfile(RAW/name,DEST/name)
    shutil.copyfile(RAW.parent/'existing_results_excerpt.json',DEST/'existing_results_excerpt.json')
    for i in range(100):shutil.copyfile(RAW/'A_cells'/f'{i:03d}.json',DEST/f'source_{i:03d}.json')
    (DEST/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,k,title in zip(axes,['psnr_db','lpips_alex','dino_cosine'],['PSNR (dB), higher better','LPIPS, lower better','DINO cosine, higher better']):
        for method in ['B1','O1']:
            data=[x for x in means if x['method']==method]
            ax.plot([x['snr_db'] for x in data],[x[k] for x in data],marker='o',label=method)
        ax.plot([float(x['snr_db']) for x in p4084],[float(x[k]) for x in p4084],'k--s',label='P4084 (approx.)')
        ax.set(xlabel='Equivalent SNR (dB)',title=title);ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Task A: real 100 development sources, 3 noise seeds; Dc decoder')
    fig.tight_layout();fig.savefig(DEST/'oracle_vs_lmmse.svg');fig.savefig(RAW/'oracle_vs_lmmse.png',dpi=160);plt.close(fig)
    index=dict(status='REAL_TASK_A_AND_EXACT_LIMIT_PUBLICATION',rows=len(rows),sources=100,bootstrap=dict(resamples=10000,seed=20260929,unit='source image after averaging 3 seeds',CI='pointwise paired percentile 95%'),
               raw_registration_sha256=sha(RAW/'A_registration.json'),files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(DEST.iterdir()) if p.is_file()})
    assert all(v['bytes']<10_000_000 for v in index['files'].values())
    (DEST/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(result=str(DEST),decision=decision,files=len(index['files']))))
if __name__=='__main__':main()
