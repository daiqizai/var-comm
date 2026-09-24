"""Publish completed QPSK development and matched continuous comparisons on CPU."""
import csv, itertools, json, os, shutil
from functools import lru_cache
from pathlib import Path
import numpy as np
from tools.publish_grid_milestones import ROOT, OUT, EXP, check, read, write, csvwrite, split_csv, digest
from token_efficiency.statistics import FrameTable, METRICS, minimum_tested_uses, paired

sha = lru_cache(maxsize=None)(digest)
DEST = ROOT/'results/token_channel_efficiency_20260923/QPSK_development_v1'

def load_completed(relative, stage, methods):
    folder=OUT/relative
    done=read(folder/'completion.json');reg=read(folder/'registration.json')
    check(done['synthetic'] is False and done['frame_rows']==len(methods)*1500, 'real complete development')
    check(sha(folder/'registration.json')==done['registration_sha256'], 'registration identity')
    receipt=read(OUT/'delivery_chain_v1/stages'/f'{stage}.json')
    check(receipt['returncode']==0 and sha(receipt['snapshot'])==receipt['completion_sha256']
          ==sha(folder/'completion.json'), 'stage completion identity')
    for name,h in reg['context']['bindings'].items():check(sha(name)==h,'binding '+name)
    for name,h in done['files'].items():check(sha(folder/name)==h,'original result '+name)
    check(len(reg['sources'])==100 and reg['snrs']==[1,4,7,13,19] and reg['seeds']==[2001,2002,2003], 'development population')
    check(len(done['cell_sha256'])==100*len(methods),'cell count')
    rows=[];timing=[];warm=[]
    for name,h in done['cell_sha256'].items():
        p=folder/'cells'/name;c=read(p)
        check(sha(p)==h==read(p.with_suffix('.seal.json'))['sha256'],'cell seal')
        check(c['registration_sha256']==done['registration_sha256'],'cell registration')
        check(c['method'] in methods and all(r['method']==c['method'] and r['source_id']==c['source_id'] for r in c['rows']), 'cell method/source')
        for path,expected in c['previews'].items():check(sha(path)==expected,'received preview')
        rows.extend(c['rows']);timing.extend(c['timing']);warm.extend(c['warmup'])
    table=FrameTable(rows,'development',reg['sources'],reg['snrs'],reg['seeds'],methods)
    check(table.sha256==done['table_sha256'],'canonical frame table')
    old={(r['method'],int(r['snr_db'])):r for r in csv.DictReader((folder/'summary.csv').open())}
    summary=table.summary();check(len(old)==len(summary),'summary coverage')
    for r in summary:
        for k in [*METRICS,'E_mean','E_min','E_p05','E_p95','E_max']:
            check(abs(float(old[(r['method'],r['snr_db'])][k])-r[k])<1e-9,'summary '+k)
    ids=[list(reg['sources'])[i] for i in range(0,100,11)]
    keys={(r['method'],r['source_id'],r['snr_db'],r['noise_seed'],r['repeat']) for r in timing}
    expected=set(itertools.product(methods,ids,reg['snrs'],[2001],[0,1]))
    check(keys==expected and len(timing)==len(expected)==done['timed_calls'],'complete timing grid')
    expected_warm=set(itertools.product(methods,ids,range(3)))
    check(len(warm)==len(expected_warm) and {(r['method'],r['source_id'],r['repeat']) for r in warm}==expected_warm,'complete warmup grid')
    for r in timing:
        ctx=table.context[r['method']]
        check(r['preprocessing_id']==reg['sources'][r['source_id']] and
              all(r[k]==ctx[k] for k in ['N','context_sha256','run_id']), 'timing identity')
        check(0<=r['quality_max_abs_error']<=2e-5 and r['tx_ms']>0 and r['rx_ms']>0 and
              abs(r['total_ms']-r['tx_ms']-r['rx_ms'])<1e-7, 'actual timing/replay')
        check(r['timing_endpoints']=='CPU uint8 RGB -> CPU waveform; CPU observation -> CPU float RGB; noise outside RX','timing endpoints')
    return table,timing,warm,reg,done

def resource_bootstrap(table,policy,targets,repeats=10000):
    """Unreached bootstrap samples remain counted; savings intervals are conditional."""
    result=[]
    for snr in table.snrs:
        methods=['P2048','P3060','P4084']+sorted({c['method'] for c in policy['choices'] if c['snr_db']==snr})
        values=np.stack([table.source_values(m,snr)[:,[1,2]] for m in methods])
        rng=np.random.default_rng(20260923);means=[]
        for start in range(0,repeats,250):
            ids=rng.integers(0,len(table.ids),(min(250,repeats-start),len(table.ids)))
            means.append(values[:,ids,:].mean(axis=2))
        means=np.concatenate(means,axis=1);budgets=np.array([table.context[m]['N'] for m in methods])
        for level,target in targets['targets'].items():
            met=(means[:,:,0]>=target['psnr_min'])&(means[:,:,1]<=target['lpips_max'])
            cn=np.where(met[:3],budgets[:3,None],np.inf).min(axis=0)
            for family in ['raw','arithmetic']:
                indices=[i for i,m in enumerate(methods) if table.context[m]['family']==family]
                dn=np.where(met[indices],budgets[indices,None],np.inf).min(axis=0)
                valid=np.isfinite(cn)&np.isfinite(dn);savings=1-dn[valid]/cn[valid]
                result.append({'snr_db':snr,'target':level,'family':family,'repeats':repeats,
                    'unit':'source image after averaging noise; shared source resample across methods',
                    'policy_refit':False,'training_seed_variation_included':False,
                    'both_attained_repeats':int(valid.sum()),'unreached_repeats':int((~valid).sum()),
                    'conditional_saving_ci95':np.quantile(savings,[.025,.975]).tolist() if len(savings) else None,
                    'interval_condition':'both families attain the mean target in the tested grid',
                    'continuous_N_counts':{str(int(n)):int((cn==n).sum()) for n in [2048,3060,4084]},
                    'digital_N_counts':{str(int(n)):int((dn==n).sum()) for n in [2048,3060,4084]},
                    'continuous_unreached':int((~np.isfinite(cn)).sum()),'digital_unreached':int((~np.isfinite(dn)).sum()),
                    'table_sha256':table.sha256})
    return result

def main():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU mask required')
    check(not DEST.exists(),'immutable publication already exists')
    cal=OUT/'digital_grid_v1/QPSK/calibration'
    policy=read(cal/'policy.json');caldone=read(cal/'completion.json')
    check(sha(cal/'policy.json')==caldone['policy_sha256'] and policy['calibration_table_sha256']==caldone['table_sha256'],'frozen calibration identity')
    published=ROOT/'results/token_channel_efficiency_20260923/grid_milestones_v1/QPSK_calibration'
    check(sha(published/'policy.json')==sha(cal/'policy.json') and sha(published/'completion.json')==sha(cal/'completion.json'),'previously audited calibration')
    methods=sorted({r['method'] for r in csv.DictReader((published/'summary.csv').open())})
    check(len(methods)==26,'QPSK eligible roster')
    digital,dt,dw,dr,dd=load_completed('digital_grid_v1/QPSK/development','QPSK_development',methods)
    continuous,ct,cw,cr,cd=load_completed('continuous_grid_v1','continuous_development',['P2048','P3060','P4084'])
    check(dr['policy_sha256']==dd['policy_sha256']==sha(cal/'policy.json'),'development frozen policy')
    check(dr['sources']==cr['sources'] and dr['context']['decoder_sha256']==cr['context']['decoder_sha256'],'common population/preprocessing/decoder')
    qual=read(OUT/'continuous_grid_v1/qualification.json')
    check(qual['status']=='REAL_SELECTED_CONTINUOUS_REPLAY_PASS' and len(qual['checks'])==12 and all(r['actual_selected_replay_pass'] for r in qual['checks']),'actual selected qualification')
    for s in cr['context']['selected'].values():
        check(sha(s['checkpoint'])==s['checkpoint_sha256'] and sha(s['selected'])==s['selected_sha256'],'continuous selected identity')
    rows=list(digital.frames.values())+list(continuous.frames.values())
    table=FrameTable(rows,'development',dr['sources'],dr['snrs'],dr['seeds'],[*methods,*continuous.methods])
    targets=read(EXP/'quality_targets.json')
    minimum=minimum_tested_uses(table,policy,targets,list(continuous.methods))
    comparisons=[paired(table,c['method'],f"P{c['N']}",c['snr_db']) for c in policy['choices']]
    intervals=resource_bootstrap(table,policy,targets)
    DEST.mkdir(parents=True)
    for name in ['registration.json','completion.json','summary.csv','timing.csv','warmup.csv','candidates.json']:
        source=OUT/'digital_grid_v1/QPSK/development'/name
        if source.exists():shutil.copyfile(source,DEST/name)
    split_csv(OUT/'digital_grid_v1/QPSK/development/per_frame.csv',DEST/'per_frame_parts')
    for name,source in [('policy.json',cal/'policy.json'),('quality_targets.json',EXP/'quality_targets.json'),
                        ('stage_receipt.json',OUT/'delivery_chain_v1/stages/QPSK_development.json')]:shutil.copyfile(source,DEST/name)
    csvwrite(DEST/'combined_summary.csv',table.summary())
    write(DEST/'paired.json',comparisons);write(DEST/'minimum_uses.json',minimum)
    csvwrite(DEST/'minimum_uses.csv',[{k:v for k,v in r.items() if k not in ['digital_evidence','continuous_evidence']} for r in minimum])
    write(DEST/'minimum_uses_bootstrap.json',intervals)
    (DEST/'source_means').mkdir()
    for i,m in enumerate(table.methods):
        sr=[]
        for snr in table.snrs:
            for sid,v in zip(table.ids,table.source_values(m,snr)):
                sr.append({'method':m,'population':'development','source_id':sid,'preprocessing_id':table.sources[sid],
                    'snr_db':snr,'noise_seeds':json.dumps(table.seeds),**table.context[m],
                    **dict(zip(METRICS,map(float,v))),'table_sha256':table.sha256})
        csvwrite(DEST/'source_means'/f'method_{i:02d}.csv',sr)
    timings=dt+ct;ts=[]
    for m in table.methods:
        for snr in table.snrs:
            rs=[r for r in timings if r['method']==m and r['snr_db']==snr]
            ts.append({'method':m,'snr_db':snr,'calls':len(rs),
                **{k:float(np.mean([r[k] for r in rs])) for k in ['tx_ms','rx_ms','total_ms']},'table_sha256':table.sha256})
    csvwrite(DEST/'combined_timing_summary.csv',ts)
    failures=[]
    for m in table.methods:
        for snr in table.snrs:
            rs=[table.frames[m,s,snr,k] for s in table.ids for k in table.seeds]
            failures.append({'method':m,'snr_db':snr,'frames':len(rs),
                'header_failures':sum(r['header_ok'] is False for r in rs),
                'body_crc_failures':sum(r['body_crc_ok'] is False for r in rs),
                'source_overflow_erasures':sum(r['source_overflow_erasure'] is True for r in rs),
                'table_sha256':table.sha256})
    csvwrite(DEST/'failures.csv',failures)
    plots(table,policy,ts,minimum)
    lineage={str(OUT/r/'completion.json'):sha(OUT/r/'completion.json') for r in ['digital_grid_v1/QPSK/calibration','digital_grid_v1/QPSK/development','continuous_grid_v1']}
    write(DEST/'audit.json',{'status':'REAL_QPSK_DEVELOPMENT_AND_MATCHED_CONTINUOUS_VERIFIED','synthetic':False,
        'combined_table_sha256':table.sha256,'digital_frame_rows':len(digital.frames),'continuous_frame_rows':len(continuous.frames),
        'digital_timings':len(dt),'digital_warmups':len(dw),'continuous_timings':len(ct),'continuous_warmups':len(cw),
        'lineage':lineage,'frozen_policy_sha256':sha(cal/'policy.json'),'full_study_complete':False})
    write(DEST/'index.json',{'publisher_sha256':sha(__file__),'table_sha256':table.sha256,
        'files':{str(p.relative_to(DEST)):digest(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
        'continuous_raw_records':'../grid_milestones_v1/continuous_development/per_frame_parts/manifest.json',
        'pending':['16QAM calibration/development','combined B1/B2','C','historical GPU workers','final combined delivery']})
    print(json.dumps({'published':str(DEST),'table_sha256':table.sha256,'minimum_uses':[{k:r[k] for k in ['snr_db','target','family','continuous_N_required','digital_N_required','saving','status']} for r in minimum]}),flush=True)

def plots(table,policy,timing,minimum):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='qpsk-development-v1'
    lookup={(r['method'],r['snr_db']):r for r in table.summary()}
    times={(r['method'],r['snr_db']):r for r in timing}
    for axis in ['N','time']:
        fig,axs=plt.subplots(4,5,figsize=(20,12))
        for si,snr in enumerate(table.snrs):
            for mi,metric in enumerate(METRICS):
                ax=axs[mi,si]
                for family in ['continuous','raw','arithmetic']:
                    ms=[f'P{n}' for n in [2048,3060,4084]] if family=='continuous' else [next(c['method'] for c in policy['choices'] if c['family']==family and c['N']==n and c['snr_db']==snr) for n in [2048,3060,4084]]
                    xs=[table.context[m]['N'] if axis=='N' else times[m,snr]['total_ms'] for m in ms]
                    ax.plot(xs,[lookup[m,snr][metric] for m in ms],'.-',label=family)
                ax.set(xlabel='Complex channel uses' if axis=='N' else 'Online TX+RX (ms)',ylabel=metric,title=f'{snr} dB')
                ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=8);fig.suptitle('Frozen calibration policies: QPSK vs selected continuous; original 100-source development')
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(DEST/f'quality_vs_{axis}.svg');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,level in zip(axs,targets_order(minimum)):
        for family in ['raw','arithmetic']:
            rs=[r for r in minimum if r['target']==level and r['family']==family]
            ax.plot([r['snr_db'] for r in rs],[100*r['saving'] if r['saving'] is not None else np.nan for r in rs],'.-',label=family)
            for r in rs:
                if r['saving'] is None:ax.text(r['snr_db'],0,'NR',ha='center',fontsize=7)
        ax.axhline(0,color='grey',lw=.6);ax.set(title=level,xlabel='SNR (dB)',ylabel='Digital N saving vs continuous (%)',xlim=(0,20),ylim=(-110,10),xticks=table.snrs);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Minimum tested N; NR = at least one family does not attain target; no interpolation')
    fig.tight_layout();fig.savefig(DEST/'minimum_N_savings.svg');plt.close(fig)
    for p in DEST.glob('*.svg'):p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')

def targets_order(rows):return list(dict.fromkeys(r['target'] for r in rows))

if __name__=='__main__':main()
