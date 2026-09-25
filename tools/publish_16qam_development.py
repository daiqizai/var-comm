"""CPU-only publication of completed 16QAM development and frozen-policy comparisons."""
import csv, json, os, shutil, math
from pathlib import Path
import numpy as np
from tools.publish_grid_milestones import ROOT, OUT, EXP, check, read, write, csvwrite, split_csv, digest
from tools.publish_qpsk_development import load_completed, resource_bootstrap, sha
from token_efficiency.statistics import FrameTable, METRICS, minimum_tested_uses, paired

REL='digital_grid_v1/16QAM/development'
DEST=ROOT/'results/token_channel_efficiency_20260923/QAM16_development_v1'

def verify_row(row, candidate, source_id, source_index, reg):
    check(row['method']==candidate['method'] and row['source_id']==source_id
          and row['source_index']==source_index, 'candidate/source identity')
    check(row['context_sha256']==reg['context_sha256']
          and row['run_id']=='TOKEN-CHANNEL-EFFICIENCY-20260923/'+REL, 'registered execution')
    namespace=candidate['method'].rsplit('/m',1)[0]
    check(row['noise_namespace']==namespace
          and row['noise_id']==f"{namespace}|{source_id}|{row['noise_seed']}", 'noise identity')
    for key in ['N','N_header','N_data','coded_slots','m_requested','mcs','family']:
        check(row[key]==candidate[key], 'candidate resource '+key)
    check(row['mcs']=='16QAM' and row['energy_constraint']=='fixed_constellation_average_2_per_symbol',
          'fixed constellation average-energy identity')
    check(math.isfinite(row['E']) and row['E']>0, 'actual frame energy')
    check(row['N_header']+row['N_data']==row['N'] and row['coded_slots']==4*row['N_data'],
          'paid header and modulation ledger')
    if row['source_overflow_erasure']:
        check(row['payload_bits']==row['mother_bits']==0, 'erasure ledger')
    else:
        check(row['payload_bits']>0 and row['payload_bits']+22<=row['coded_slots']
              and row['mother_bits']==2*(row['payload_bits']+22), 'source/FEC capacity ledger')

def registered_source(reg,index):
    sid=list(reg['sources'])[index];binding=reg['image_bindings'][index]
    check(binding['index']==index and binding['rgb_sha256']==reg['sources'][sid], 'development source binding')
    return sid

def verify_development_cells(reg,done):
    folder=OUT/REL
    candidates=read(folder/'candidates.json')
    eligible=[c for c in candidates if c['status']=='ELIGIBLE']
    check(len(candidates)==30 and len(eligible)==29,'candidate roster')
    check(reg['role']=='development' and reg['mcs']=='16QAM','development axes')
    expected={f'{i:04d}_{j:02d}.json' for i in range(100) for j in range(29)}
    check(set(done['cell_sha256'])==expected,'complete cell names')
    actual={p.name for p in (folder/'cells').glob('*.json') if not p.name.endswith('.seal.json')}
    check(actual==expected,'no extra unsealed cells')
    for name in sorted(expected):
        i,j=map(int,Path(name).stem.split('_'))
        p=folder/'cells'/name;cell=read(p)
        check(read(p.with_suffix('.seal.json'))['registration_sha256']==done['registration_sha256'],'seal registration')
        sid=registered_source(reg,i)
        for row in cell['rows']:verify_row(row,eligible[j],sid,i,reg)

def energy_figure(rows,policy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    lookup={(r['method'],r['snr_db']):r for r in rows}
    for ax,N in zip(axs,[2048,3060,4084]):
        for family in ['raw','arithmetic']:
            cs=sorted([c for c in policy['choices'] if c['N']==N and c['family']==family],key=lambda c:c['snr_db'])
            rs=[lookup[c['method'],c['snr_db']] for c in cs]
            means=np.array([r['E_mean']/N for r in rs])
            ax.errorbar([c['snr_db'] for c in cs],means,
                yerr=[means-np.array([r['E_min']/N for r in rs]),np.array([r['E_max']/N for r in rs])-means],
                fmt='.-',capsize=3,label=family)
        ax.axhline(2,color='gray',linestyle='--')
        ax.set(title=f'N={N}',xlabel='SNR (dB)',ylabel='Actual E/N: mean and min/max',xticks=[1,4,7,13,19])
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Frozen 16QAM policy; fixed constellation average energy, not per-frame 2N')
    fig.tight_layout();fig.savefig(DEST/'actual_energy.svg');plt.close(fig)
    p=DEST/'actual_energy.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')


def main():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU mask required')
    check(not DEST.exists(),'immutable publication already exists')
    cal=OUT/'digital_grid_v1/16QAM/calibration'
    policy=read(cal/'policy.json');caldone=read(cal/'completion.json')
    check(sha(cal/'policy.json')==caldone['policy_sha256'] and policy['calibration_table_sha256']==caldone['table_sha256'],'frozen calibration identity')
    published=ROOT/'results/token_channel_efficiency_20260923/QAM16_calibration_v1'
    check(sha(published/'policy.json')==sha(cal/'policy.json') and sha(published/'completion.json')==sha(cal/'completion.json'),'previously audited calibration')
    methods=sorted({r['method'] for r in csv.DictReader((published/'summary.csv').open())})
    check(len(methods)==29,'16QAM eligible roster')
    digital,dt,dw,dr,dd=load_completed('digital_grid_v1/16QAM/development','16QAM_development',methods)
    verify_development_cells(dr,dd)
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
        source=OUT/'digital_grid_v1/16QAM/development'/name
        if source.exists():shutil.copyfile(source,DEST/name)
    split_csv(OUT/'digital_grid_v1/16QAM/development/per_frame.csv',DEST/'per_frame_parts')
    for name,source in [('policy.json',cal/'policy.json'),('quality_targets.json',EXP/'quality_targets.json'),
                        ('stage_receipt.json',OUT/'delivery_chain_v1/stages/16QAM_development.json')]:shutil.copyfile(source,DEST/name)
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
    energy_rows=[{'method':r['method'],'snr_db':r['snr_db'],'N':r['N'],**{k:r[k] for k in ['E_mean','E_min','E_p05','E_p95','E_max']},'energy_constraint':'fixed_constellation_average_2_per_symbol','table_sha256':table.sha256} for r in digital.summary()]
    csvwrite(DEST/'actual_energy.csv',energy_rows)
    energy_figure(energy_rows,policy)
    lineage={str(OUT/r/'completion.json'):sha(OUT/r/'completion.json') for r in ['digital_grid_v1/16QAM/calibration','digital_grid_v1/16QAM/development','continuous_grid_v1']}
    write(DEST/'audit.json',{'status':'REAL_16QAM_DEVELOPMENT_AND_MATCHED_CONTINUOUS_VERIFIED','synthetic':False,
        'combined_table_sha256':table.sha256,'digital_frame_rows':len(digital.frames),'continuous_frame_rows':len(continuous.frames),
        'digital_timings':len(dt),'digital_warmups':len(dw),'continuous_timings':len(ct),'continuous_warmups':len(cw),
        'lineage':lineage,'frozen_policy_sha256':sha(cal/'policy.json'),'full_study_complete':False})
    write(DEST/'index.json',{'publisher_sha256':sha(__file__),'table_sha256':table.sha256,
        'audit_libraries':{str(p):sha(p) for p in [ROOT/'tools/publish_qpsk_development.py',ROOT/'tools/publish_grid_milestones.py']},
        'files':{str(p.relative_to(DEST)):digest(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
        'continuous_raw_records':'../grid_milestones_v1/continuous_development/per_frame_parts/manifest.json',
        'pending':['combined B1/B2','C','historical GPU workers','final combined delivery']})
    print(json.dumps({'published':str(DEST),'table_sha256':table.sha256,'minimum_uses':[{k:r[k] for k in ['snr_db','target','family','continuous_N_required','digital_N_required','saving','status']} for r in minimum]}),flush=True)

def plots(table,policy,timing,minimum):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='16qam-development-v1'
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
        axs[0,0].legend(fontsize=8);fig.suptitle('100-source development: 16QAM average-symbol energy vs per-frame 2N continuous')
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(DEST/f'quality_vs_{axis}.svg');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,level in zip(axs,targets_order(minimum)):
        for family in ['raw','arithmetic']:
            rs=[r for r in minimum if r['target']==level and r['family']==family]
            ax.plot([r['snr_db'] for r in rs],[100*r['saving'] if r['saving'] is not None else np.nan for r in rs],'.-',label=family)
            for r in rs:
                if r['saving'] is None:ax.text(r['snr_db'],0,'NR',ha='center',fontsize=7)
        ax.axhline(0,color='grey',lw=.6);ax.set(title=level,xlabel='SNR (dB)',ylabel='Digital N saving vs continuous (%)',xlim=(0,20),ylim=(-110,60),xticks=table.snrs);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Minimum tested N; different frame-energy constraints; NR = target not reached; no interpolation')
    fig.tight_layout();fig.savefig(DEST/'minimum_N_savings.svg');plt.close(fig)
    for p in DEST.glob('*.svg'):p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')

def targets_order(rows):return list(dict.fromkeys(r['target'] for r in rows))

if __name__=='__main__':main()
