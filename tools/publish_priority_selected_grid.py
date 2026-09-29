"""Independent CPU publication of the completed, explicitly partial 14-model grid."""
import argparse
import csv
import itertools
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
from tools.publish_grid_milestones import ROOT, OUT, EXP, check, read, write, csvwrite, split_csv, digest
from tools.run_cpu_checks import SOURCE_DIRS
for _source in SOURCE_DIRS:sys.path.insert(0,str(_source))
from token_efficiency.statistics import FrameTable, METRICS
from token_efficiency.resource_report import read_grid, export_compact
from token_efficiency.C_report import grouped_pair, calibration_diagnostic
from short_prefix.protocol import allocation, SIZES

PRIORITY=OUT/'C_priority_selected_v1'
GRID=PRIORITY/'selected_grid'
DEST=ROOT/'results/token_channel_efficiency_20260923/C_priority_selected_development_v1'

def exact_keys(rows, fields, expected, label):
    keys=[tuple(r[k] for k in fields) for r in rows]
    check(len(keys)==len(set(keys)) and set(keys)==set(expected), label)

def resource_ledger(row):
    if row['family']=='continuous':
        check(row['N_header']==row['N_data']==0 and row['N_continuous']==row['N'], 'pure segments')
        return dict(source_payload_bits='not_applicable', header_information_bits=0,
                    header_crc_bits=0, header_tail_bits=0, header_mother_bits=0,
                    header_coded_slots=0, body_crc_bits=0, body_tail_bits=0,
                    body_mother_bits=0, body_coded_slots=0)
    m=int(row['m_actual']); a=allocation(m,row['N'])
    check([row['N_header'],row['N_data'],row['N_continuous']]==[a['NH'],a['ND'],a['NA']], 'hybrid segments')
    bits=12*sum(s*s for s in SIZES[:m])
    return dict(source_payload_bits=bits, header_information_bits=12,
                header_crc_bits=16, header_tail_bits=6, header_mother_bits=68,
                header_coded_slots=136, body_crc_bits=16, body_tail_bits=6,
                body_mother_bits=2*(bits+22), body_coded_slots=2*a['ND'])

def audit():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'explicit CPU mask required')
    rows,timing,reg,done,_=read_grid(GRID)
    spec=read(PRIORITY/'spec.json'); methods=[r['method'] for r in spec['models']]
    check(len(methods)==14 and len(set(methods))==14 and spec['new_holdout'] is False, 'frozen early scope')
    check(done['status']=='REAL_C_SELECTED_GRID_COMPLETE' and done['new_holdout'] is False, 'real completion')
    check(len(rows)==done['frame_rows']==21000 and len(timing)==done['timed_calls']==1400, 'full primary grid')
    check(len(reg['sources'])==100 and reg['snrs']==[1,4,7,13,19] and reg['seeds']==[2001,2002,2003], 'axes')
    table=FrameTable(rows,'development',reg['sources'],reg['snrs'],reg['seeds'],methods)
    check(table.sha256==done['table_sha256'], 'canonical table')
    qual=read(GRID/'qualification.json')
    check(qual['status']=='REAL_C_SELECTED_CACHE_ONLINE_REPLAY_PASS' and qual['synthetic'] is False
          and qual['registration_sha256']==done['registration_sha256'], 'real qualification')
    checks=qual['checks']
    check(len(checks)==56 and all(r['cache_training_online_match'] is True for r in checks), '56 qualification checks')
    for method in methods:check(sum(r['method']==method for r in checks)==4, 'qualification roster')
    for name,h in read(PRIORITY/'registration.json')['bindings'].items():check(digest(name)==h,'priority binding '+name)
    warm=[];base=[];pairs=[]
    for name,h in done['cell_sha256'].items():
        p=GRID/'cells'/name;c=read(p);seal=read(p.with_suffix('.seal.json'))
        check(seal=={'sha256':h,'registration_sha256':done['registration_sha256']}, 'seal identity')
        method=c['method'];sid=c['source_id'];pairs.append((method,sid))
        check(all(r['method']==method and r['source_id']==sid for r in c['rows']), 'cell identity')
        exact_keys(c['rows'],['snr_db','noise_seed'],itertools.product(reg['snrs'],reg['seeds']),'cell full frame keys')
        meta=reg['context']['models'][method]
        for r in c['rows']:
            check(r['training_seed']==meta['training_seed'] and r['selected_step']==meta['selected']['step']
                  and r['training_completed_step']==meta['training_completed_step'], 'selected lineage')
            resource_ledger(r)
        expected=itertools.product(reg['snrs'],reg['seeds'],['B_RX','C_RX']) if meta['kind']=='hybrid' else []
        exact_keys(c['base_rows'],['snr_db','noise_seed','condition'],expected,'received-base complete keys')
        lookup={(r['snr_db'],r['noise_seed']):r for r in c['rows']}
        for r in c['base_rows']:
            parent=lookup[r['snr_db'],r['noise_seed']]
            check(r['method']==method and r['source_id']==sid and r['N_paid']==parent['N']
                  and r['enhancement_removed'] is True and r['header_ok']==parent['header_ok']
                  and r['body_crc_ok']==parent['body_crc_ok'], 'base RX identity')
            check(all(np.isfinite(r[k]) for k in METRICS), 'base finite metrics')
        warm.extend(c['warmup']);base.extend(c['base_rows'])
    check(len(pairs)==1400 and len(set(pairs))==1400 and set(pairs)==set(itertools.product(methods,reg['sources'])), 'cell population')
    ids=[list(reg['sources'])[i] for i in read(EXP/'protocol.json')['fixed_examples']['development_indices']]
    exact_keys(timing,['method','source_id','snr_db','noise_seed','repeat'],itertools.product(methods,ids,reg['snrs'],[2001],[0,1]),'timing keys')
    exact_keys(warm,['method','source_id','repeat'],itertools.product(methods,ids,range(3)),'warmup keys')
    check(len(base)==36000 and len(warm)==420, 'diagnostic/warmup counts')
    for r in timing:
        p=table.frames[r['method'],r['source_id'],r['snr_db'],r['noise_seed']]
        check(r['context_sha256']==p['context_sha256'] and r['N']==p['N'] and abs(r['E']-p['E'])<.02, 'timing context/resource')
        check(0<=r['quality_max_abs_error']<=2e-5, 'timing quality replay')
    for r in timing+warm:
        check(all(np.isfinite(r[k]) and r[k]>0 for k in ['tx_ms','rx_ms','total_ms'])
              and abs(r['total_ms']-r['tx_ms']-r['rx_ms'])<1e-6, 'timing arithmetic')
    for filename,expected in [('per_frame.csv',rows),('timing.csv',timing)]:
        with (GRID/filename).open() as f:actual=list(csv.DictReader(f))
        check(len(actual)==len(expected),'original CSV count')
        check(all(set(b)<=set(a) and all(a[k]==('' if b.get(k) is None else str(b[k])) for k in a) for a,b in zip(actual,expected)),'original CSV complete fields')
    old={(r['method'],float(r['snr_db'])):r for r in csv.DictReader((GRID/'summary.csv').open())}
    check(len(old)==70,'original summary count')
    for r in table.summary():
        for k in [*METRICS,'E_mean','E_min','E_p05','E_p95','E_max']:
            check(abs(float(old[r['method'],r['snr_db']][k])-r[k])<1e-9,'summary replay '+k)
    return table,timing,warm,base,reg,done

def name(arm,N=4084,seed=2026092304):return f'{arm}_N{N}_seed{seed}'

def resource_comparisons(table):
    targets=read(EXP/'quality_targets.json')
    check(targets['status']=='FROZEN_FROM_CALIBRATION_BEFORE_NEW_DEVELOPMENT','frozen targets')
    # Fixed calibration-selected m6, first seed only. No development method selection.
    rosters={'continuous':['P2048','P3060',name('P4084')],
             'hybrid_V':[name('H6-V',3060),name('H6-V')],
             'hybrid_P':[name('H6-P',3060),name('H6-P')]}
    out=[]
    for snr in table.snrs:
        methods=sum(rosters.values(),[]);v=np.stack([table.source_values(m,snr)[:,[1,2]] for m in methods])
        budgets=np.array([table.context[m]['N'] for m in methods]);rng=np.random.default_rng(20260923);draws=[]
        for _ in range(40):draws.append(v[:,rng.integers(0,len(table.ids),(250,len(table.ids))),:].mean(2))
        draws=np.concatenate(draws,axis=1);point=v.mean(1)
        for level,target in targets['targets'].items():
            met=(draws[:,:,0]>=target['psnr_min'])&(draws[:,:,1]<=target['lpips_max'])
            pm=(point[:,0]>=target['psnr_min'])&(point[:,1]<=target['lpips_max'])
            cn=np.where(met[:3],budgets[:3,None],np.inf).min(0)
            pc=float(np.where(pm[:3],budgets[:3],np.inf).min())
            for family,ix in [('hybrid_V',[3,4]),('hybrid_P',[5,6])]:
                hn=np.where(met[ix],budgets[ix,None],np.inf).min(0)
                ph=float(np.where(pm[ix],budgets[ix],np.inf).min());valid=np.isfinite(cn)&np.isfinite(hn)
                savings=1-hn[valid]/cn[valid]
                out.append(dict(snr_db=snr,target=level,family=family,
                    continuous_min_N=int(pc) if np.isfinite(pc) else None,
                    hybrid_min_N=int(ph) if np.isfinite(ph) else None,
                    saving=1-ph/pc if np.isfinite(pc) and np.isfinite(ph) else None,
                    repeats=10000,both_attained_repeats=int(valid.sum()),
                    continuous_unreached=int((~np.isfinite(cn)).sum()),hybrid_unreached=int((~np.isfinite(hn)).sum()),
                    conditional_saving_ci95=np.quantile(savings,[.025,.975]).tolist() if len(savings) else None,
                    condition='both families attain frozen mean PSNR/LPIPS target in tested grid',
                    rosters=rosters,training_seed_variation_included=False,policy_refit=False,
                    table_sha256=table.sha256))
    return out

def publish():
    check(not DEST.exists(),'immutable publication already exists')
    primary,timing,warm,base,reg,done=audit()
    rows=list(primary.frames.values());contexts={reg['context_sha256']:reg['context']};lineage={str(GRID/'completion.json'):digest(GRID/'completion.json')}
    # Keep old P4084 10k distinct from the newly trained P4084 selected model.
    for folder in [OUT/'continuous_grid_v1',OUT/'digital_grid_v1/QPSK/development',OUT/'digital_grid_v1/16QAM/development']:
        rr,tt,r,d,_=read_grid(folder)
        check(r['sources']==reg['sources'] and r['context']['decoder_sha256']==reg['context']['decoder_sha256'], 'reference population/decoder')
        check(FrameTable(rr,'development',r['sources'],r['snrs'],r['seeds'],sorted({x['method'] for x in rr})).sha256==d['table_sha256'],'reference table')
        rows.extend(rr);timing.extend(tt);contexts[r['context_sha256']]=r['context'];lineage[str(folder/'completion.json')]=digest(folder/'completion.json')
    table=FrameTable(rows,'development',reg['sources'],reg['snrs'],reg['seeds'],sorted({r['method'] for r in rows}))
    check(len(table.frames)==108000 and len(timing)==7200,'combined population')
    pairs={(name(a),name('P4084')) for a in ['H6-V','H7-V','H8-V']}
    pairs|={(name(f'H{m}-V'),name(f'H{m}-P')) for m in [6,7]}
    pairs|={(name(a,3060),'P3060') for a in ['H6-V','H6-P']}
    pairs.add((name('H6-V',3060),name('H6-P',3060)))
    for seed in [2026092404,2026092504]:
        for a in ['H6-P','H8-V','P4084']:
            if name(a,seed=seed) in table.methods:pairs.add((name('H6-V',seed=seed),name(a,seed=seed)))
    comparisons=[grouped_pair(table,a,b,s) for a,b in sorted(pairs) for s in [[1],[4],[7],[13],[19],[1,4,7],[13,19],[1,4,7,13,19]]]
    policies={}
    for mcs in ['QPSK','16QAM']:
        p=OUT/'digital_grid_v1'/mcs/'calibration/policy.json';policies[mcs]=read(p);lineage[str(p)]=digest(p)
        check(digest(p)==read(p.parent/'completion.json')['policy_sha256'],'frozen digital policy')
        for c in policies[mcs]['choices']:
            arms=['H6-V','H7-V','H8-V'] if c['N']==4084 else ['H6-V'] if c['N']==3060 else []
            comparisons.extend(grouped_pair(table,name(a,c['N']),c['method'],[c['snr_db']]) for a in arms)
    minimum=resource_comparisons(table);summary=table.summary()
    for r in summary:r['U_image']=r['mse']+.1*r['lpips_alex']
    lookup={(r['method'],r['snr_db']):r for r in summary}
    variation=[]
    for arm in ['H6-V','H6-P','H8-V','P4084']:
        seeds=[s for s in [2026092304,2026092404,2026092504] if name(arm,seed=s) in primary.methods]
        for snr in table.snrs:
            for metric in [*METRICS,'U_image']:
                v=[lookup[name(arm,seed=s),snr][metric] for s in seeds]
                variation.append(dict(arm=arm,N=4084,snr_db=snr,metric=metric,training_seeds=json.dumps(seeds),seed_count=len(seeds),mean=float(np.mean(v)),sample_std=float(np.std(v,ddof=1)),min=min(v),max=max(v),image_bootstrap_interval=False))
    ts=[]
    for m in table.methods:
        for snr in table.snrs:
            rr=[r for r in timing if r['method']==m and r['snr_db']==snr]
            ts.append(dict(method=m,snr_db=snr,calls=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['tx_ms','rx_ms','total_ms']},table_sha256=table.sha256))
    diag,diagmeta=calibration_diagnostic()
    DEST.mkdir(parents=True)
    for n in ['registration.json','qualification.json','completion.json','summary.csv','timing.csv']:shutil.copyfile(GRID/n,DEST/n)
    for n in ['spec.json','handoff.json','evaluation_verified.json']:shutil.copyfile(PRIORITY/n,DEST/('priority_'+n))
    split_csv(GRID/'per_frame.csv',DEST/'per_frame_parts')
    write(DEST/'execution_contexts.json',contexts);write(DEST/'selected.json',reg['context']['models']);write(DEST/'lineage.json',lineage)
    export_compact(table,DEST/'combined_development')
    csvwrite(DEST/'combined_summary.csv',summary);csvwrite(DEST/'combined_timing_summary.csv',ts)
    csvwrite(DEST/'warmup.csv',warm);write(DEST/'paired.json',comparisons);write(DEST/'hybrid_minimum_uses.json',minimum)
    write(DEST/'frozen_policies.json',policies);shutil.copyfile(EXP/'quality_targets.json',DEST/'quality_targets.json')
    csvwrite(DEST/'training_seed_variation.csv',variation);csvwrite(DEST/'source_choice_diagnostic.csv',diag);write(DEST/'source_choice_diagnostic.json',diagmeta)
    for i in range(0,len(base),10000):csvwrite(DEST/f'received_base_{i//10000:03d}.csv',base[i:i+10000])
    ledgers=[]
    for r in sorted(primary.frames.values(),key=lambda r:(r['method'],r['source_id'],r['snr_db'],r['noise_seed'])):
        ledgers.append({k:r[k] for k in ['method','source_id','preprocessing_id','snr_db','noise_seed','run_id','context_sha256','N','N_header','N_data','N_continuous','E','header_ok','body_crc_ok']}|resource_ledger(r)|{'bit_fields_basis':'derived from frozen fixed-mode PHY, not decoded TX truth','E_basis':'actual online waveform measurement'})
    for i in range(0,len(ledgers),5000):csvwrite(DEST/f'resource_ledger_{i//5000:03d}.csv',ledgers[i:i+5000])
    (DEST/'source_means').mkdir()
    for i,m in enumerate(table.methods):
        csvwrite(DEST/'source_means'/f'method_{i:02d}.csv',[dict(method=m,source_id=sid,preprocessing_id=table.sources[sid],snr_db=s,**dict(zip(METRICS,map(float,v))),table_sha256=table.sha256) for s in table.snrs for sid,v in zip(table.ids,table.source_values(m,s))])
    failures=[]
    for m in primary.methods:
        for snr in table.snrs:
            rr=[r for r in primary.frames.values() if r['method']==m and r['snr_db']==snr]
            failures.append(dict(method=m,snr_db=snr,frames=len(rr),header_failures=sum(r['header_ok'] is False for r in rr),body_crc_failures=sum(r['body_crc_ok'] is False for r in rr),source_overflow_erasures=sum(r['source_overflow_erasure'] is True for r in rr)))
    csvwrite(DEST/'failures.csv',failures)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tlookup={(r['method'],r['snr_db']):r['total_ms'] for r in ts}
    for metric in ['psnr_db','lpips_alex','dino_cosine','U_image']:
        for axis in ['N','time']:
            fig,axes=plt.subplots(1,5,figsize=(19,4))
            for ax,snr in zip(axes,table.snrs):
                rosters={'continuous':['P2048','P3060',name('P4084')],'H6-V':[name('H6-V',3060),name('H6-V')],'H6-P':[name('H6-P',3060),name('H6-P')]}
                for mcs,policy in policies.items():
                    for fam in ['raw','arithmetic']:rosters[mcs+' '+fam]=[c['method'] for c in sorted(policy['choices'],key=lambda c:c['N']) if c['snr_db']==snr and c['family']==fam]
                for label,ms in rosters.items():
                    x=[table.context[m]['N'] if axis=='N' else tlookup[m,snr] for m in ms]
                    ax.plot(x,[lookup[m,snr][metric] for m in ms],'o-',label=label,markersize=3)
                ax.set(title=f'{snr} dB',xlabel='Complex uses N' if axis=='N' else 'Online TX+RX ms',ylabel=metric);ax.grid(alpha=.2)
            axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(DEST/f'quality_{axis}_{metric}.svg');plt.close(fig)
    write(DEST/'audit.json',dict(status='REAL_PRIORITY_14_SELECTED_DEVELOPMENT_VERIFIED',synthetic=False,new_holdout=False,full_delivery=False,
        primary_frames=21000,received_base_rows=36000,timed_calls=1400,warmup_calls=420,qualification_checks=56,combined_frames=len(table.frames),combined_timed_calls=len(timing),primary_table_sha256=primary.sha256,combined_table_sha256=table.sha256,
        source_bootstrap_repeats=10000,seed_variation_separate=True,partial_seed_coverage={'H6-V':3,'H6-P':3,'H8-V':2,'P4084':2},
        pending=['third-seed H8-V and fresh P4084 training/evaluation','original full 16-model grid and final report','four historical GPU workers and compatibility/timing','final all-method delivery'],
        no_timing_from_training_or_calibration=True,no_deployed_selector=True))
    files={str(p.relative_to(DEST)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in DEST.rglob('*') if p.is_file()}
    check(all(x['bytes']<10_000_000 for x in files.values()),'lightweight publication files')
    write(DEST/'index.json',{'files':files,'audit_sha256':digest(DEST/'audit.json')})
    print(json.dumps({'destination':str(DEST),'files':len(files),'primary_sha256':primary.sha256,'combined_sha256':table.sha256}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:
        t,tt,w,b,r,d=audit();print(json.dumps({'status':'REAL_PRIORITY_GRID_CPU_AUDIT_PASS','frames':len(t.frames),'timed_calls':len(tt),'warmup':len(w),'base_rows':len(b),'table_sha256':t.sha256}))
    else:publish()
if __name__=='__main__':main()
