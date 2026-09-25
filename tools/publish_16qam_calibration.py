"""CPU-only immutable publication of the completed 16QAM calibration."""
import csv, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from tools.publish_grid_milestones import ROOT, OUT, METRICS, check, digest, read, write, csvwrite, verify_grid

REL = 'digital_grid_v1/16QAM/calibration'
DEST = ROOT/'results/token_channel_efficiency_20260923/QAM16_calibration_v1'

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

def figures(dest, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='16qam-calibration-publication-v1'
    for xkey,facet in [('snr_db','N'),('N','snr_db')]:
        facets=sorted({r[facet] for r in summary})
        fig,axes=plt.subplots(4,len(facets),figsize=(4*len(facets),12),squeeze=False)
        handles={}
        for i,metric in enumerate(METRICS):
            for j,value in enumerate(facets):
                groups=defaultdict(list)
                for r in summary:
                    if r[facet]==value: groups[(r['family'],int(r['method'].rsplit('/m',1)[1]))].append(r)
                ax=axes[i,j]
                for (family,m),rs in sorted(groups.items()):
                    rs=sorted(rs,key=lambda r:r[xkey]); label=f'{family} m{m}'
                    handles[label],=ax.plot([r[xkey] for r in rs],[r[metric] for r in rs],
                        '.--' if family=='raw' else '.-',color=f'C{m-6}',label=label)
                ax.set(xlabel='SNR (dB)' if xkey=='snr_db' else 'N (complex channel uses)',
                       ylabel=metric,title=f'{facet} = {value:g}')
                ax.set_xticks(sorted({r[xkey] for r in summary}));ax.grid(alpha=.2)
        fig.legend(handles.values(),handles.keys(),loc='lower center',ncol=5)
        fig.suptitle('16QAM calibration: all eligible candidates; noise averaged within source')
        fig.tight_layout(rect=(0,.06,1,.965));fig.savefig(dest/f'quality_{xkey}.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,N in zip(axes,[2048,3060,4084]):
        rs=sorted([r for r in summary if r['N']==N and r['snr_db']==1],key=lambda r:r['method'])
        labels=[r['family']+' m'+r['method'].rsplit('/m',1)[1] for r in rs]
        means=np.array([r['E_mean']/N for r in rs])
        ax.errorbar(range(len(rs)),means,
            yerr=[means-np.array([r['E_min']/N for r in rs]),np.array([r['E_max']/N for r in rs])-means],
            fmt='o',capsize=3)
        ax.axhline(2,linestyle='--',color='gray');ax.set_xticks(range(len(rs)),labels,rotation=90)
        ax.set(title=f'N = {N}',ylabel='Actual E / N (mean and min/max)',xlabel='Fixed candidate')
    fig.suptitle('16QAM fixed constellation: average-symbol constraint; frame energy is not forced to 2N')
    fig.tight_layout();fig.savefig(dest/'actual_energy.svg');plt.close(fig)

def main():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU mask required')
    check(not DEST.exists(),'immutable publication already exists')
    folder=OUT/REL;reg=read(folder/'registration.json');done=read(folder/'completion.json')
    candidates=read(folder/'candidates.json');eligible=[c for c in candidates if c['status']=='ELIGIBLE']
    check(len(candidates)==30 and len(eligible)==29,'registered candidate roster')
    check(reg['seeds']==[4101,4102,4103] and reg['role']=='calibration' and reg['mcs']=='16QAM','calibration axes')
    expected={f'{i:04d}_{j:02d}.json' for i in range(1000) for j in range(29)}
    check(set(done['cell_sha256'])==expected,'complete cell names')
    check({p.name for p in (folder/'cells').glob('*.json') if not p.name.endswith('.seal.json')}==expected,
          'no extra unsealed cells')
    energy=defaultdict(list)
    for name in sorted(expected):
        i,j=map(int,Path(name).stem.split('_'));cell=read(folder/'cells'/name)
        check(read((folder/'cells'/name).with_suffix('.seal.json'))['registration_sha256']==done['registration_sha256'],
              'seal registration')
        sid=reg['image_bindings'][i]['image_id'];candidate=eligible[j]
        for row in cell['rows']:
            verify_row(row,candidate,sid,i,reg)
            energy[(row['method'],row['snr_db'])].append(row['E'])
    summary=verify_grid(REL,'16QAM_calibration',DEST,1000,29)
    csvwrite(DEST/'energy_summary.csv',[{'method':r['method'],'snr_db':r['snr_db'],'N':r['N'],
        **{k:r[k] for k in ['E_mean','E_min','E_p05','E_p95','E_max']},
        'mean_energy_per_use':r['E_mean']/r['N'],'constraint':'fixed_constellation_average_2_per_symbol',
        'table_sha256':done['table_sha256']} for r in summary])
    policy=read(folder/'policy.json')
    csvwrite(DEST/'policy_choices.csv',[{k:c[k] for k in ['family','N','snr_db','method','calibration_utility']}
        for c in policy['choices']])
    figures(DEST,summary)
    for svg in DEST.glob('*.svg'):
        svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    with (DEST/'failure_counts.csv').open() as f: failures=list(csv.DictReader(f))
    totals={k:sum(int(r[k]) for r in failures) for k in ['header_ok_failure_count','body_crc_ok_failure_count','source_overflow_erasure']}
    write(DEST/'publication_receipt.json',{'status':'REAL_16QAM_CALIBRATION_AUDITED',
        'frames':435000,'cells':29000,'policy_choices':len(policy['choices']),'failure_totals':totals,
        'table_sha256':done['table_sha256'],'policy_sha256':done['policy_sha256'],
        'energy_min':min(r['E_min'] for r in summary),'energy_max':max(r['E_max'] for r in summary),
        'development_savings':'PENDING_COMPLETE_DEVELOPMENT','study_complete':False})
    write(DEST/'index.json',{'status':'REAL_CALIBRATION_PUBLISHED_FULL_STUDY_PENDING',
        'publisher_sha256':digest(__file__),'audit_library_sha256':digest(ROOT/'tools/publish_grid_milestones.py'),
        'files':{str(p.relative_to(DEST)):digest(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
        'pending':['16QAM development and paired resource statistics','B1/B2','C',
                   'historical GPU metrics, compatibility and timing','final merged report and remote validation']})
    print('PUBLISHED',DEST,totals,flush=True)

if __name__=='__main__':main()
