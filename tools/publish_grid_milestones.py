"""Publish completed real grids without touching any active execution dependency.

Per-frame CSV parts preserve every original byte (concatenate in manifest order).
Only metrics, ledgers, hashes and receipts are exported; no pixels or tensors.
"""
import csv, hashlib, itertools, json, os, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/token_channel_efficiency_20260923'
OUT = ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
sys.path.insert(0, str(EXP/'src'))
from token_efficiency.statistics import FrameTable, METRICS, paired, target_rates

def check(ok, message):
    if not ok: raise RuntimeError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def csvwrite(path, rows):
    rows=list(rows)
    check(bool(rows), 'empty derived table')
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def split_csv(source, destination, limit=7_500_000):
    destination.mkdir()
    files=[];handle=None;size=0
    with Path(source).open('rb') as f:
        for line in f:
            if handle is None or size+len(line)>limit:
                if handle: handle.close()
                name=f'part_{len(files):03d}.csv';files.append(name)
                handle=(destination/name).open('wb');size=0
            handle.write(line);size+=len(line)
    if handle: handle.close()
    h=hashlib.sha256()
    for name in files: h.update((destination/name).read_bytes())
    check(h.hexdigest()==digest(source),'lossless CSV parts')
    write(destination/'manifest.json', {'format':'concatenate parts in order as bytes; header only in first part',
          'original_sha256':h.hexdigest(),'original_bytes':Path(source).stat().st_size,
          'parts':[{'path':n,'sha256':digest(destination/n),'bytes':(destination/n).stat().st_size} for n in files]})

def verify_grid(relative, stage, destination, expected_sources, expected_methods):
    folder=OUT/relative;done=read(folder/'completion.json');reg=read(folder/'registration.json')
    check(done['synthetic'] is False and done['frame_rows']==expected_sources*expected_methods*15,'real complete scope')
    check(digest(folder/'registration.json')==done['registration_sha256'],'registration SHA')
    receipt=read(OUT/'delivery_chain_v1/stages'/f'{stage}.json')
    check(receipt['returncode']==0 and digest(receipt['snapshot'])==receipt['completion_sha256']
          and digest(folder/'completion.json')==receipt['completion_sha256'],'stage completion identity')
    for name,sha in reg['context']['bindings'].items(): check(digest(name)==sha,'active binding '+name)
    for name,sha in done['files'].items():check(digest(folder/name)==sha,'result SHA '+name)
    sources=reg['sources'];snrs=reg['snrs'];seeds=reg['seeds']
    check(len(sources)==expected_sources and snrs==[1,4,7,13,19] and len(seeds)==3,'population axes')
    check(len(done['cell_sha256'])==expected_sources*expected_methods,'cell count')
    groups={}
    for name in done['cell_sha256']:
        groups.setdefault(Path(name).stem.split('_',1)[1],[]).append(name)
    roster={}
    for suffix,names in groups.items():
        c=read(folder/'cells'/names[0]);roster[c['method']]=names
    check(len(roster)==expected_methods,'method roster')
    population=reg.get('population',reg.get('role'))
    all_summaries=[];timing=[];warm=[];continuous_rows=[];failure=[];policy_scores={}
    h=hashlib.sha256();h.update(b'[');first=True
    destination.mkdir(parents=True)
    (destination/'source_means').mkdir()
    for mi,method in enumerate(sorted(roster)):
        rows=[]
        for name in sorted(roster[method]):
            path=folder/'cells'/name;c=read(path);sha=digest(path)
            check(sha==done['cell_sha256'][name]==read(path.with_suffix('.seal.json'))['sha256'],'cell seal')
            check(c['registration_sha256']==done['registration_sha256'] and c['method']==method,'cell execution identity')
            for name,sha in c['previews'].items():check(digest(name)==sha,'local preview integrity')
            rows.extend(c['rows']);timing.extend(c['timing']);warm.extend(c['warmup'])
        table=FrameTable(rows,population,sources,snrs,seeds,[method])
        for key in sorted(table.frames):
            if not first:h.update(b',')
            h.update(json.dumps(table.frames[key],sort_keys=True,separators=(',',':'),allow_nan=False).encode());first=False
        all_summaries.extend(table.summary())
        source_means=[]
        for snr in snrs:
            values=table.source_values(method,snr)
            policy_scores[(method,snr)]=float(np.mean(values[:,0]+.1*values[:,2]))
            for sid,v in zip(table.ids,values):
                source_means.append({'method':method,'population':population,'source_id':sid,
                    'preprocessing_id':sources[sid],'snr_db':snr,'noise_seeds':json.dumps(seeds),
                    **table.context[method],**dict(zip(METRICS,map(float,v))), 'table_sha256':done['table_sha256']})
            subset=[r for r in rows if r['snr_db']==snr]
            failure.append({'method':method,'snr_db':snr,'frames':len(subset),
                **{k+'_failure_count':sum(r[k] is False for r in subset) for k in ['header_ok','body_crc_ok']},
                'source_overflow_erasure':sum(r['source_overflow_erasure'] is True for r in subset),
                'table_sha256':done['table_sha256']})
        csvwrite(destination/'source_means'/f'method_{mi:02d}.csv',source_means)
        if population=='development':continuous_rows.extend(rows)
        print('verified',relative,mi+1,len(roster),method,flush=True)
    h.update(b']');check(h.hexdigest()==done['table_sha256'],'canonical sealed frame table identity')
    with (folder/'summary.csv').open() as f: original={(r['method'],int(r['snr_db'])):r for r in csv.DictReader(f)}
    check(len(original)==len(all_summaries),'summary coverage')
    for r in all_summaries:
        old=original[(r['method'],r['snr_db'])]
        for k in [*METRICS,'E_mean','E_min','E_p05','E_p95','E_max']:
            check(abs(float(old[k])-r[k])<1e-9,'source summary mismatch '+k)
        r['table_sha256']=done['table_sha256']
    check(len(timing)==done['timed_calls'],'timing count')
    for n in ['registration.json','completion.json','summary.csv','qualification.json','policy.json','candidates.json','timing.csv','warmup.csv']:
        if (folder/n).exists():shutil.copyfile(folder/n,destination/n)
    shutil.copyfile(OUT/'delivery_chain_v1/stages'/f'{stage}.json',destination/'stage_receipt.json')
    split_csv(folder/'per_frame.csv',destination/'per_frame_parts')
    csvwrite(destination/'failure_counts.csv',failure)
    if population=='calibration':
        policy=read(folder/'policy.json')
        check(digest(folder/'policy.json')==done['policy_sha256'] and
              policy['calibration_table_sha256']==done['table_sha256'],'frozen policy identity')
        expected={(r['family'],r['N'],r['snr_db']) for r in all_summaries}
        found=set()
        for c in policy['choices']:
            key=(c['family'],c['N'],c['snr_db']);check(key not in found,'duplicate policy');found.add(key)
            candidates={r['method']:policy_scores[(r['method'],c['snr_db'])] for r in all_summaries
                        if (r['family'],r['N'],r['snr_db'])==key}
            check(set(candidates)==set(c['candidate_scores']),'policy candidate coverage')
            for m,v in candidates.items():check(abs(v-c['candidate_scores'][m])<1e-12,'policy score')
            check(min(candidates,key=lambda m:(candidates[m],m))==c['method'],'policy minimum')
        check(found==expected,'policy coverage')
    else:
        qual=read(folder/'qualification.json')
        check(qual['status']=='REAL_SELECTED_CONTINUOUS_REPLAY_PASS' and len(qual['checks'])==12
              and all(r['actual_selected_replay_pass'] for r in qual['checks']),'actual selected acceptance')
        table=FrameTable(continuous_rows,population,sources,snrs,seeds,sorted(roster))
        check(table.sha256==done['table_sha256'],'continuous table')
        for N,s in reg['context']['selected'].items():
            check(digest(s['checkpoint'])==s['checkpoint_sha256'] and digest(s['selected'])==s['selected_sha256'],'selected lineage')
            check(all(r['checkpoint_sha256']==s['checkpoint_sha256'] and r['selected_step']==s['step']
                      for r in continuous_rows if r['N']==int(N)),'row selected identity')
        timing_keys={(r['method'],r['source_id'],r['snr_db'],r['noise_seed'],r['repeat']) for r in timing}
        ids=[list(sources)[i] for i in range(0,100,11)]
        check(timing_keys==set(itertools.product(roster,ids,snrs,[2001],[0,1])) and len(timing)==300,'timing full grid')
        check(len(warm)==90 and {(r['method'],r['source_id'],r['repeat']) for r in warm}
              ==set(itertools.product(roster,ids,range(3))),'warmup grid')
        for r in timing:
            ctx=table.context[r['method']]
            check(r['preprocessing_id']==sources[r['source_id']] and r['context_sha256']==ctx['context_sha256']
                  and r['run_id']==ctx['run_id'] and r['N']==ctx['N'],'timing context')
            check(r['quality_max_abs_error']<=2e-5 and abs(r['tx_ms']+r['rx_ms']-r['total_ms'])<1e-7,'timing replay/total')
        write(destination/'paired_budget_differences.json',[paired(table,a,b,snr) for a,b in itertools.combinations(sorted(roster),2) for snr in snrs])
        targets=read(EXP/'quality_targets.json');shutil.copyfile(EXP/'quality_targets.json',destination/'quality_targets.json')
        target_rows=[]
        for snr in snrs:
            for level,t in targets['targets'].items():
                evidence={m:target_rates(table,m,snr,t) for m in roster}
                n=min((table.context[m]['N'] for m,r in evidence.items() if r['mean_target_met']),default=None)
                target_rows.append({'snr_db':snr,'target':level,'continuous_minimum_tested_N':n,
                    'status':'MEASURED_TARGET_MET' if n is not None else 'NOT_REACHED_IN_MEASURED_GRID',
                    'interpolation_used':False,'digital_savings':'PENDING_COMPLETE_DIGITAL_DEVELOPMENT',
                    'table_sha256':table.sha256,'evidence':evidence})
        write(destination/'continuous_targets.json',target_rows)
        tr=[]
        for method in roster:
            for snr in snrs:
                subset=[r for r in timing if r['method']==method and r['snr_db']==snr]
                tr.append({'method':method,'snr_db':snr,'calls':len(subset),
                    **{k:float(np.mean([r[k] for r in subset])) for k in ['tx_ms','rx_ms','total_ms']},
                    'table_sha256':table.sha256})
        csvwrite(destination/'timing_summary.csv',tr)
    write(destination/'audit.json',{'status':'COMPLETED_REAL_GRID_AND_SOURCE_STATISTICS_VERIFIED',
        'synthetic':False,'original':str(folder),'table_sha256':done['table_sha256'],
        'frame_rows':done['frame_rows'],'sources':expected_sources,'methods':expected_methods,
        'cells_verified':len(done['cell_sha256']),'bound_files_verified':len(reg['context']['bindings']),
        'timed_calls':len(timing),'warmups':len(warm),'failures_retained':True,
        'training_seed_variation_included':False,'study_complete':False})
    return all_summaries

def figures(destination, summary, calibration=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='token-efficiency-grid-milestone-v1'
    for xkey in ['snr_db','N']:
        if calibration:
            facet = 'N' if xkey == 'snr_db' else 'snr_db'
            facets = sorted({r[facet] for r in summary})
            fig, axes = plt.subplots(4, len(facets), figsize=(4*len(facets), 12), squeeze=False)
            handles = {}
            for row, metric in enumerate(METRICS):
                for col, value in enumerate(facets):
                    ax = axes[row, col]
                    groups = {}
                    for r in summary:
                        if r[facet] == value:
                            m = int(r['method'].rsplit('/m', 1)[1])
                            groups.setdefault((r['family'], m), []).append(r)
                    for (family, m), rs in sorted(groups.items()):
                        rs = sorted(rs, key=lambda r:r[xkey])
                        label = f'{family} m{m}'
                        line, = ax.plot([r[xkey] for r in rs], [r[metric] for r in rs],
                                        marker='.', linestyle='--' if family=='raw' else '-',
                                        color=f'C{m-6}', label=label, linewidth=1.2)
                        handles[label] = line
                    ax.set(xlabel='SNR (dB)' if xkey=='snr_db' else 'N (complex channel uses)',
                           ylabel=metric, title=f'{facet} = {value:g}')
                    ax.set_xticks(sorted({r[xkey] for r in summary}))
                    ax.grid(alpha=.2)
            fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=5, fontsize=9)
            fig.suptitle('QPSK calibration: all eligible candidates; average noise within source')
            fig.tight_layout(rect=(0,.06,1,.965))
        else:
            fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
            for metric,ax in zip(METRICS,axes.flat):
                groups={}
                for r in summary:
                    key=r['method'] if xkey=='snr_db' else f"SNR {r['snr_db']:g} dB"
                    groups.setdefault(key,[]).append(r)
                for key,rs in groups.items():
                    rs=sorted(rs,key=lambda r:r[xkey])
                    ax.plot([r[xkey] for r in rs],[r[metric] for r in rs],'.-',label=key,linewidth=1)
                ax.set(xlabel='SNR (dB)' if xkey=='snr_db' else 'N (complex channel uses)',ylabel=metric)
                ax.grid(alpha=.2)
            axes.flat[0].legend(fontsize=9)
            fig.suptitle('Selected continuous: original 100-source development; average noise within source')
        fig.savefig(destination/f'quality_{xkey}.svg');plt.close(fig)
    if not calibration:
        with (destination/'timing_summary.csv').open() as f:rows=list(csv.DictReader(f))
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,k in zip(axes,['tx_ms','rx_ms','total_ms']):
            for method in sorted({r['method'] for r in rows}):
                rs=[r for r in rows if r['method']==method];ax.plot([int(r['snr_db']) for r in rs],[float(r[k]) for r in rs],'.-',label=method)
            ax.set(xlabel='SNR (dB)',ylabel=k);ax.legend();ax.grid(alpha=.2)
        fig.savefig(destination/'online_timing.svg');plt.close(fig)

def main():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU publisher requires explicit CUDA mask')
    dest=ROOT/'results/token_channel_efficiency_20260923/grid_milestones_v1'
    check(not dest.exists(),'immutable publication already exists')
    for relative,stage,label,ns,nm in [
        ('digital_grid_v1/QPSK/calibration','QPSK_calibration','QPSK_calibration',1000,26),
        ('continuous_grid_v1','continuous_development','continuous_development',100,3)]:
        d=dest/label;summary=verify_grid(relative,stage,d,ns,nm);figures(d,summary,ns==1000)
    for svg in dest.rglob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    write(dest/'index.json',{'status':'TWO_REAL_STAGES_PUBLISHED_FULL_STUDY_PENDING',
        'publisher_sha256':digest(__file__),'files':{str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file()},
        'pending':['QPSK development','16QAM calibration/development','B1/B2 combined resource report','C','historical GPU workers','final combined report and remote validation']})
    print('PUBLISHED',dest,flush=True)

if __name__=='__main__':main()
