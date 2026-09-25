"""CPU audit of the scheduler's immutable B1/B2 compact publication."""
import csv, json, os
from pathlib import Path
from tools.publish_grid_milestones import ROOT, OUT, check, read, write, digest
from token_efficiency.statistics import FrameTable, minimum_tested_uses, paired

DEST=ROOT/'results/token_channel_efficiency_20260923/B1_B2_v1'
REVIEW=ROOT/'results/token_channel_efficiency_20260923/B1_B2_review_v1'

FIELDS=('waveform_sha256','observation_sha256','snr_db','noise_seed','mse','psnr_db','lpips_alex','dino_cosine','E','header_ok','body_crc_ok','source_overflow_erasure','source_complete','m_actual','payload_bits','mother_bits','coded_slots','length_field','overflow_lower_m','decoded_mode','decoded_label','source_error','header_crc_ok','rx_source_overflow_erasure')

def compare_compact(row,original):
    check(set(row)=={'source_index','method_index',*FIELDS}, 'complete compact fields')
    for key,value in row.items():
        if key in ['source_index','method_index']:continue
        expected=original.get(key,'')
        check(value==('' if expected is None else str(expected)), 'compact frame field '+key)

def compact_grid(folder,grids):
    index=read(folder/'index.json')
    contexts={};all_sources=None;expected_rows=0
    for rel in grids:
        grid=OUT/rel;reg=read(grid/'registration.json');done=read(grid/'completion.json')
        if all_sources is None:all_sources=reg['sources']
        check(all_sources==reg['sources'],'same population')
        methods=sorted({r['method'] for r in csv.DictReader((grid/'summary.csv').open())})
        if rel=='continuous_grid_v1':ordered=['P2048','P3060','P4084']
        else:ordered=[c['method'] for c in read(grid/'candidates.json') if c['status']=='ELIGIBLE']
        check(set(methods)==set(ordered),'explicit method roster')
        for method in methods:contexts[method]=(grid,reg,done,ordered.index(method))
        expected_rows+=done['frame_rows']
    check(index['sources']==[{'source_id':s,'preprocessing_id':h} for s,h in all_sources.items()],'compact source index')
    check([m['method'] for m in index['methods']]==sorted(contexts),'compact method index')
    ids=list(all_sources);last=None;cell=None;keys=set();count=0;development_rows=[]
    for file in sorted(folder.glob('frames_*.csv')):
        for row in csv.DictReader(file.open()):
            mi=int(row['method_index']);si=int(row['source_index'])
            check(0<=mi<len(index['methods']) and 0<=si<len(ids),'compact integer indices')
            method=index['methods'][mi]['method'];sid=ids[si]
            grid,reg,done,j=contexts[method];cellkey=(method,si)
            if last!=cellkey:
                name=f'{si:04d}_{int(method[1:])}.json' if method.startswith('P') else f'{si:04d}_{j:02d}.json'
                p=grid/'cells'/name;cell=read(p)
                seal=read(p.with_suffix('.seal.json'))
                check(digest(p)==done['cell_sha256'][name]==seal['sha256'],'immutable compact source cell')
                check(cell['registration_sha256']==seal['registration_sha256']==done['registration_sha256'],'compact cell registration')
                check(cell['method']==method and cell['source_id']==sid,'compact original method/source')
                bykey={(r['snr_db'],r['noise_seed']):r for r in cell['rows']};last=cellkey
            k=(int(row['snr_db']),int(row['noise_seed']))
            original=bykey[k]
            check(original['preprocessing_id']==all_sources[sid],'compact preprocessing')
            check(all(index['methods'][mi][field]==original[field] for field in index['methods'][mi]),'compact context')
            compare_compact(row,original)
            key=(method,sid,*k);check(key not in keys,'duplicate compact frame');keys.add(key);count+=1
            if original['population']=='development':development_rows.append(original)
    check(count==expected_rows,'complete compact population')
    return count,development_rows,index

def main():
    check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU mask')
    check(not REVIEW.exists(),'immutable review already exists')
    done=read(DEST/'completion.json')
    check(done['synthetic'] is False and done['frame_rows']==87000,'real full B1/B2')
    for name,h in done['files'].items():check(digest(DEST/name)==h,'generated artifact '+name)
    stage=read(OUT/'delivery_chain_v1/stages/B1_B2_resource_report.json')
    check(stage['returncode']==0 and digest(stage['snapshot'])==stage['completion_sha256']==digest(DEST/'completion.json'),'report stage')
    check(digest(ROOT/'reports/token_channel_efficiency_B1_B2_20260924.md')==done['report_sha256'],'report identity')
    lineage=read(DEST/'lineage.json')
    for p,h in lineage.items():check(digest(p)==h,'report lineage')
    counts={}
    for mcs in ['QPSK','16QAM']:
        counts[mcs+'_calibration'],_,_=compact_grid(DEST/'calibration'/mcs,[f'digital_grid_v1/{mcs}/calibration'])
    counts['development'],rows,index=compact_grid(DEST/'development',['digital_grid_v1/QPSK/development','digital_grid_v1/16QAM/development','continuous_grid_v1'])
    reg=read(OUT/'continuous_grid_v1/registration.json')
    table=FrameTable(rows,'development',reg['sources'],reg['snrs'],reg['seeds'],[r['method'] for r in index['methods']])
    check(table.sha256==index['table_sha256']==done['table_sha256'],'complete common development object')
    summary=list(csv.DictReader((DEST/'summary.csv').open()))
    expected=table.summary();check(len(summary)==len(expected),'summary count')
    for a,b in zip(summary,expected):
        for k,v in b.items():check(a[k]==str(v),'summary '+k)
    minimum=minimum_tested_uses(table,read(DEST/'policy.json'),read(ROOT/'experiments/token_channel_efficiency_20260923/quality_targets.json'),['P2048','P3060','P4084'])
    check(minimum==read(DEST/'minimum_uses.json'),'minimum N / failures / savings')
    policy=read(DEST/'policy.json')
    checks=[paired(table,c['method'],f"P{c['N']}",c['snr_db']) for c in policy['choices']]
    check(checks==read(DEST/'paired.json'),'complete paired source-bootstrap statistics')
    timing=list(csv.DictReader((DEST/'timing.csv').open()))
    originals=[]
    for rel in ['digital_grid_v1/QPSK/development','digital_grid_v1/16QAM/development','continuous_grid_v1']:
        originals.extend(csv.DictReader((OUT/rel/'timing.csv').open()))
    originals=[{k:r.get(k,'') for k in timing[0]} for r in originals]
    check(timing==originals and len(timing)==5800,'all previously audited complete online timing rows')
    REVIEW.mkdir()
    write(REVIEW/'audit.json',{'status':'REAL_B1_B2_PUBLICATION_INDEPENDENT_AUDIT_PASS','synthetic':False,
        'compact_frame_counts':counts,'total_compact_frames':sum(counts.values()),'development_table_sha256':table.sha256,
        'original_artifacts':len(done['files']),'paired_comparisons':len(checks),'online_timings':len(timing),
        'completion_sha256':digest(DEST/'completion.json'),'lineage':lineage,
        'remaining':['C first matrix and followups','four historical GPU workers','final all-method paired delivery'],
        'full_study_complete':False})
    write(REVIEW/'index.json',{'auditor_sha256':digest(__file__),'files':{'audit.json':digest(REVIEW/'audit.json')}})
    print('B1_B2_AUDITED',counts,table.sha256,flush=True)

if __name__=='__main__':main()
