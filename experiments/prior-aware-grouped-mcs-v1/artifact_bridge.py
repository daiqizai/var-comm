"""Hash-bound bridges between calibration stages; no development access."""
from pathlib import Path
import csv
import math
from uep_common import read,sha,seal,require,identity

def registered_states(profiles_path,output,extra_bindings=None):
    rows=read(profiles_path)
    states={p['full_state_id'] for p in rows}|{p['prefix_state_id'] for p in rows if p.get('prefix_state_id')}
    value=dict(registered_before_quality=True,development_read=False,population='calibration',
        profiles_sha256=sha(profiles_path),states=[dict(state_id=s,m=int(s.split('_')[0][1:]),K=int(s.split('_K')[1])) for s in sorted(states)],
        source_bindings=extra_bindings or {},training_updates=0)
    seal(output,value);return value

def quality_bundle(directory,output):
    directory=Path(directory);done=read(directory/'completion.json');reg=read(directory/'registration.json')
    require(done['status']=='SOURCE_QUALITY_COMPLETE' and done['population']=='calibration' and not done['development_read'],'Calibration Q not complete')
    require(sha(directory/'registration.json')==done['registration_sha256'],'Quality registration changed')
    table=directory/'quality_per_source.csv';require(done['outputs'][str(table)]==sha(table),'Quality table changed')
    rows=[];seen=set()
    with table.open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key=tuple(row[k] for k in ('source_id','state_id','receiver'))
            require(key not in seen and row['population']=='calibration','Quality row duplicate/scope mismatch');seen.add(key)
            v=float(row['dinov2_vitl14_cosine']);require(math.isfinite(v),'Nonfinite objective')
            rows.append(dict(source_id=row['source_id'],state_id=row['state_id'],receiver=row['receiver'],dinov2_vitl14_cosine=v))
    require(len(rows)==done['rows'] and done['source_ids']==reg['source_ids'][:done['sources']],'Quality source population differs')
    value=dict(population='calibration',development_read=False,synthetic=False,source_ids=done['source_ids'],calibration_source_ids=reg['source_ids'],rows=rows,
        completion_path=str(directory/'completion.json'),completion_sha256=sha(directory/'completion.json'),
        registration_sha256=sha(directory/'registration.json'),table_sha256=sha(table))
    seal(output,value);return value

def freeze_shortlist(profiles_path,screen_path,output):
    """One preset screen only; retain top3, validation, and full matched pools."""
    profiles=read(profiles_path);screen=read(screen_path)
    require(screen['stage']=='screen' and screen['source_count']==300 and not screen['development_read'] and not screen['synthetic'],'Fixed300 screen required')
    ids=set();matched=set()
    for cell in screen['cells']:
        for family in cell['independent_optima'].values():
            for result in family.get('top3',[]):
                ids.add(result['stable_id'])
                p=result['profile']
                if p['G']==2:matched.add((p['N'],p['m'],p['K'],p['j']))
        ids.add(cell['strong_baseline']['stable_id'])
        neighbor=cell['frozen_distinct_validation_candidate']
        if neighbor['status']=='FROZEN':ids.add(neighbor['selected']['stable_id'])
        for family in cell['matched_to_B3'].values():
            if 'selected' in family:ids.add(family['selected']['stable_id'])
    matched|={(p['N'],p['m'],p['K'],p['j']) for p in profiles if p['stable_id'] in ids and p['G']==2}
    ids|={p['stable_id'] for p in profiles if p['G']==2 and (p['N'],p['m'],p['K'],p['j']) in matched}
    rows=[p for p in profiles if p['stable_id'] in ids]
    require(ids=={p['stable_id'] for p in rows} and rows,'Shortlist profile missing')
    seal(output,rows)
    receipt=dict(status='FIXED300_SHORTLIST_FROZEN',screen_sha256=sha(screen_path),all_profiles_sha256=sha(profiles_path),
        shortlist_sha256=sha(output),candidates=len(rows),source_count=300,development_read=False,
        rule='family top3 + strongest baseline + frozen distinct candidate + matched controls; complete same(N,m,K,j) G2 pools for every retained G2 profile',
        final_claim='Full1000 optimum over this frozen shortlist only; no full-grid1000 optimum claim')
    seal(str(output)+'.receipt.json',receipt);return receipt

def ready_actual(selection):
    cells=[c for c in selection['cells'] if c['N']==1024 and c['snr_db'] in (4,7,10,13)]
    return len(cells)==4 and all(c.get('ready_for_actual_link') for c in cells)

def combined_cost_rule(q_hours,coarse_remaining_hours,refine_hours,actual_hours,report_hours=1.):
    values=[q_hours,coarse_remaining_hours,refine_hours,actual_hours,report_hours]
    require(all(math.isfinite(v) and v>=0 for v in values),'Invalid measured/provisional cost')
    # Quality and coarse lookup overlap; refinement depends on both.
    total=max(q_hours,coarse_remaining_hours)+refine_hours+actual_hours+report_hours
    return dict(estimated_total_hours=total,threshold_hours=48.,screen300=total>48.,
        default_quality_sources=300 if total>48. else 1000,
        accounting='max(Q, remaining coarse CPU) + finalist refinement + actual-link + reporting',
        Q_hours=q_hours,coarse_remaining_hours=coarse_remaining_hours,refine_hours=refine_hours,
        actual_hours=actual_hours,report_hours=report_hours,development_read=False)
