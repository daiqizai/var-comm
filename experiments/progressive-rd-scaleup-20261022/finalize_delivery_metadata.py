"""Close report/figure metadata; no model, grid, or statistical computation."""
from pathlib import Path
import sys, json, hashlib, os, time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/progressive-rd-scaleup-20261022'
sys.path.insert(0,str(E))
from rdscale_common import L,O,read,pin,checked,save,process

def gone(v):
    now=process(v['pid'])
    return now is None or now['start_ticks']!=v['start_ticks']

closure=read(O/'delivery_v1/closure_audit.json')
assert closure['status']=='PASS' and len(closure['jobs'])==6
waits=[]
for job in closure['jobs']:
    w=read(checked(job['wait']))
    assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process'])
    assert not (L/'training_v1'/job['job']/'failure.json').exists()
    waits.append(job['wait'])
for key,expected in [('original_outer_actual_wait',1),('recovery_outer_actual_wait',0)]:
    d=closure['execution_lineage'][key];w=read(checked(d))
    assert type(w['returncode'])is int and w['returncode']==expected
    for k in ['process','parent_process']:
        if k in w:assert gone(w[k])
    waits.append(d)
for stage,rc in [('summary_dispatch_v1',0),('report_dispatch_v1',0),('figure_dispatch_v1',1),('figure_dispatch_v2',0),('figure_dispatch_v3',0)]:
    w=read(L/stage/'actual_wait.json')
    assert type(w['returncode'])is int and w['returncode']==rc and w['actual_wait']
    assert not any(w.get(k,False)for k in ['timed_out','terminated','killed'])
    assert gone(w['process']) and gone(w['parent_process'])
    waits.append(pin(L/stage/'actual_wait.json'))
for d in read(O/'delivery_v1/manifest.json')['files']:checked(d)
for d in read(O/'report_v1/manifest.json')['files']:checked(d)
for d in read(O/'figures_v3/manifest.json')['files']:checked(d)
review=read(O/'figures_v3/visual_review.json')
assert review['status']=='PASS' and len(review['actual_opened_previews'])==10
for d in review['actual_opened_previews']:checked(d)

archive=O/'execution_receipts_v1';archive.mkdir(exist_ok=False)
roots=['summary_dispatch_v1','report_dispatch_v1','figure_dispatch_v1','figure_dispatch_v2','figure_dispatch_v3',
       'qualification_v1','qualification_resource_v2','registration_control_v1','registration_control_resource_v2',
       'recovery_qualification_v1','summary_qualification_v1','summary_lineage_qualification_v2',
       'initial_control_v1','outer_dispatch_v1','dispatch_v1','recovery_initial_control_v1',
       'recovery_outer_dispatch_v1','unlaunched_training_recovery_v1','validation_delivery_v1']
sources=[]
for name in roots:
    for p in sorted((L/name).rglob('*')):
        if not p.is_file() or 'mplconfig' in p.parts:continue
        if p.suffix not in ['.json','.log','.txt','.stdout','.stderr']:continue
        assert not p.is_symlink() and p.stat().st_size<2_000_000,p
        sources.append(p)
for job in closure['jobs']:
    sources.extend([checked(job['completion'])])
records=[]
for p in sorted(set(sources)):
    before=pin(p);b=p.read_bytes();assert pin(p)==before
    dest=archive/p.relative_to(L);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    after=pin(dest);assert before['sha256']==after['sha256'] and before['bytes']==after['bytes']
    records.append(dict(source=before,copy=after))
save(archive/'manifest.json',dict(files=records,original_bytes_unchanged=True,arrays_and_checkpoints_not_copied=True))
final=O/'final_delivery_v1';final.mkdir(exist_ok=False)
save(final/'completion_evidence.json',dict(status='SCIENCE_AND_PRODUCTS_CLOSED_READY_FOR_REPOSITORY_PUBLICATION',
    observed_unix=time.time(),registration=pin(O/'registration.json'),closure=pin(O/'delivery_v1/closure_audit.json'),
    actual_waits=waits,figures=pin(O/'figures_v3/manifest.json'),visual_review=pin(O/'figures_v3/visual_review.json'),
    report=pin(O/'report_v1/manifest.json'),interpretation=pin(O/'FINAL_REPORT_zh.md'),
    receipts=pin(archive/'manifest.json'),original_controller_exit0=False,scientific_children_exit0=6,
    final_plot_directory='figures_v3',retained_render_history=['figures_v1','figures_v2'],
    new_scientific_calls_in_this_metadata_step=0,repository_publication='PENDING_SEPARATE_ACTUAL_CHECKS_COMMIT_PUSH',
    backup_deferred=True))
print(json.dumps(dict(status='DELIVERY_READY',receipt_files=len(records),completion=pin(final/'completion_evidence.json'))))
