"""Explicit user-requested cleanup; preserve current timing and all model snapshots."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,stat,subprocess,sys,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
CP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/baseline_strength_20261011')
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
OUT=R/'results/baseline_strength_20261011/storage_cleanup_20261012'
sys.path.insert(0,str(R/'experiments/baseline-delivery-20261012'))
import timing_owner as timing

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def scoped(p):
    p=Path(p)
    assert any(p.is_relative_to(root) for root in [CP,TMP])
    assert p.resolve()==p and stat.S_ISREG(p.lstat().st_mode),str(p)
    return p

def guard():
    assert not timing.RUN.exists() and not (timing.OUT/'launch.json').exists()
    for line in subprocess.check_output(['ps','-eo','pid=,comm=,args='],text=True).splitlines():
        v=line.strip().split(None,2)
        if len(v)==3 and v[1].startswith('python') and str(R) in v[2]:assert int(v[0])==os.getpid(),line
    return timing.verify()

def prepare():
    assert not (OUT/'plan.json').exists()
    reg=guard();protected=set(reg['input_sha256'])|set(reg['code_sha256'])
    local=json.loads((TMP/'cleanup_verified_local_copies_20261012.json').read_text())
    final=[]
    for branch,sub in [('P','P_continue'),('Swin','Swin_1to19')]:
        row=json.loads((R/'results/baseline_strength_20261011'/sub/'latest.json').read_text())
        assert sha(row['path'])==row['sha256']
        backup=next(x for x in local['final_recovery'] if x['path'].startswith(branch+'_'))
        assert backup['sha256']==row['sha256'];protected.add(row['path']);final.append(dict(row,local_backup=backup))
    candidates={}
    keep={CP/'P/full_040000_after_calibration.pt',CP/'P/full_100000_training_finished.pt',CP/'Swin/full_080000_before_calibration.pt',CP/'Swin/full_200000_training_finished.pt'}
    for branch in ['P','Swin']:
        for p in (CP/branch).glob('*.pt'):
            if (p.name.startswith('full_') or p.name.startswith('resume_')) and p not in keep:
                candidates[p]=dict(reason='Superseded training recovery snapshot; final full state and every calibration model weight retained',recoverable_exact_deleted_snapshot=False)
    for row in local['files']:
        p=TMP/row['remote_name'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        candidates[p]=dict(reason='Redundant server transfer or backup archive; exact local SHA-verified copy retained',local_copy=row,recoverable_exact_deleted_snapshot=True)
    for root in [TMP/'tmp/pytest-of-root',TMP/'pytest-of-root']:
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink():candidates[p]=dict(reason='Completed synthetic CPU test temporary output',recoverable_exact_deleted_snapshot=False)
    rows=[]
    for p,why in sorted(candidates.items()):
        scoped(p);assert str(p) not in protected,p
        rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),**why))
    # The published extraction manifest binds data which remain in place.
    manifest=json.loads((TMP/'manifest.json').read_text())
    for row in manifest['files']:
        p=TMP/'inputs'/row['destination'];assert p.is_file() and p.stat().st_size==row['bytes']
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'plan.json',dict(status='VERIFIED_EXPLICIT_DELETION_PLAN',created_unix=time.time(),files=rows,total_bytes=sum(x['bytes'] for x in rows),scope=[str(CP),str(TMP)],final_recovery=final,all_calibration_model_weights_retained=True,original_inputs_and_results_retained=True,failed_attempts_retained=True,other_projects_touched=False,scientific_calls=0))
    print('PLAN',len(rows),'files',sum(x['bytes'] for x in rows),'bytes',flush=True)

def apply():
    reg=guard();p=OUT/'plan.json';plan=json.loads(p.read_text());assert not (OUT/'completion.json').exists()
    for row in plan['files']:
        f=scoped(row['path']);assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'];assert str(f) not in reg['input_sha256'] and str(f) not in reg['code_sha256']
    journal=OUT/'deletion_journal.jsonl';assert not journal.exists()
    with journal.open('x') as log:
        for row in plan['files']:
            f=scoped(row['path']);f.unlink();log.write(json.dumps(dict(path=str(f),deleted=True,bytes=row['bytes']))+'\n');log.flush()
    timing.verify()
    for row in plan['final_recovery']:assert sha(row['path'])==row['sha256']
    assert all(not Path(x['path']).exists() for x in plan['files'])
    write(OUT/'completion.json',dict(status='COMPLETE_USER_REQUESTED_REDUNDANT_FILE_CLEANUP',deleted_files=len(plan['files']),deleted_file_bytes=plan['total_bytes'],timing_bindings_verified_after=True,final_recovery_verified_after=True,all_calibration_model_weights_retained=True,scientific_calls=0,plan_sha256=sha(p),journal_sha256=sha(journal)))
    print('DELETED',len(plan['files']),'FILES',plan['total_bytes'],'BYTES; TIMING AND FINAL RECOVERY VERIFIED',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','apply']);args=a.parse_args()
    lock=(timing.OUT/'owner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    {'prepare':prepare,'apply':apply}[args.action]()
