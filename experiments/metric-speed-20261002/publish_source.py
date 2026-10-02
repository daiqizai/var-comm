"""Publish the qualified runtime extension after original M1 delivery."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'outputs/METRIC-SPEED-20261002'
PARENT=ROOT/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
RESULT=ROOT/'results/metric_speed_20261002'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def write(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(t,p)
def sources():return {str(p):sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.md')}
def verify(items):
    if not isinstance(items,dict) or not items:raise RuntimeError('Required source/evidence bindings missing')
    for p,h in items.items():
        if sha(p)!=h:raise RuntimeError('Frozen runtime input changed: '+p)
def command(args,log=None,capture=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    p=subprocess.run([str(x) for x in args],cwd=ROOT,env=env,stdout=subprocess.PIPE if capture else log,stderr=subprocess.PIPE if capture else subprocess.STDOUT)
    if log:log.flush()
    if p.returncode:raise RuntimeError('Publication command failed: '+repr(args))
    return p.stdout if capture else b''
def git(*args):return command(['git',*args],capture=True).decode().strip()
def names(*args):return set(filter(None,command(['git',*args,'-z'],capture=True).decode().split('\0')))

def verify_pending_manifest(pending,bound,qualification_sha,head,manifest_sha,staged_sha):
    if (pending.get('source_bindings')!=bound or pending.get('qualification_sha256')!=qualification_sha
            or pending.get('base_commit')!=head or pending.get('manifest_sha256')!=manifest_sha
            or pending.get('manifest_sha256')!=staged_sha):
        raise RuntimeError('A dirty manifest is not this unchanged pending publication')
    verify(pending['published_files'])

def gate(qualification,source_identity):
    q=qualification
    if (q.get('status'),q.get('selected_implementation')) not in [('QUALIFIED','accelerated'),('USE_ORIGINAL','original')]:
        raise RuntimeError('A completed explicit runtime decision is required')
    if q.get('source_bindings')!=source_identity:raise RuntimeError('Qualified speed source changed')
    verify(source_identity);verify(q['base_source_bindings'])
    if q.get('development_read') is not False or q.get('training_updates')!=0:
        raise RuntimeError('Only zero-training calibration engineering qualification is authorized')
    if q.get('status')=='QUALIFIED':
        speedup=float(q.get('speedup',0))
        if q.get('strict_equality_passed') is not True or not math.isfinite(speedup) or speedup<1.10:
            raise RuntimeError('Accelerated execution requires strict equality and measured improvement')

def push(rec,path,log):
    command(['git','push','origin','main'],log)
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if remote!=rec['commit']:raise RuntimeError('Remote SHA differs after normal push')
    rec.update(status='PUSHED',remote_commit=remote,time=time.time());write(path,rec)
    return rec

def main():
    OUT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
    parent=read(PARENT/'m1_complete.json');pub=parent.get('publication',{})
    if parent.get('status')!='M1_COMPLETE' or pub.get('status')!='PUSHED' or pub.get('commit')!=pub.get('remote_commit'):
        raise RuntimeError('Original M1 must be completely published before any new source is staged')
    verify(pub['source_bindings'])
    qpath=OUT/'m2_speed_qualification.json';q=read(qpath);bound=sources();gate(q,bound)
    if git('branch','--show-current')!='main' or Path(git('rev-parse','--show-toplevel')).resolve()!=ROOT:
        raise RuntimeError('The sole original main worktree is required')
    if git('remote','get-url','origin') not in ('git@github.com:daiqizai/var-comm.git','https://github.com/daiqizai/var-comm.git'):
        raise RuntimeError('Unexpected Git repository')
    path=OUT/'source_publication.json';previous=read(path) if path.exists() else None
    with (OUT/'source_publication.log').open('a') as log:
        if previous:
            if previous.get('source_bindings')!=bound or previous.get('checks')!='PASS':raise RuntimeError('Prior publication differs')
            verify(previous['published_files'])
            command(['git','fetch','origin'],log)
            if previous['status']=='PUSHED':
                command(['git','merge-base','--is-ancestor',previous['commit'],'origin/main'],log);return previous
            if previous['status']=='COMMITTED':
                if git('rev-parse','HEAD')!=previous['commit'] or names('diff','--name-only') or names('diff','--cached','--name-only'):
                    raise RuntimeError('Pending commit changed before a retry')
                command(['git','merge-base','--is-ancestor','origin/main','HEAD'],log)
                return push(previous,path,log)
            raise RuntimeError('Unknown prior publication status')
        prefixes=(HERE.relative_to(ROOT).as_posix()+'/',RESULT.relative_to(ROOT).as_posix()+'/')
        dirty=names('diff','--name-only')|names('diff','--cached','--name-only')
        pending_path=OUT/'publication_checks_pending.json'
        if 'release_manifest.json' in dirty:
            if not pending_path.exists():raise RuntimeError('Unrelated dirty manifest must be preserved')
            verify_pending_manifest(read(pending_path),bound,sha(qpath),git('rev-parse','HEAD'),
                sha(ROOT/'release_manifest.json'),hashlib.sha256(command(['git','show',':release_manifest.json'],capture=True)).hexdigest())
        if any(n!='release_manifest.json' and not any(n.startswith(prefix) for prefix in prefixes) for n in dirty):
            raise RuntimeError('Unrelated tracked edits must be preserved')
        shutil.copyfile(qpath,RESULT/'qualification.json')
        for name in ('handoff.json','controller_registration.json'):
            if (OUT/name).exists():shutil.copyfile(OUT/name,RESULT/name)
        (RESULT/'README.md').write_text('# Runtime acceleration qualification\n\n'
            'This is a calibration-only engineering measurement, not an image-quality experiment. '
            'The original data, model weights, loss, policy grids and noise seeds are unchanged. '
            'The receiver uses the candidate only when token sequences, latent tensors and diagnostics match exactly and aggregate paired runtime improves by at least 10%. '
            'Otherwise the original receiver is selected. Full scientific results remain in the original two-method result directory.\n')
        files=[Path(p) for p in bound]+[p for p in RESULT.iterdir() if p.is_file()]
        allowed={p.relative_to(ROOT).as_posix() for p in files}|{'release_manifest.json'}
        if names('diff','--cached','--name-only')-allowed:raise RuntimeError('Unrelated staged files present')
        published={str(p):sha(p) for p in files}
        command(['git','add','--',*[str(p.relative_to(ROOT)) for p in files]],log)
        command([sys.executable,'tools/update_repository_manifest.py'],log)
        command(['git','add','--','release_manifest.json'],log)
        pending=dict(source_bindings=bound,qualification_sha256=sha(qpath),base_commit=git('rev-parse','HEAD'),
            manifest_sha256=sha(ROOT/'release_manifest.json'),published_files=published)
        write(pending_path,pending)
        command([sys.executable,'tools/verify_repository.py'],log)
        command([sys.executable,'tools/run_cpu_checks.py'],log)
        for test in sorted(HERE.glob('test_*.py')):command([sys.executable,str(test)],log)
        gate(read(qpath),bound);verify(published)
        verify_pending_manifest(pending,bound,sha(qpath),git('rev-parse','HEAD'),sha(ROOT/'release_manifest.json'),
            hashlib.sha256(command(['git','show',':release_manifest.json'],capture=True)).hexdigest())
        if names('diff','--cached','--name-only')-allowed or names('diff','--name-only'):
            raise RuntimeError('Publication scope changed during checks')
        for p,h in published.items():
            if hashlib.sha256(command(['git','show',':'+Path(p).relative_to(ROOT).as_posix()],capture=True)).hexdigest()!=h:
                raise RuntimeError('Staged bytes differ from checked inputs')
        command(['git','fetch','origin'],log)
        command(['git','merge-base','--is-ancestor','origin/main','HEAD'],log)
        command(['git','merge-base','--is-ancestor',pub['commit'],'origin/main'],log)
        if names('diff','--cached','--name-only'):
            message=OUT/'commit_message.txt';message.write_text('Qualify conditional receiver runtime acceleration without changing experiment definitions\n')
            command(['git','commit','-F',str(message)],log)
        record=dict(status='COMMITTED',commit=git('rev-parse','HEAD'),checks='PASS',source_bindings=bound,
            qualification_sha256=sha(qpath),published_files=published,m1_commit=pub['commit'],time=time.time())
        write(path,record);return push(record,path,log)

if __name__=='__main__':main()
