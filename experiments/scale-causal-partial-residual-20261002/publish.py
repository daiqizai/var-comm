"""Checked publication, preserving source/result history and large CSV bytes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
RESULT=ROOT/'results/scale_causal_partial_residual_20261002'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8_000_000),b''):h.update(block)
    return h.hexdigest()
def write(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)
def command(args,log,cpu=False):
    env=os.environ.copy()
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    result=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    log.flush()
    if result.returncode:raise RuntimeError('Check failed: '+' '.join(map(str,args)))

def source_bindings():
    return {str(p):sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.json','.md']}

def verify_analysis(receipt):
    if receipt.get('status')!='COMPLETE' or receipt.get('synthetic',True) or receipt.get('training_updates')!=0:
        raise RuntimeError('Actual completed zero-training analysis required')
    for category in ['inputs','outputs']:
        entries=receipt.get(category)
        if not isinstance(entries,dict) or not entries:raise RuntimeError('Analysis '+category+' bindings missing')
        for path,h in entries.items():
            if not Path(path).is_file() or sha(path)!=h:raise RuntimeError('Analysis '+category+' changed: '+path)

def export_cells(stage):
    folder=OUT/stage/'cells'
    cells=sorted(folder.glob('*.json'))
    expected_sources={'m1_screen':200,'m1_calibration':1000}[stage]
    if [p.name for p in cells]!=[f'{i:04d}.json' for i in range(expected_sources)]:
        raise RuntimeError('Calibration archive source cells missing/extra: '+stage)
    done=read(OUT/(stage+'_complete.json'));regpath=OUT/(stage+'_registration.json');reg=read(regpath)
    identity=sha(regpath)
    if done.get('status')!='COMPLETE' or done.get('sources')!=expected_sources or done.get('training_updates')!=0 or done.get('development_read') is not False:
        raise RuntimeError('Invalid calibration archive completion: '+stage)
    if len(reg.get('source_ids',[]))<expected_sources or len(reg.get('preprocessing_ids',[]))<expected_sources:
        raise RuntimeError('Calibration source registration incomplete')
    if reg.get('calibration_or_development')!=stage or reg.get('training_updates')!=0:
        raise RuntimeError('Calibration source role changed')
    for path,h in reg.get('input_artifacts',{}).items():
        if sha(path)!=h:raise RuntimeError('Calibration input artifact changed: '+path)
    dest=RESULT/('m1_shortlist.json' if stage=='m1_screen' else 'm1_policy.json')
    if done.get('policy_sha256')!=sha(dest) or read(dest).get('registration_sha256')!=identity:
        raise RuntimeError('Calibration archive policy binding differs')
    expected_rows=int(done['rows'])
    if expected_rows%expected_sources or (stage=='m1_screen' and expected_rows!=185000):
        raise RuntimeError('Calibration archive row count differs')
    p=RESULT/(stage+'_per_frame.csv');tmp=p.with_suffix('.tmp')
    first=read(cells[0]);fields=list(first['rows'][0]);n=0;grid=None
    try:
        with tmp.open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
            for i,cell in enumerate(cells):
                obj=read(cell);rows=obj['rows']
                if obj.get('identity')!=identity:raise RuntimeError('Unbound calibration cell')
                if len(rows)!=expected_rows//expected_sources:raise RuntimeError('Incomplete per-source calibration grid')
                keys=[]
                for row in rows:
                    if set(row)!=set(fields):raise RuntimeError('Calibration row schema changed')
                    if row.get('source_index')!=i or row.get('source_id')!=reg['source_ids'][i] or row.get('preprocessing_id')!=reg['preprocessing_ids'][i]:
                        raise RuntimeError('Calibration row bound to a different source')
                    keys.append((row['N'],row['phy_family'],row['snr_db'],row['noise_seed'],row['action_id']))
                if len(set(keys))!=len(keys):raise RuntimeError('Duplicate calibration observation/action')
                if grid is None:grid=set(keys)
                elif set(keys)!=grid:raise RuntimeError('Calibration candidate grid differs between sources')
                w.writerows(rows);n+=len(rows)
        if n!=expected_rows:raise RuntimeError('Calibration archive total rows differ')
        os.replace(tmp,p)
    finally:tmp.unlink(missing_ok=True)
    write(RESULT/'provenance'/(stage+'_archive.json'),dict(cells=len(cells),rows=n,identity=identity,sha256=sha(p)))

def archive_tables(scope):
    # Independent stage manifests allow M1 to be published before M2 exists.
    table_dir=RESULT/(scope+'_table_shards');table_dir.mkdir(parents=True,exist_ok=True)
    manifest={}
    for p in sorted(RESULT.rglob('*.csv')):
        if not p.name.startswith(scope+'_') or p.stat().st_size<=8_000_000 or any('table_shards' in s for s in p.parts):continue
        rel=p.relative_to(RESULT).as_posix();parts=[];total=0
        with p.open('rb') as h:
            header=h.readline();buf=bytearray(header);n=0;index=0
            def flush(block,count,idx):
                name=rel.replace('/','_').replace('\\','_')+f'.part{idx:04d}.csv'
                part=table_dir/name
                if part.exists() and part.read_bytes()!=block:raise RuntimeError('Archived table changed')
                if not part.exists():part.write_bytes(block)
                return dict(path=part.relative_to(RESULT).as_posix(),rows=count,sha256=sha(part),bytes=len(block))
            for line in h:
                if len(buf)+len(line)>8_000_000 and n:
                    parts.append(flush(bytes(buf),n,index));total+=n;index+=1;buf=bytearray(header);n=0
                buf.extend(line);n+=1
            if n:parts.append(flush(bytes(buf),n,index));total+=n
        digest=hashlib.sha256();digest.update(header)
        for part in parts:
            with (RESULT/part['path']).open('rb') as h:
                if h.readline()!=header:raise RuntimeError('Archive header changed')
                for block in iter(lambda:h.read(8_000_000),b''):digest.update(block)
        if digest.hexdigest()!=sha(p):raise RuntimeError('Exact table restoration failed')
        index_path=table_dir/(rel.replace('/','_')+'.index.md')
        manifest[rel]=dict(sha256=sha(p),rows=total,parts=parts,index=index_path.relative_to(RESULT).as_posix())
        lines=[f'# {rel}', '',f'{total} data rows. Complete original SHA-256: `{sha(p)}`.', '',
            'Each part repeats the same CSV header. Restore the original bytes by keeping the first header and appending each ordered part without its header; then check the complete SHA-256.', '',
            '| Part | Rows | SHA-256 |','|---|---:|---|']
        for part in parts:
            name=Path(part['path']).name
            lines.append(f"| [{name}]({name}) | {part['rows']} | `{part['sha256']}` |")
        index_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(table_dir/'manifest.json',dict(run=OUT.name,scope=scope,exact_bytes=True,tables=manifest))
    (table_dir/'README.md').write_text('# Complete original CSV archives\n\n'+
        '\n'.join(f"- [{name}]({Path(rec['index']).name}): {rec['rows']} rows" for name,rec in manifest.items())+'\n',encoding='utf-8')
    ignored=['/'+(RESULT/name).relative_to(ROOT).as_posix() for name in manifest]
    p=ROOT/'.gitignore';lines=p.read_text().splitlines()
    p.write_text('\n'.join(lines+[line for line in ignored if line not in lines])+'\n')
    return manifest

def published_report(report,target,tables):
    """Resolve all relative links, routing ignored CSVs to tracked shard indexes."""
    def relocate(match):
        raw=match[1]
        if '://' in raw or raw.startswith('#') or re.match(r'^[A-Za-z]+:',raw):return match[0]
        value=raw[1:-1] if raw.startswith('<') and raw.endswith('>') else raw
        path,separator,fragment=value.partition('#')
        resolved=(report.parent/path).resolve()
        try:entry=tables.get(resolved.relative_to(RESULT.resolve()).as_posix())
        except ValueError:entry=None
        if entry:resolved=(RESULT/entry['index']).resolve();separator=fragment=''
        moved=Path(os.path.relpath(resolved,target.parent)).as_posix()
        if separator:moved+='#'+fragment
        if ' ' in moved:moved='<'+moved+'>'
        return ']('+moved+')'
    content=re.sub(r'\]\(([^)]+)\)',relocate,report.read_text(encoding='utf-8'))
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content,encoding='utf-8')
    return content

def prepare(scope):
    if scope=='m1':
        for name in ['m1_development','m1_rate_curve','m1_timing']:
            if read(OUT/(name+'_complete.json')).get('status')!='COMPLETE':raise RuntimeError(name+' incomplete')
        receipt=read(RESULT/'m1_analysis_completion.json')
        verify_analysis(receipt)
        for stage in ['m1_screen','m1_calibration']:export_cells(stage)
    else:
        for name in ['m2_calibration','m2_evaluation','m2_actual','m2_timing']:
            if not (OUT/(name+'_complete.json')).exists():raise RuntimeError(name+' receipt missing')
            rec=read(OUT/(name+'_complete.json'))
            if rec.get('status')!=name.upper()+'_COMPLETE' or rec.get('synthetic',True) or rec.get('training_updates')!=0:raise RuntimeError(name+' invalid completion')
        receipt=read(RESULT/'m2_analysis_completion.json')
        verify_analysis(receipt)
    prov=RESULT/'provenance';prov.mkdir(parents=True,exist_ok=True)
    for p in OUT.glob('*.json'):
        if 'status' in p.name or 'failure' in p.name:continue
        shutil.copyfile(p,prov/p.name)
    archive_tables(scope)
    tables={}
    for p in RESULT.glob('*_table_shards/manifest.json'):tables.update(read(p)['tables'])
    report=RESULT/(scope+'_report.md')
    target=ROOT/'reports'/f'scale_causal_{scope}_20261002.md'
    companion=RESULT/(scope+'_published_report.md')
    published_report(report,companion,tables);published_report(report,target,tables)
    write(prov/(scope+'_report_publication.json'),dict(analysis_report_sha256=sha(report),
        published_report=str(companion),published_report_sha256=sha(companion),
        repository_report=str(target),repository_report_sha256=sha(target),
        transform='Relative links relocated; archived CSVs point to tracked part indexes; analysis report bytes preserved'))
    text=f'## {scope.upper()} scale-causal study delivered 2026-10-02\n\nZero neural training updates. Frozen calibration choices, original100 development sources and failure-inclusive metrics. See [complete report](reports/{target.name}); historical models and queues are preserved. '+('Method2 proceeds only after the verified method1 publication.' if scope=='m1' else 'The two-method authorized pipeline stops after this verified publication; actual-link branch status is explicit in the report.')+'\n\n'
    for name in ['RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md']:
        p=ROOT/name;old=p.read_text()
        if text not in old:p.write_text(text+old)
    readme=RESULT/'README.md'
    readme.write_text('# Scale-causal partial transmission and residual guidance\n\n'+
        '\n'.join(f'- [{p.stem}]({p.name})' for p in sorted(RESULT.glob('*_published_report.md')))+
        '\n\nAll original large CSV bytes are preserved by ordered stage-specific `*_table_shards/manifest.json` records. Restore by keeping the first header and concatenating each ordered part without repeated headers; validate the complete SHA-256. Code, policies, paired intervals, and independent timing distinguish engineering checks from real images.\n')
    write(RESULT/'provenance'/(scope+'_published_files.json'),{p.relative_to(RESULT).as_posix():sha(p) for p in RESULT.rglob('*') if p.is_file() and p.stat().st_size<=10_000_000 and p.name!=(scope+'_published_files.json')})
    return target

def mark_complete(scope,record):
    if scope=='m1':write(OUT/'m1_complete.json',dict(status='M1_COMPLETE',publication=record,training_updates=0,synthetic=False))
    elif scope=='m2':write(OUT/'completion.json',dict(status='AUTHORIZED_TWO_METHODS_COMPLETE',publication=record,training_updates=0,synthetic=False,stop=True))

def verify_previous_publication(record):
    if record.get('checks')!='PASS' or record.get('commit')!=record.get('remote_commit'):
        raise RuntimeError('Previous publication is not a verified push')
    for args in [['git','fetch','origin'],['git','cat-file','-e',record['commit']+'^{commit}'],
                 ['git','merge-base','--is-ancestor',record['commit'],'origin/main']]:
        result=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if result.returncode:raise RuntimeError('Previous publication commit is not verified in remote main ancestry')

def main(scope):
    OUT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
    receipt_path=OUT/(('source' if scope=='source' else scope)+'_publication.json')
    bindings=source_bindings()
    if receipt_path.exists():
        j=read(receipt_path)
        if j.get('status')=='PUSHED' and j.get('source_bindings')==bindings:
            verify_previous_publication(j)
            mark_complete(scope,j)
            return
    paths=[str(HERE.relative_to(ROOT))]
    if scope!='source':
        target=prepare(scope);paths+=[str(RESULT.relative_to(ROOT)),str(target.relative_to(ROOT)),'.gitignore','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md']
    with (OUT/(scope+'_publication_checks.log')).open('a') as log:
        command(['git','add','--',*paths],log)
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
        if any(p!='release_manifest.json' and not any(p==s or p.startswith(s+'/') for s in paths) for p in staged):raise RuntimeError('Unrelated staged changes present')
        command([sys.executable,'tools/update_repository_manifest.py'],log)
        command(['git','add','release_manifest.json'],log)
        command([sys.executable,'tools/verify_repository.py'],log)
        command([sys.executable,'tools/run_cpu_checks.py'],log)
        for p in sorted(HERE.glob('test_*.py')):command([sys.executable,str(p)],log,cpu=True)
        command(['git','fetch','origin'],log)
        command(['git','merge-base','--is-ancestor','origin/main','HEAD'],log)
        for path,h in bindings.items():
            data=subprocess.check_output(['git','show',':'+str(Path(path).relative_to(ROOT))],cwd=ROOT)
            if hashlib.sha256(data).hexdigest()!=h:raise RuntimeError('Staged execution source differs')
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip():
            msg=OUT/(scope+'_commit_message.txt');msg.write_text(('Register ordered scale-causal partial and residual experiments' if scope=='source' else f'Publish {scope} scale-causal experiment and paired results')+'\n')
            command(['git','commit','-F',str(msg)],log)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write(receipt_path,dict(status='COMMITTED',commit=commit,checks='PASS',source_bindings=bindings,time=time.time()))
        command(['git','push','origin','main'],log)
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
        if remote!=commit:raise RuntimeError('Remote commit mismatch')
    record=dict(status='PUSHED',commit=commit,remote_commit=remote,checks='PASS',source_bindings=bindings,time=time.time())
    write(receipt_path,record)
    mark_complete(scope,record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['source','m1','m2'],required=True);a=p.parse_args()
    try:main(a.scope)
    except Exception as e:
        import traceback
        write(OUT/('publication_failure_'+str(time.time_ns())+'.json'),dict(error=str(e),traceback=traceback.format_exc()));raise
