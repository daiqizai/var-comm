"""Checked normal publication of the complete matched five-method comparison.

Only new dedicated artifacts are staged. Existing scientific bytes, including
large raw tables, are retained. An interrupted transaction requires explicit
review before the delivery controller can be resumed.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import full_comparison_common as c
from full_comparison_delivery import upstream_gate,external_delivery_gate

LIMIT=8_000_000


def bind(target,values):
    c.verify(values)
    for path,digest in values.items():
        c.require(path not in target or target[path]==digest,'Conflicting publication input: '+path)
        target[path]=digest


def copy_exact(source,target,root):
    source,target=Path(source),Path(target)
    c.require(not source.is_symlink() and not target.is_symlink() and Path(root).resolve() in target.resolve().parents,
        'Invalid artifact publication path')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():c.require(c.sha(source)==c.sha(target),'Existing published artifact differs: '+str(target))
    else:shutil.copyfile(source,target)
    return target


def put_text(path,text):
    path=Path(path);data=text.encode('utf-8')
    if path.exists():c.require(path.read_bytes()==data,'Existing publication view differs: '+str(path))
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return path


def archive_bytes(path,limit=LIMIT):
    """Lossless base64 parts for a large JSON proof; never edit the original."""
    path=Path(path);folder=path.parent/'byte_shards';folder.mkdir(exist_ok=True)
    raw_size=(limit//4)*3-3;c.require(raw_size>0,'Invalid archive bound')
    parts=[];digest=hashlib.sha256();size=0
    with path.open('rb') as source:
        while data:=source.read(raw_size):
            digest.update(data);size+=len(data);encoded=base64.b64encode(data)+b'\n'
            part=folder/(path.name+'.part'+str(len(parts)).zfill(4)+'.b64.txt')
            if part.exists():c.require(part.read_bytes()==encoded,'Archive part changed')
            else:part.write_bytes(encoded)
            parts.append(dict(path=part.relative_to(path.parent).as_posix(),sha256=c.sha(part),raw_bytes=len(data)))
    record=dict(format='exact-byte-base64-v1',original=path.name,sha256=digest.hexdigest(),bytes=size,parts=parts)
    manifest=folder/(path.name+'.manifest.json');c.seal(manifest,record)
    verify_byte_archive(path.parent,record)
    note=folder/(path.name+'.index.md')
    put_text(note,'# Lossless archive of '+path.name+'\n\nThe original file is unchanged. Decode each base64 part in manifest order and concatenate the decoded bytes. Verify the resulting SHA256 `'+record['sha256']+'`.\n\n[Manifest]('+manifest.name+')\n')
    return record,[manifest,note,*[path.parent/p['path'] for p in parts]],note


def verify_byte_archive(folder,record):
    c.require(record.get('format')=='exact-byte-base64-v1','Unknown byte archive format')
    digest=hashlib.sha256();size=0
    for part in record['parts']:
        path=Path(folder)/part['path'];c.require(c.sha(path)==part['sha256'],'Archive part hash differs')
        raw=base64.b64decode(path.read_bytes().strip(),validate=True)
        c.require(len(raw)==part['raw_bytes'],'Archive decoded length differs');digest.update(raw);size+=len(raw)
    c.require(size==record['bytes'] and digest.hexdigest()==record['sha256'],'Archive does not restore exact original bytes')


def archive_helper(config):
    path=Path(config['root'])/'experiments/unified-metrics-20261002/publish.py'
    c.require(config['bindings'].get(str(path))==c.sha(path),'Unregistered CSV archive helper')
    spec=importlib.util.spec_from_file_location('_full_comparison_csv_archive',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def archive_dataset(folder,files,helper):
    manifest=helper.archive_tables(folder,limit=LIMIT);links={}
    for relative,entry in manifest['tables'].items():
        original=folder/relative;c.require(c.sha(original)==entry['sha256'],'Original CSV changed while archiving')
        helper.verify_archive_entry(folder,entry);links[relative]=entry['index']
        if original in files:files.remove(original)
    files.update(p for p in (folder/'table_shards').iterdir() if p.is_file())
    for path in sorted(files.copy()):
        if path.parent!=folder or path.stat().st_size<=LIMIT:continue
        c.require(path.suffix in ('.json','.md','.txt','.svg','.pdf','.png'),'Unsupported large artifact: '+str(path))
        _,parts,index=archive_bytes(path);files.remove(path);files.update(parts);links[path.name]=index.relative_to(folder).as_posix()
    return links


def prepare(config,config_path):
    c.validate_config(config);p=c.locations(config['root']);root=p['root'];out=p['out'];folder=p['delivery']
    inputs={str(config_path):c.sha(config_path),**config['bindings']};files=set();proofs={}
    for stage in c.STAGES[:-1]:
        from full_comparison_delivery import failure_paths
        c.require(not any(path.exists() for path in failure_paths(config,stage)),'Stage requires review: '+stage)
        proofs[stage]=c.stage_proof(config,stage)
        for path,proof in proofs[stage].items():
            bind(inputs,{path:proof['sha256']});bind(inputs,c.receipt_bindings(c.read(path)))
    bind(inputs,c.final_table_gate(config))
    for name,gate in (('upstream_admission.json',upstream_gate),('earlier_delivery_admission.json',external_delivery_gate)):
        value=gate(config);saved=c.read(folder/name)
        c.require(saved==value,'Process/completion admission changed before publication')
        bind(inputs,value['bindings']);bind(inputs,{str(folder/name):c.sha(folder/name)})
    def add(source,target):
        source=Path(source);bind(inputs,{str(source):c.sha(source)})
        files.add(copy_exact(source,target,root))
    runtime=Path(config['runtime']);experiment=root/'experiments/external-comparison-20261004'
    # Only our new continuation and own-method sources. Previous external source
    # bytes are already published; any same-name conflict stops for review.
    for source in sorted(runtime.iterdir()):
        if source.name.startswith(('own_controls','full_comparison','publish_full_comparison','test_full_comparison')) and source.suffix in ('.py','.md','.json'):
            c.require(config['bindings'].get(str(source))==c.sha(source),'Unregistered continuation source')
            add(source,experiment/source.name)
    provenance=p['result']/'full_provenance'
    add(config_path,provenance/'config.json')
    for name in ('upstream_admission.json','earlier_delivery_admission.json'):add(folder/name,provenance/name)
    all_receipts={Path(path) for stage in proofs.values() for path in stage}
    for source in sorted(all_receipts):
        name=source.relative_to(out).as_posix().replace('/','__');add(source,provenance/name)
        receipt=c.read(source)
        for path in receipt.get('outputs',{}):
            artifact=Path(path)
            if p['result'] in artifact.parents:
                c.require(artifact.suffix in ('.csv','.json','.md','.svg','.png','.pdf','.txt'),'Unexpected scientific artifact')
                files.add(artifact)
            elif artifact.parent==p['cost'] and artifact.name in ('per_frame.csv','summary.csv'):
                add(artifact,p['result']/'receiver_cost'/artifact.name)
            elif artifact.suffix=='.json' and ('registration' in artifact.name or artifact.name in
                ('metric_batch_qualification.json','score_input_admission.json')):
                c.require(out in artifact.parents,'Registered proof lies outside this completed experiment')
                add(artifact,provenance/artifact.relative_to(out).as_posix().replace('/','__'))
    # Per-frame and summary provenance remains in its original dataset. No
    # tensor/image caches, weight files, or original source population is staged.
    datasets=sorted({path.parent for path in files if path.stat().st_size>LIMIT})
    helper=archive_helper(config);links={}
    for dataset in datasets:links[dataset]=archive_dataset(dataset,files,helper)
    final=p['result']/'final_comparison';original=final/'MATCHED_COMPARISON_REPORT.md'
    c.require(original in files,'Full scientific report is not a completed output')
    replacements={}
    for dataset,mapping in links.items():
        if dataset==final or final in dataset.parents:
            for name,index in mapping.items():
                replacements[(dataset/name).relative_to(final).as_posix()]=(dataset/index).relative_to(final).as_posix()
    text=original.read_text(encoding='utf-8')
    for name,index in replacements.items():text=text.replace(']('+name+')',']('+index+')')
    view=final/'PUBLISHED_REPORT.md'
    files.add(put_text(view,'> Publication view: large artifacts link to lossless archives. Original report, tables, and proofs are unchanged.\n\n'+text))
    report=root/'reports/full_external_comparison_20261004.md'
    files.add(put_text(report,'# 完整同预算五方法对照\n\n'
        'P、无类别数字 VAR、方法一、SwinJSCC 和冻结 HiFi-DiffCom 共 9,000 条配对结果，覆盖 N1024/N2048、1/7/13 dB、100 张源图和三个噪声重复。\n\n'
        '[完整质量、区间、固定样例与接收代价报告](../results/external_comparison_20261004/final_comparison/PUBLISHED_REPORT.md)\n\n'
        '全部 13 项指标、16 张固定源图的 480 个方法单元，以及相同四张校准源图的 120 帧接收代价均已通过登记门禁。具体结论及训练预算限制见完整报告。未访问 holdout；没有填补缺测点或改变历史科学输出。\n'))
    for artifact in files:c.require(not artifact.is_symlink() and artifact.stat().st_size<=LIMIT,'Staged artifact exceeds size/path contract: '+str(artifact))
    return sorted(files),inputs


def cpu_environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')


def run_command(args,root,log,capture=False):
    log.write('COMMAND '+repr(args)+'\n');log.flush()
    result=subprocess.run(args,cwd=root,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=cpu_environment())
    # Git blobs may be PNG/PDF: captured stdout must never enter a text log.
    if not capture:log.write(result.stdout.decode(errors='replace'))
    log.write(result.stderr.decode(errors='replace'));log.flush()
    c.require(result.returncode==0,'Publication command failed: '+repr(args));return result.stdout


def exact_blobs(bindings,revision,root,command):
    for path,digest in bindings.items():
        relative=Path(path).relative_to(root).as_posix()
        blob=command(['git','show',revision+':'+relative],True)
        c.require(hashlib.sha256(blob).hexdigest()==digest,'Committed artifact bytes differ: '+relative)


def verify_prior_publication(record,root,command):
    c.require(record.get('status')=='PUSHED' and record.get('checks')=='PASS'
        and record.get('commit')==record.get('remote_commit'),'Earlier normal publication is incomplete')
    exact_blobs(record['published_files'],record['commit'],root,command)
    command(['git','merge-base','--is-ancestor',record['commit'],'origin/main'])


def publish(config_path):
    import fcntl
    config_path=Path(config_path).resolve();config=c.validate_config(c.read(config_path));p=c.locations(config['root'])
    root=p['root'];out=p['delivery'];out.mkdir(parents=True,exist_ok=True)
    c.require(not (out/'publication_failure.json').exists(),'Prior publication failure requires review')
    with (out/'publication.lock').open('a+') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (root/'.git/m2-recovery-publication.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)  # Wait for the prior normal transaction.
            with (out/'publication.log').open('a',encoding='utf-8') as log:
                def command(args,capture=False):return run_command(args,root,log,capture)
                def git(*args):return command(['git',*args],True).decode().strip()
                c.require(git('rev-parse','--show-toplevel')==str(root) and git('branch','--show-current')=='main','Wrong worktree')
                c.require(git('remote','get-url','origin') in ('git@github.com:daiqizai/var-comm.git','https://github.com/daiqizai/var-comm.git'),'Wrong remote')
                receipt_path=out/'publication.json';pending_path=out/'publication_pending.json'
                previous=c.read(receipt_path) if receipt_path.exists() else None
                pending=c.read(pending_path) if pending_path.exists() else None
                command(['git','fetch','origin','main'])
                if previous:
                    c.require(previous.get('phase')=='full_results' and previous.get('checks')=='PASS'
                        and previous.get('full_plan_complete') is True and previous.get('status') in ('COMMITTED','PUSHED'), 'Unknown final publication')
                    c.verify(previous['inputs']);exact_blobs(previous['published_files'],previous['commit'],root,command)
                    if previous['status']=='PUSHED':
                        command(['git','merge-base','--is-ancestor',previous['commit'],'origin/main']);return previous
                    c.require(git('rev-parse','HEAD')==previous['commit'] and not git('diff','--cached','--name-only')
                        and not git('diff','--name-only'),'Unpushed final commit or worktree changed')
                    record=previous
                else:
                    # Old manifests may now differ on disk. Their original Git
                    # blobs at the proven public commit are authoritative.
                    verify_prior_publication(c.read(p['out']/'results_publication.json'),root,command)
                    if pending is None:
                        c.require(not git('diff','--cached','--name-only') and not git('diff','--name-only'),'Preserve tracked/staged edits')
                        base=git('rev-parse','HEAD');c.require(base==git('rev-parse','origin/main'),'Local/remote history needs review')
                        files,inputs=prepare(config,config_path)
                        pending=dict(phase='full_results',base_commit=base,inputs=inputs,published_files={str(path):c.sha(path) for path in files})
                        c.write(pending_path,pending)
                    c.require(pending.get('phase')=='full_results','Wrong pending transaction')
                    c.verify(pending['inputs']);c.verify(pending['published_files'])
                    checked=pending['published_files'];relative=[Path(path).relative_to(root).as_posix() for path in checked]
                    allowed=set(relative)|{'release_manifest.json'}
                    for args in (('diff','--cached','--name-only'),('diff','--name-only')):
                        c.require(set(git(*args).splitlines())<=allowed,'Preserve unrelated repository edits')
                    head=git('rev-parse','HEAD')
                    if head!=pending['base_commit']:
                        c.require(pending.get('checks')=='PASS' and pending.get('tree')==git('rev-parse','HEAD^{tree}')
                            and git('rev-parse','HEAD^')==pending['base_commit'],'Unexpected interrupted-commit HEAD')
                        exact_blobs(checked,head,root,command)
                    else:
                        c.require(git('rev-parse','origin/main')==pending['base_commit'],'Remote changed before final commit')
                        for start in range(0,len(relative),80):command(['git','add','--',*relative[start:start+80]])
                        command([config['report_python'],'-B','tools/update_repository_manifest.py'])
                        command(['git','add','--','release_manifest.json'])
                        command([config['report_python'],'-B',str(Path(config['runtime'])/'test_full_comparison.py')])
                        for script in ('tools/verify_repository.py','tools/run_cpu_checks.py'):
                            command([config['report_python'],'-B',script])
                        c.verify(checked);c.verify(pending['inputs'])
                        c.require(set(git('diff','--cached','--name-only').splitlines())<=allowed and not git('diff','--name-only')
                            and git('rev-parse','HEAD')==pending['base_commit'],'Concurrent repository change')
                        checked[str(root/'release_manifest.json')]=c.sha(root/'release_manifest.json')
                        exact_blobs(checked,'',root,command);pending.update(checks='PASS',tree=git('write-tree'));c.write(pending_path,pending)
                        message=out/'commit_message.txt';message.write_text('Publish complete matched five-method quality, fixed examples and receiver costs\n',encoding='utf-8')
                        command(['git','commit','-F',str(message)]);head=git('rev-parse','HEAD')
                        c.require(git('rev-parse','HEAD^{tree}')==pending['tree'],'Final commit tree changed')
                    record=dict(status='COMMITTED',phase='full_results',checks='PASS',full_plan_complete=True,rows=9000,
                        figure_cells=480,receiver_cost_frames=120,commit=head,published_files=checked,inputs=pending['inputs'],time=time.time())
                    c.write(receipt_path,record)
                command(['git','push','origin','main']);remote=git('ls-remote','origin','refs/heads/main').split()[0]
                c.require(remote==record['commit'],'Normal remote push not confirmed')
                record.update(status='PUSHED',remote_commit=remote,verified_time=time.time());c.write(receipt_path,record)
                print('PUSHED_FULL_MATCHED_COMPARISON',remote,flush=True);return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    try:publish(args.config)
    except BaseException as error:
        path=args.config.parent/'publication_failure.json'
        if not path.exists():c.write(path,dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False,
            preserve_index_and_worktree=True,time=time.time()))
        raise
