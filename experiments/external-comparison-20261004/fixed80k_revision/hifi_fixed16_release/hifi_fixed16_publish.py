"""Publish the complete exploratory fixed16 paired result, then stay paused."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from hifi_fixed16_report import paths,dependencies,require,validate_rows,validate_summary,validate_pairs,read_csv

HERE=Path(__file__).resolve().parent
LIMIT=8_000_000

def prepare(root,delivery_config_path):
    p=paths(root);c,_,fixed=dependencies(root);delivery=c.read(delivery_config_path)
    require(Path(delivery['root']).resolve()==p['root'],'Publication worktree differs')
    c.verify(delivery['bindings']);inputs={str(delivery_config_path):c.sha(delivery_config_path),**delivery['bindings']};files=set()
    import publish_full_comparison as helper
    def bind(items):c.verify(items);inputs.update(items)
    def add(source,target):
        source=Path(source);bind({str(source):c.sha(source)});files.add(helper.copy_exact(source,target,p['root']))
    receipts={}
    for name,status in [('reconstruction_completion.json','HIFI_FIXED16_RECONSTRUCTIONS_COMPLETE'),
        ('completion.json','HIFI_FIXED16_EVALUATION_COMPLETE'),('report_completion.json','HIFI_FIXED16_REPORT_COMPLETE')]:
        path=p['out']/name;value=c.read(path)
        require(value.get('status')==status and value.get('rows')==128 and value.get('sources')==16 and value.get('physical_frames')==64
            and value.get('synthetic') is False,'Fixed16 paired release incomplete: '+name)
        require(value.get('full_comparison_complete') is False,'Fixed16 paired release cannot declare full comparison complete')
        for field in ('bindings','input_bindings','outputs'):
            if field in value:bind(value[field])
        bind({str(path):c.sha(path)});receipts[name]=value
    report=receipts['report_completion.json'];metrics=receipts['completion.json']
    require(report.get('figure_cells')==128 and report.get('fixed_sources')==16
        and report.get('png_figures')==16 and report.get('pdf_figures')==4 and report.get('metric_groups')==8
        and report.get('paired_metric_groups')==52 and report.get('exploratory_fixed_subset') is True
        and report.get('full_evaluation_resume_authorized') is False
        and report.get('selected_step')==80000 and report.get('sample_selection_uses_quality') is False,
        'Paired fixed16 figures/metrics incomplete')
    require(report['evaluation_completion_sha256']==c.sha(p['out']/'completion.json')
        and metrics['reconstruction_completion_sha256']==c.sha(p['out']/'reconstruction_completion.json'),
        'Reconstruction, metrics and report differ')
    indexed=validate_rows(read_csv(p['result']/'metrics_per_frame.csv'))
    validate_summary(read_csv(p['result']/'metrics_summary.csv'),indexed)
    validate_pairs(read_csv(p['result']/'metrics_paired_intervals.csv'),indexed)
    require(metrics.get('selection_uses_development') is False and metrics.get('holdout_access') is False,
        'Development selection/holdout forbidden')
    config=c.read(p['config']);selected,selection=fixed.validate_selection(config['training_output'],root);bind(selection)
    experiment=p['root']/'experiments/external-comparison-20261004/fixed80k_revision/hifi_fixed16_release'
    for file in sorted([*HERE.glob('*.py'),*HERE.glob('*.md')]):
        require(delivery['bindings'].get(str(file))==c.sha(file),'New release source not bound before execution: '+file.name)
        add(file,experiment/file.name)
    # Preserve the unchanged parent adapters and registered dependencies at
    # the same canonical paths the later full publisher uses. Re-copying the
    # identical bytes later is admitted; no parent source is rewritten.
    for file in sorted([*HERE.parent.glob('*.py'),*HERE.parent.glob('*.md')]):
        if str(file) in delivery['bindings']:
            require(delivery['bindings'][str(file)]==c.sha(file),'Changed parent adapter')
            add(file,experiment.parent/file.name)
    runtime=p['base']/'runtime'
    for file in sorted(runtime.iterdir()):
        if file.suffix in ('.py','.json','.md') and str(file) in delivery['bindings']:
            require(delivery['bindings'][str(file)]==c.sha(file),'Changed frozen runtime dependency')
            add(file,experiment.parent/'frozen_dependencies'/file.name)
    provenance=p['result']/'provenance'
    add(delivery_config_path,provenance/'delivery_config.json');add(p['config'],provenance/'science_config.json')
    for file in (p['child']/'user_request.json',p['out']/'reconstruction_registration.json',p['out']/'report_registration.json'):
        add(file,provenance/file.name)
    # Include the fixed selection receipts and unchanged prerequisites needed
    # to audit this early result; the original training is never marked done.
    for file in [HERE.parent/'fixed80k_adapter.py',HERE.parent/'README.md',HERE.parent/'user_request.json',
        HERE.parent/'pause_verified.json',HERE.parent/'fixed80k_registration.json',HERE.parent/'origin_pause_snapshot.json']:
        require(file.exists(),'Missing fixed80k provenance: '+str(file));add(file,provenance/('fixed80k_'+file.name))
    for name in ('registration.json','qualification.json','selected_swin.json','completion.json'):
        add(Path(config['training_output'])/name,provenance/('training_view_'+name))
    for name,value in receipts.items():
        add(p['out']/name,provenance/name)
        for path in value.get('outputs',{}):
            artifact=Path(path)
            if p['result'] in artifact.parents:
                require(artifact.suffix in ('.csv','.json','.md','.svg','.png','.pdf','.txt'),'Unexpected result artifact')
                files.add(artifact)
            elif artifact.suffix=='.json' and ('registration' in artifact.name or artifact.name in ('metric_batch_qualification.json','label_metadata.json')):
                require(p['child'] in artifact.parents,'Unregistered auxiliary proof')
                add(artifact,provenance/artifact.relative_to(p['child']).as_posix().replace('/','__'))
    # Use the exact registered historical archive implementation.
    archive_path=p['root']/'experiments/unified-metrics-20261002/publish.py'
    require(delivery['bindings'].get(str(archive_path))==c.sha(archive_path),'Archive helper not frozen')
    archive=helper.archive_helper({'bindings':delivery['bindings'],'root':str(p['root'])})
    links={}
    for folder in sorted({path.parent for path in files if path.stat().st_size>LIMIT}):
        links[folder]=helper.archive_dataset(folder,files,archive)
    original=p['result']/'HIFI_FIXED16_REPORT.md'
    require(original in files,'Fixed16 report not a completed output')
    text=original.read_text(encoding='utf-8')
    require('81,551' in text and '80,000' in text and 'budget_truncated=True' in text,'Missing unfinished training disclosure')
    for folder,mapping in links.items():
        if folder==p['result'] or p['result'] in folder.parents:
            for name,index in mapping.items():text=text.replace(']('+(folder/name).relative_to(p['result']).as_posix()+')',
                ']('+(folder/index).relative_to(p['result']).as_posix()+')')
    view=p['result']/'PUBLISHED_REPORT.md';files.add(helper.put_text(view,
        '> Exploratory fixed16 paired release: 64 physical frames and 128 outputs. Full evaluation remains paused.\n\n'+text))
    landing=p['root']/'reports/hifi_fixed16_20261004.md'
    files.add(helper.put_text(landing,'# HiFi-DiffCom 第一档：固定 16 图配对评测\n\n'
        '已完成 N1024/N2048、7/13 dB、原固定 16 张图、噪声种子 2001 的 64 个实际信道帧，'
        '得到 128 个同波形 Swin/HiFi 配对输出。包含全部 13 项统一指标、16 源图 bootstrap 区间、'
        '52 项配对差值及 128 个重建对照单元。\n\n'
        '[完整指标、配对区间和原图/Swin/HiFi样例](../results/external_comparison_20261004/fixed80k_revision/hifi_fixed16_release/evaluation/PUBLISHED_REPORT.md)\n\n'
        '**这是探索性固定样例集合，不代表完整 100 张源图或 ImageNet 总体。分类器分歧也不是人工语义判决。** '
        '模型固定第 80,000 步；训练在 81,551 步暂停，预算截断，不宣称收敛。\n\n'
        '本档完成并推送后停止，完整评测继续保持暂停，等待下一步决定。\n'))
    for file in files:require(not file.is_symlink() and file.stat().st_size<=LIMIT,'Invalid/oversized staged artifact: '+str(file))
    return sorted(files),inputs

def exact_blobs(bindings,revision,root,command):
    for path,digest in bindings.items():
        relative=Path(path).relative_to(root).as_posix()
        require(hashlib.sha256(command(['git','show',revision+':'+relative],True)).hexdigest()==digest,'Published bytes differ: '+relative)

def publish(root,delivery_config_path):
    import fcntl
    p=paths(root);c,_,_=dependencies(root);config=c.read(delivery_config_path)
    require(Path(config['root']).resolve()==p['root'],'Wrong delivery root');c.verify(config['bindings'])
    import publish_full_comparison as helper
    python=config.get('report_python') or config.get('python') or str(p['root']/'outputs/UNIFIED-METRICS-20261002/environment/bin/python')
    child=p['child'];receipt_path=child/'release_publication.json';pending_path=child/'publication_pending.json'
    require(not (child/'publication_failure.json').exists(),'Prior publication failure requires review')
    with (child/'publication.lock').open('a+') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (p['root']/'.git/m2-recovery-publication.lock').open('a+') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            with (child/'publication.log').open('a',encoding='utf-8') as log:
                def command(args,capture=False):return helper.run_command(args,p['root'],log,capture)
                def git(*args):return command(['git',*args],True).decode().strip()
                require(git('rev-parse','--show-toplevel')==str(p['root']) and git('branch','--show-current')=='main','Wrong worktree')
                require(git('remote','get-url','origin') in ('git@github.com:daiqizai/var-comm.git','https://github.com/daiqizai/var-comm.git'),'Wrong remote')
                command(['git','fetch','origin','main'])
                previous=c.read(receipt_path) if receipt_path.exists() else None
                pending=c.read(pending_path) if pending_path.exists() else None
                if previous:
                    require(previous.get('phase')=='hifi_fixed16_release' and previous.get('checks')=='PASS'
                        and previous.get('full_comparison_complete') is False and previous.get('status') in ('COMMITTED','PUSHED'),'Unknown early transaction')
                    c.verify(previous['inputs']);exact_blobs(previous['published_files'],previous['commit'],p['root'],command)
                    if previous['status']=='PUSHED':
                        command(['git','merge-base','--is-ancestor',previous['commit'],'origin/main']);return previous
                    require(git('rev-parse','HEAD')==previous['commit'] and not git('diff','--cached','--name-only')
                        and not git('diff','--name-only'),'Interrupted early commit changed');record=previous
                else:
                    helper.verify_prior_publication(c.read(p['base']/'fixed80k_revision/swin_early_release/release_publication.json'),p['root'],command)
                    if pending is None:
                        require(not git('diff','--cached','--name-only') and not git('diff','--name-only'),'Preserve unrelated repository edits')
                        base=git('rev-parse','HEAD');require(base==git('rev-parse','origin/main'),'Local/remote history needs review')
                        files,inputs=prepare(root,delivery_config_path)
                        pending=dict(phase='hifi_fixed16_release',base_commit=base,inputs=inputs,published_files={str(v):c.sha(v) for v in files})
                        c.write(pending_path,pending)
                    require(pending.get('phase')=='hifi_fixed16_release','Wrong pending transaction')
                    c.verify(pending['inputs']);c.verify(pending['published_files'])
                    checked=pending['published_files'];relative=[Path(v).relative_to(p['root']).as_posix() for v in checked]
                    allowed=set(relative)|{'release_manifest.json'}
                    for args in (('diff','--cached','--name-only'),('diff','--name-only')):
                        require(set(git(*args).splitlines())<=allowed,'Preserve unrelated worktree/index changes')
                    head=git('rev-parse','HEAD')
                    if head!=pending['base_commit']:
                        require(pending.get('checks')=='PASS' and pending.get('tree')==git('rev-parse','HEAD^{tree}')
                            and git('rev-parse','HEAD^')==pending['base_commit'],'Interrupted early commit identity differs')
                        exact_blobs(checked,head,p['root'],command)
                    else:
                        require(git('rev-parse','origin/main')==pending['base_commit'],'Remote changed before publication')
                        for start in range(0,len(relative),80):command(['git','add','-f','--',*relative[start:start+80]])
                        command([python,'-B','tools/update_repository_manifest.py']);command(['git','add','--','release_manifest.json'])
                        for name in ('test_hifi16.py','test_hifi_fixed16_report.py','test_hifi_fixed16_delivery.py'):
                            require((HERE/name).exists(),'Missing required early release check '+name)
                            command([python,'-B',str(HERE/name)])
                        for script in ('tools/verify_repository.py','tools/run_cpu_checks.py'):command([python,'-B',script])
                        c.verify(checked);c.verify(pending['inputs'])
                        require(set(git('diff','--cached','--name-only').splitlines())<=allowed and not git('diff','--name-only')
                            and git('rev-parse','HEAD')==pending['base_commit'],'Concurrent repository changes')
                        checked[str(p['root']/'release_manifest.json')]=c.sha(p['root']/'release_manifest.json')
                        exact_blobs(checked,'',p['root'],command);pending.update(checks='PASS',tree=git('write-tree'));c.write(pending_path,pending)
                        message=child/'commit_message.txt';message.write_text('Publish exploratory fixed16 paired HiFi and Swin evaluation; keep full run paused\n',encoding='utf-8')
                        command(['git','commit','-F',str(message)]);head=git('rev-parse','HEAD')
                        require(git('rev-parse','HEAD^{tree}')==pending['tree'],'Early commit tree differs')
                    record=dict(status='COMMITTED',phase='hifi_fixed16_release',checks='PASS',full_comparison_complete=False,
                        rows=128,sources=16,physical_frames=64,metric_groups=8,figure_cells=128,selected_step=80000,actual_training_pause_step=81551,
                        exploratory_fixed_subset=True,full_evaluation_resume_authorized=False,
                        original_training_finished=False,commit=head,published_files=checked,inputs=pending['inputs'],time=time.time())
                    c.write(receipt_path,record)
                command(['git','push','origin','main']);remote=git('ls-remote','origin','refs/heads/main').split()[0]
                require(remote==record['commit'],'Normal fixed16 push not verified')
                record.update(status='PUSHED',remote_commit=remote,verified_time=time.time());c.write(receipt_path,record)
                print('PUSHED_HIFI_FIXED16_RELEASE',remote,flush=True);return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--delivery-config',type=Path,required=True)
    args=parser.parse_args()
    try:publish(args.root,args.delivery_config)
    except BaseException as error:
        p=paths(args.root);c,_,_=dependencies(args.root)
        failure=p['child']/'publication_failure.json'
        if not failure.exists():c.write(failure,dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False,
            preserve_index_and_worktree=True,time=time.time()))
        raise

if __name__=='__main__':main()
