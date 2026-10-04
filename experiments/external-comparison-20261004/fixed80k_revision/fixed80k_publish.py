"""Checked normal publication of the full comparison using exact 80k Swin.

Scientific gates require all actual outputs. The original unfinished training
receives no invented completion. Git publication follows the existing checked
transaction, with explicit files, shared lock and no destructive recovery.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys
import time
import fixed80k_delivery_common as c

LIMIT=8_000_000

def bind(inputs,bindings):c.verify(bindings);inputs.update(bindings)

def prepare(config,config_path):
    c.validate_config(config);p=c.locations(config['root']);root=p['root'];out=p['out']
    original=c.original_modules(config)
    import publish_full_comparison as helper
    inputs={str(config_path):c.sha(config_path),**config['bindings']};files=set();proofs={}
    c.assert_original_paused(config)
    for stage in c.STAGES[:-1]:
        proofs[stage]=c.stage_proof(config,stage)
        for path,proof in proofs[stage].items():
            bind(inputs,{path:proof['sha256']});bind(inputs,c.receipt_bindings(c.read(path)))
    bind(inputs,c.final_table_gate(config))
    def add(source,target):
        source=Path(source);bind(inputs,{str(source):c.sha(source)})
        files.add(helper.copy_exact(source,target,root))
    experiment=root/'experiments/external-comparison-20261004/fixed80k_revision'
    for source in sorted([*p['revision'].glob('*.py'),*p['revision'].glob('*.md')]):
        c.require(config['bindings'].get(str(source))==c.sha(source),'Unregistered revision source')
        add(source,experiment/source.name)
    # Publish frozen dependencies separately; no old source artifact is replaced.
    for source in sorted(Path(config['runtime']).iterdir()):
        if source.suffix in ('.py','.json','.md') and str(source) in config['bindings']:
            c.require(config['bindings'][str(source)]==c.sha(source),'Changed frozen runtime dependency: '+source.name)
            add(source,experiment/'frozen_dependencies'/source.name)
    provenance=p['result']/'fixed80k_revision/provenance'
    add(config_path,provenance/'delivery_config.json')
    for name in ('user_request.json','pause_verified.json','fixed80k_registration.json','external_eval_config.json','origin_pause_snapshot.json'):
        add(p['revision']/name,provenance/name)
    for name in ('registration.json','qualification.json','selected_swin.json','completion.json'):
        add(p['revision']/'training_view'/name,provenance/('training_view_'+name))
    for stage,proof in proofs.items():
        for name in proof:
            path=Path(name);add(path,provenance/(stage+'__'+path.name));value=c.read(path)
            for name in value.get('outputs',{}):
                artifact=Path(name)
                if p['result'] in artifact.parents:
                    c.require(artifact.suffix in ('.csv','.json','.md','.svg','.png','.pdf','.txt'),'Unexpected result artifact')
                    files.add(artifact)
                elif artifact.parent==p['cost'] and artifact.name in ('per_frame.csv','summary.csv'):
                    add(artifact,p['result']/'receiver_cost'/artifact.name)
                elif artifact.suffix=='.json' and ('registration' in artifact.name or artifact.name in
                    ('metric_batch_qualification.json','score_input_admission.json')):
                    c.require(out in artifact.parents,'Registration proof outside the experiment')
                    add(artifact,provenance/artifact.relative_to(out).as_posix().replace('/','__'))
    archive=helper.archive_helper(config);links={}
    for folder in sorted({f.parent for f in files if f.stat().st_size>LIMIT}):
        links[folder]=helper.archive_dataset(folder,files,archive)
    final=p['result']/'final_comparison';report=final/'MATCHED_COMPARISON_REPORT.md'
    c.require(report in files,'Full real comparison report missing')
    text=report.read_text(encoding='utf-8')
    c.require('81,551' in text and '80,000' in text and 'budget_truncated=True' in text,'Report hides fixed milestone/unfinished training')
    for folder,mapping in links.items():
        if folder==final or final in folder.parents:
            for name,index in mapping.items():
                text=text.replace(']('+(folder/name).relative_to(final).as_posix()+')',
                    ']('+(folder/index).relative_to(final).as_posix()+')')
    view=final/'PUBLISHED_REPORT.md';files.add(helper.put_text(view,
        '> Exact user-selected 80k evaluation. Original training remains paused at 81,551 updates; no convergence claim.\n\n'+text))
    landing=root/'reports/fixed80k_external_comparison_20261004.md'
    files.add(helper.put_text(landing,'# 用户指定 80k 模型的完整同预算对照\n\n'
        '本次固定评测 SwinJSCC 第 80,000 步模型。原训练在 81,551 步安全暂停，后续权重保留但未用于本表；'
        '模型按用户指定步数选择，没有改用校准集选出的其他检查点，也不宣称原训练已经收敛。\n\n'
        'P、无类别数字 VAR、方法一、SwinJSCC 和冻结 HiFi-DiffCom 共 9,000 条实测结果，'
        '覆盖 N1024/N2048、1/7/13 dB、100 张图和三个噪声重复。包含 16 张固定源图、480 个方法单元，'
        '以及相同四张校准图的 120 帧接收代价。\n\n'
        '[完整指标、配对区间、固定样例与接收代价](../results/external_comparison_20261004/final_comparison/PUBLISHED_REPORT.md)\n\n'
        '本次评测完成不代表原 240k 上限与收敛训练协议已经完成。budget_truncated=True，user_requested_pause=True。\n'))
    for path in files:c.require(not path.is_symlink() and path.stat().st_size<=LIMIT,'Invalid/oversized publication artifact: '+str(path))
    return sorted(files),inputs

def exact_blobs(bindings,revision,root,command):
    for path,digest in bindings.items():
        relative=Path(path).relative_to(root).as_posix()
        c.require(hashlib.sha256(command(['git','show',revision+':'+relative],True)).hexdigest()==digest,
            'Committed bytes differ: '+relative)

def publish(config_path):
    import fcntl
    config_path=Path(config_path).resolve();config=c.validate_config(c.read(config_path));p=c.locations(config['root'])
    root=p['root'];out=p['delivery'];out.mkdir(parents=True,exist_ok=True)
    c.original_modules(config)
    import publish_full_comparison as helper
    c.require(not (out/'publication_failure.json').exists(),'Previous publication failure requires review')
    with (out/'publication.lock').open('a+') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (root/'.git/m2-recovery-publication.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with (out/'publication.log').open('a',encoding='utf-8') as log:
                def command(args,capture=False):return helper.run_command(args,root,log,capture)
                def git(*args):return command(['git',*args],True).decode().strip()
                c.require(git('rev-parse','--show-toplevel')==str(root) and git('branch','--show-current')=='main','Wrong worktree')
                c.require(git('remote','get-url','origin') in ('git@github.com:daiqizai/var-comm.git','https://github.com/daiqizai/var-comm.git'),'Wrong remote')
                receipt_path=out/'publication.json';pending_path=out/'publication_pending.json'
                previous=c.read(receipt_path) if receipt_path.exists() else None
                pending=c.read(pending_path) if pending_path.exists() else None
                command(['git','fetch','origin','main'])
                if previous:
                    c.require(previous.get('phase')=='user_fixed80k_full_results' and previous.get('checks')=='PASS'
                        and previous.get('full_plan_complete') is True and previous.get('status') in ('COMMITTED','PUSHED'),'Unknown fixed80k publication')
                    c.verify(previous['inputs']);exact_blobs(previous['published_files'],previous['commit'],root,command)
                    if previous['status']=='PUSHED':
                        command(['git','merge-base','--is-ancestor',previous['commit'],'origin/main']);return previous
                    c.require(git('rev-parse','HEAD')==previous['commit'] and not git('diff','--cached','--name-only')
                        and not git('diff','--name-only'),'Unpushed commit or worktree changed');record=previous
                else:
                    helper.verify_prior_publication(c.read(p['out']/'start_publication.json'),root,command)
                    if pending is None:
                        c.require(not git('diff','--cached','--name-only') and not git('diff','--name-only'),'Preserve unrelated tracked/staged edits')
                        base=git('rev-parse','HEAD');c.require(base==git('rev-parse','origin/main'),'Local/remote history needs review')
                        files,inputs=prepare(config,config_path)
                        pending=dict(phase='user_fixed80k_full_results',base_commit=base,inputs=inputs,
                            published_files={str(path):c.sha(path) for path in files});c.write(pending_path,pending)
                    c.require(pending.get('phase')=='user_fixed80k_full_results','Wrong pending transaction')
                    c.verify(pending['inputs']);c.verify(pending['published_files'])
                    checked=pending['published_files'];relative=[Path(path).relative_to(root).as_posix() for path in checked]
                    allowed=set(relative)|{'release_manifest.json'}
                    for args in (('diff','--cached','--name-only'),('diff','--name-only')):
                        c.require(set(git(*args).splitlines())<=allowed,'Preserve unrelated repository edits')
                    head=git('rev-parse','HEAD')
                    if head!=pending['base_commit']:
                        c.require(pending.get('checks')=='PASS' and pending.get('tree')==git('rev-parse','HEAD^{tree}')
                            and git('rev-parse','HEAD^')==pending['base_commit'],'Interrupted commit HEAD differs')
                        exact_blobs(checked,head,root,command)
                    else:
                        c.require(git('rev-parse','origin/main')==pending['base_commit'],'Remote changed before commit')
                        for start in range(0,len(relative),80):command(['git','add','-f','--',*relative[start:start+80]])
                        command([config['report_python'],'-B','tools/update_repository_manifest.py'])
                        command(['git','add','--','release_manifest.json'])
                        command([config['report_python'],'-B',str(c.HERE/'test_fixed80k_delivery.py')])
                        command([config['report_python'],'-B',str(Path(config['runtime'])/'test_full_comparison.py')])
                        for script in ('tools/verify_repository.py','tools/run_cpu_checks.py'):
                            command([config['report_python'],'-B',script])
                        c.verify(checked);c.verify(pending['inputs'])
                        c.require(set(git('diff','--cached','--name-only').splitlines())<=allowed and not git('diff','--name-only')
                            and git('rev-parse','HEAD')==pending['base_commit'],'Concurrent repository change')
                        checked[str(root/'release_manifest.json')]=c.sha(root/'release_manifest.json')
                        exact_blobs(checked,'',root,command);pending.update(checks='PASS',tree=git('write-tree'));c.write(pending_path,pending)
                        message=out/'commit_message.txt';message.write_text('Publish matched evaluation of user-selected exact 80k Swin checkpoint\n',encoding='utf-8')
                        command(['git','commit','-F',str(message)]);head=git('rev-parse','HEAD')
                        c.require(git('rev-parse','HEAD^{tree}')==pending['tree'],'Committed tree changed')
                    record=dict(status='COMMITTED',phase='user_fixed80k_full_results',checks='PASS',full_plan_complete=True,
                        selected_step=80000,actual_training_pause_step=81551,original_training_finished=False,
                        rows=9000,figure_cells=480,receiver_cost_frames=120,commit=head,published_files=checked,
                        inputs=pending['inputs'],time=time.time());c.write(receipt_path,record)
                command(['git','push','origin','main']);remote=git('ls-remote','origin','refs/heads/main').split()[0]
                c.require(remote==record['commit'],'Normal push not confirmed')
                record.update(status='PUSHED',remote_commit=remote,verified_time=time.time());c.write(receipt_path,record)
                print('PUSHED_USER_FIXED80K_COMPARISON',remote,flush=True);return record
