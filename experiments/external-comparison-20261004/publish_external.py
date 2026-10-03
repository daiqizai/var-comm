"""Publish reviewed external-comparison artifacts, preserving every old study."""
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import re
import subprocess
import sys
import time

ROOT=Path('/home/liulu/projects/VAR_COMM')
OUT=ROOT/'outputs/EXTERNAL-COMPARISON-20261004'
EXP=ROOT/'experiments/external-comparison-20261004'
RESULT=ROOT/'results/external_comparison_20261004'
LIMIT=8_000_000


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    with t.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(t,p)
def verify(bindings):
    if not isinstance(bindings,dict): raise RuntimeError('Missing hash bindings')
    for p,h in bindings.items():
        if sha(p)!=h: raise RuntimeError('Changed input: '+str(p))
def require(value,message):
    if not value: raise RuntimeError(message)
def bind(inputs,bindings):
    verify(bindings)
    for p,h in bindings.items():
        require(p not in inputs or inputs[p]==h,'Conflicting hash: '+p);inputs[p]=h
def resolved_outputs(receipt,folder):
    require(bool(receipt.get('outputs')),'Empty completion outputs')
    return {str(Path(k) if Path(k).is_absolute() else folder/k):v for k,v in receipt['outputs'].items()}
def copy(source,target):
    source,target=Path(source),Path(target)
    if source.is_symlink() or target.is_symlink() or (source.stat().st_size>LIMIT and source.suffix!='.csv'): raise RuntimeError('Invalid publish file')
    require(ROOT.resolve() in target.resolve().parents,'Publish target escaped repository')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and sha(source)!=sha(target): raise RuntimeError('Existing artifact differs: '+str(target))
    if not target.exists(): shutil.copyfile(source,target)
    return target


def prerequisites(inputs):
    folder=OUT/'step0_statistics';p=folder/'completion.json';value=read(p)
    require(value.get('status')=='STEP0_STATISTICS_COMPLETE' and value.get('rows')==6600
        and value.get('cells')==22 and value.get('bootstrap')==10000 and value.get('source_unit') is True
        and value.get('model_inference') is False,'Completed historical statistics required')
    require(value.get('code_sha256')==sha(OUT/'runtime/step0_statistics.py'),'Statistics source differs')
    bind(inputs,value['input_bindings']);bind(inputs,resolved_outputs(value,folder));bind(inputs,{str(p):sha(p)})
    folder=OUT/'reference_metrics_cpu_full';p=folder/'completion.json';value=read(p)
    conditions=['identity','unrelated','same_class','gaussian_blur_sigma1','gaussian_noise_sigma2_255','jpeg_quality90']
    require(value.get('status')=='SIX_REFERENCE_METRICS_COMPLETE' and value.get('sources')==100
        and value.get('rows')==600 and value.get('conditions')==conditions and value.get('bootstrap_replicates')==10000
        and value.get('bootstrap_seed')==20261002 and value.get('statistical_unit')=='source_image'
        and value.get('noise_triplication') is False and value.get('holdout_access') is False
        and value.get('training_updates')==0 and value.get('policy_selection_updates')==0,'Reference metrics incomplete')
    rp=folder/'registration.json';reg=read(rp)
    require(value['registration_sha256']==sha(rp) and value['prepared_sha256']==reg['prepared_sha256']
        and reg.get('status')=='REGISTERED_SIX_REFERENCE_METRICS' and reg.get('target_sources')==100
        and reg.get('rows')==600 and reg.get('conditions')==conditions and reg.get('device')=='cpu'
        and reg.get('reference_only') is True and reg.get('holdout_access') is False,'Reference registration differs')
    bind(inputs,reg['model_bindings']);bind(inputs,reg['source_bindings'])
    bind(inputs,resolved_outputs(value,folder));bind(inputs,{str(p):sha(p),str(rp):sha(rp)})
    perf=read(folder/'performance.json')
    require(perf.get('sources')==100 and perf.get('rows')==600 and perf.get('device')=='cpu'
        and len(perf.get('source_seconds',[]))==100,'Incomplete reference performance record')


def process_live(record):
    try:
        proc=Path('/proc')/str(int(record['pid']));fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        argv=[part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
        return fields[0]!='Z' and fields[19]==str(record['start_ticks']) and argv==record['command']
    except (OSError,KeyError,ValueError,IndexError): return False


def controller_gate(inputs,completed=False):
    cp=OUT/'controller_config.json';config=read(cp);bind(inputs,{str(cp):sha(cp),**config['bindings']})
    require(config.get('mode')!='qualification_only' and bool(config.get('evaluation')),'Full external pipeline required')
    if completed:
        p=OUT/'controller/completion.json';done=read(p)
        require(done.get('status')=='EXTERNAL_TRAINING_AND_EVALUATION_COMPLETE'
            and done.get('config_sha256')==sha(cp) and set(done.get('stages',{}))==
            {'qualification','hifi_preflight','training','hifi_qualification','reconstruct','score'},'Incomplete controller')
        bind(inputs,{str(p):sha(p)})
        for proof in done['stages'].values():
            bind(inputs,{proof['path']:proof['sha256']})
            require(read(proof['path']).get('status')==proof['status'],'Stage completion identity differs')
        return done
    require(not (OUT/'controller/failure.json').exists(),'Controller needs review')
    state=read(OUT/'controller/status.json');launch=read(OUT/'controller/controller_launch.json')
    dispatchpath=OUT/'controller/dispatch.json';dispatch=read(dispatchpath)
    require(state.get('config_sha256')==sha(cp) and state.get('status') not in
        ('PAUSED','FAILED_REQUIRES_REVIEW'),'Controller is not running')
    for item in (launch,dispatch):
        require(item['config_sha256']==sha(cp) and item['source_bindings']==config['bindings'],'Launch identity differs')
    require((launch['pid'],str(launch['start_ticks']))==(dispatch['pid'],str(dispatch['start_ticks']))
        and process_live(dispatch),'Registered controller is not alive')
    # Mutable status is captured once, rather than falsely treated as immutable.
    snapshot=OUT/'start_controller_snapshot.json'
    if not snapshot.exists():write(snapshot,dict(captured_at=time.time(),status=state,launch=launch,dispatch=dispatch,config_sha256=sha(cp)))
    require(read(snapshot)['config_sha256']==sha(cp) and 'dispatch' in read(snapshot),'Snapshot config/dispatch differs')
    bind(inputs,{str(snapshot):sha(snapshot)})
    return read(snapshot)['status']


def results_gate(inputs):
    controller_gate(inputs,True)
    folder=OUT/'evaluation';p=folder/'completion.json';done=read(p)
    require(done.get('status')=='EXTERNAL_EVALUATION_COMPLETE' and done.get('sources')==100
        and done.get('physical_frames')==1800 and done.get('rows')==3600 and done.get('metric_groups')==12
        and done.get('synthetic') is False and 'sampler_step_limit' in done and done['sampler_step_limit'] is None
        and done.get('selection_uses_development') is False and done.get('holdout_access') is False
        and done.get('paired_bootstrap_unit')=='source_mean_after_three_noise_repeats','Incomplete registered evaluation')
    bind(inputs,done['bindings']);bind(inputs,resolved_outputs(done,folder));bind(inputs,{str(p):sha(p)})
    rp=folder/'report_completion.json';report=read(rp)
    require(report.get('status')=='EXTERNAL_TWO_METHOD_REPORT_COMPLETE' and report.get('synthetic') is False
        and report.get('full_plan_complete') is False and report.get('missing_matched_controls')==['P','D_U','M1']
        and report.get('sources')==100 and report.get('physical_frames')==1800 and report.get('rows')==3600
        and report.get('fixed_sources')==16 and report.get('png_figures')==24 and report.get('pdf_figures')==6
        and report.get('figure_cells')==192 and report.get('sample_selection_uses_quality') is False
        and report.get('holdout_access') is False and report.get('evaluation_completion_sha256')==sha(p)
        and report.get('report_registration_sha256')==sha(folder/'report_registration.json'),'Two-method report incomplete')
    bind(inputs,report['input_bindings']);bind(inputs,resolved_outputs(report,folder));bind(inputs,{str(rp):sha(rp)})
    return done,report


def archive_helper(inputs):
    p=ROOT/'experiments/unified-metrics-20261002/publish.py'
    qp=ROOT/'outputs/HISTORICAL-METRICS-R3-20261003/queue_registration.json'
    expected=read(qp)['shared_metric_bindings'].get(str(p))
    require(expected==sha(p),'Archive helper differs from registered source')
    bind(inputs,{str(p):expected,str(qp):sha(qp)})
    spec=importlib.util.spec_from_file_location('_external_exact_csv_archive',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def archive_dataset(folder,files,helper):
    manifest=helper.archive_tables(folder,limit=LIMIT)
    excluded={folder/rel for rel in manifest['tables']}
    for rel,entry in manifest['tables'].items():
        require(sha(folder/rel)==entry['sha256'],'Original CSV changed')
        helper.verify_archive_entry(folder,entry)
        index=folder/entry['index']
        note=index.read_text(encoding='utf-8').replace('with `publish.py --restore-tables`',
            'with the registered `experiments/unified-metrics-20261002/publish.py` helper: '
            'call `restore_tables(Path(dataset_directory))` for this dataset directory')
        index.write_text(note,encoding='utf-8')
    files[:]=[p for p in files if p not in excluded]
    files.extend(p for p in (folder/'table_shards').iterdir() if p.is_file())
    original=folder/('EXTERNAL_REPORT.md' if folder.name=='evaluation' else 'report.md')
    if manifest['tables'] and original.exists():
        value=original.read_text(encoding='utf-8')
        for rel,entry in manifest['tables'].items():value=value.replace(']('+rel+')',']('+entry['index']+')')
        value='> Publication view: large CSV links point to lossless shards. Original report and table hashes remain unchanged.\n\n'+value
        view=folder/'PUBLISHED_REPORT.md'
        if view.exists():require(view.read_text(encoding='utf-8')==value,'Published report view differs')
        else:view.write_text(value,encoding='utf-8')
        files.append(view)
    return manifest


def confidence_gate(inputs):
    path=OUT/'history_confidence/completion.json';value=read(path)
    require(value.get('status')=='HISTORICAL_CONFIDENCE_COMPLETE' and value.get('rows')==6600
        and value.get('cells')==22 and value.get('sources')==100 and value.get('argmax_all_match') is True
        and value.get('confidence_threshold')==.5 and value.get('bootstrap')==10000
        and value.get('bootstrap_source_unit') is True and value.get('only_model_loaded')=='resnet50'
        and value.get('device')=='cpu' and value.get('synthetic') is False
        and value.get('original_inputs_modified') is False and value.get('training_updates')==0
        and value.get('policy_selection_updates')==0,'Historical confidence incomplete')
    for key in ('input_bindings','source_bindings','outputs','probability_bindings'):bind(inputs,value[key])
    bind(inputs,{str(path):sha(path)})


def prepare(phase):
    files=[];inputs={}
    prerequisites(inputs);confidence_gate(inputs)
    state=controller_gate(inputs) if phase=='start' else None
    evidence=results_gate(inputs) if phase=='results' else None
    def add(source,target):
        source=Path(source);bind(inputs,{str(source):sha(source)})
        files.append(copy(source,target))
    for p in sorted((OUT/'runtime').iterdir()):
        if p.suffix in ('.py','.json') or p.name=='own_controls_README.md':add(p,EXP/p.name)
    add(Path(__file__).resolve(),EXP/'publish_external.py')
    add(OUT/'register_controller.py',EXP/'register_controller.py')
    if (OUT/'completion_delivery.py').exists():add(OUT/'completion_delivery.py',EXP/'completion_delivery.py')
    for name in ('USER_PLAN.md','PROTOCOL.md','sampling_preregistration.json','fixed_examples.json','metric_reference_preregistration.json'):
        add(OUT/name,EXP/name)
    add(OUT/'literature/verified_primary_sources.json',EXP/'verified_primary_sources.json')
    add(OUT/'INVENTORY.md',ROOT/'reports/external_inventory_20261004.md')
    add(OUT/'inventory_v2/summary.json',RESULT/'inventory/scan_summary.json')
    add(OUT/'inventory_assets.json',RESULT/'inventory/assets.json')
    datasets=[('step0_statistics',('per_frame.csv','summary.csv','report.md','completion.json','semantic_error_resource.png','semantic_error_resource.pdf','semantic_error_resource.svg')),
              ('reference_metrics_cpu_full',('per_source.csv','summary.csv','report.md','completion.json','registration.json','performance.json')),
              ('history_confidence',('per_frame.csv','summary.csv','paired_vs_P.csv','source_confidence.csv','report.md','completion.json','registration.json'))]
    for folder,names in datasets:
        for name in names:add(OUT/folder/name,RESULT/folder/name)
    mp=OUT/'step0/step0_manifest.json';manifest=read(mp)
    require(manifest.get('status')=='EXACT_COMPLETED_FLOAT_RGB_CPU_EXPORT_PASS'
        and manifest.get('gpu_inference') is False and manifest.get('metric_inference') is False,'Historical pixel export not admitted')
    bind(inputs,manifest['input_bindings']);bind(inputs,manifest['own_source_bindings'])
    for p in sorted((OUT/'step0/figures').glob('*')):
        if p.suffix in ('.png','.pdf','.json'):add(p,RESULT/'step0_figures'/p.name)
    add(mp,RESULT/'step0_figures/step0_manifest.json')
    for name in ('native_cpu_tests.log','native_cpu_tests_after_gather_fix.log','step0_cpu_tests.log','hifi_qualification_cpu_tests.log','controller_cpu_tests.log','external_eval_cpu_tests.log','report_confidence_tests.log','own_controls_phy_tests.log','own_controls_cpu_tests.log'):
        log=(OUT/name).read_text(encoding='utf-8')
        require(re.search(r'^OK(?: \([^\r\n]+\))?\r?$',log,re.MULTILINE)
            and 'FAILED (' not in log and 'Traceback (most recent call last)' not in log,'Missing passing tests '+name)
        add(OUT/name,RESULT/'provenance'/name)
    for relative,expected in (('swin_training/qualification.json','PASS'),('hifi_preflight/preflight.json','ENGINEERING_PREFLIGHT')):
        path=OUT/relative;probe=read(path)
        require(probe.get('status')==expected and probe.get('formal_training_updates')==0,
            'Missing real GPU engineering qualification: '+relative)
        add(path,RESULT/'provenance'/relative.replace('/','_'))
    for name in ('swin_qualification_attempt1.log','swin_qualification_attempt2.log','hifi_preflight_attempt1.log'):
        add(OUT/name,RESULT/'provenance'/name)
    for name in ('calibration_cost_probe.py','calibration_cost_probe.json','calibration_cost_probe.log'):
        add(OUT/'diagnostics'/name,RESULT/'provenance'/name)
    add(OUT/'controller_config.json',RESULT/'provenance/controller_config.json')
    if phase=='start':
        dispatch_snapshot=OUT/'start_dispatch_snapshot.json'
        snapshot_dispatch=read(OUT/'start_controller_snapshot.json')['dispatch']
        if not dispatch_snapshot.exists():write(dispatch_snapshot,snapshot_dispatch)
        require(read(dispatch_snapshot)==snapshot_dispatch,'Initial dispatch snapshot differs')
        add(dispatch_snapshot,RESULT/'provenance/controller_dispatch.json')
        add(OUT/'start_controller_snapshot.json',RESULT/'provenance/start_controller_snapshot.json')
    helper=archive_helper(inputs)
    archives={folder:archive_dataset(RESULT/folder,files,helper) for folder,_ in datasets}
    if phase=='start':
        report=ROOT/'reports/external_comparison_20261004.md'
        text='''# SwinJSCC 与 HiFi-DiffCom 外部对比：已启动

本轮训练一个全新 SwinJSCC SA+RA MSE 模型，覆盖总 N1024/N2048、整数 SNR1–13 dB。使用原 ImageNet20k、校准1k、256×256 图像，只在校准集选模。旧 P512/P1024 的预算截断说明保留。

[执行协议](../experiments/external-comparison-20261004/PROTOCOL.md) · [资产盘点](external_inventory_20261004.md) · [历史统计](../results/external_comparison_20261004/step0_statistics/report.md) · [六组指标参照](../results/external_comparison_20261004/reference_metrics_cpu_full/report.md) · [分类置信度](../results/external_comparison_20261004/history_confidence/report.md)

## 已完成

- 6,600 帧历史结果的源配对统计与分类置信度完成，分类器预测逐项与旧测量一致。
- 100 源图、六组条件共 600 帧的无信道指标参照完成；具体数值及区间见上述报告。
- 固定样例已事前登记；历史资源图保留付费类别、Dc/D0 和缺测点标注。
- CPU 工程检查、真实 GPU Swin 训练/恢复资格、随机 Swin 与真实 ADM 接口预检均通过。修复了严格确定性下通道选择反向的算子兼容问题，首轮失败日志保留；测试通过不代表新模型质量已经完成测量。

## 后台执行

Swin 与 ADM 的工程资格已在 GPU 空闲时完成。后台会先确认旧补评完成并推送，再正式训练 Swin；随后进行选定模型 HiFi 资格、两接收器同波形完整采样和统一指标。

总资源包括掩码排名、功率、CRC、tail 和冗余。N1024 正文 768/头 256；N2048 正文 1664/头 384；E=2N。保留原头码率不高于 1/4 的规则，不宣称该分配已经优化。CRC 失败时两接收器均输出同一灰图，并计入指标。

20k 为首次里程碑，每 2500 步完整校准；按登记平台规则两次降低学习率，最早 80k 停止，上限 240k。未满足最终平台规则则标注 budget_truncated。吞吐、显存与 ETA 由实际 GPU 资格及稳定训练测量更新。

## 待完成

新训练、正式外部质量结果，以及同预算本项目 P/D_U/方法一的整合仍待完成。N2048 无类别数字与方法一使用独立登记的校准；未测点不填补，不用旧付费类别结果替代。两外部接收器完成不等于整个对照计划完成。全程不访问 holdout。
'''
        text+='\n提交时的后台状态快照：`'+state['status']+'`。实时状态见服务器 controller/status.json。\n'
    else:
        done,proof=evidence
        admitted=set(resolved_outputs(done,OUT/'evaluation'))|set(resolved_outputs(proof,OUT/'evaluation'))
        for value in sorted(admitted):
            p=Path(value)
            if RESULT/'evaluation' in p.parents:
                require(p.suffix in ('.json','.csv','.md','.png','.pdf','.svg'),'Unexpected evaluation artifact')
                files.append(p)
        for name in ('completion.json','report_completion.json','report_registration.json'):
            add(OUT/'evaluation'/name,RESULT/'provenance'/('evaluation_'+name))
        add(OUT/'controller/completion.json',RESULT/'provenance/controller_completion.json')
        archives['evaluation']=archive_dataset(RESULT/'evaluation',files,helper)
        view='PUBLISHED_REPORT.md' if archives['evaluation']['tables'] else 'EXTERNAL_REPORT.md'
        report=ROOT/'reports/external_baselines_20261004.md'
        text=('# SwinJSCC 与 HiFi-DiffCom：两外部接收器结果\n\n'
              '100 源图、三个噪声重复、两个预算和三个 SNR，共 1,800 个物理帧、3,600 条接收器结果。'
              '两接收器使用完全相同的实测波形；完整生成日程与统一指标均已完成。\n\n'
              '[完整外部报告](../results/external_comparison_20261004/evaluation/'+view+')\n\n'
              '**整个对照计划尚未完成。** 本次仅发布两外部接收器；P、D_U、方法一的同预算对照仍需整合。'
              '结果凭证保留 `full_plan_complete=False`，不以工程探针或缺测点代替科学结果。\n')
    for folder,manifest in archives.items():
        if manifest['tables']:text=text.replace('/'+folder+'/report.md)','/'+folder+'/PUBLISHED_REPORT.md)')
    report.parent.mkdir(parents=True,exist_ok=True)
    if report.exists():require(report.read_text(encoding='utf-8')==text,'Existing report differs')
    else:report.write_text(text,encoding='utf-8')
    files.append(report)
    for p in files:require(not p.is_symlink() and p.stat().st_size<=LIMIT,'Invalid staged file: '+str(p))
    # Do not touch the three shared status documents while R3 may publish.
    return sorted(set(files)),inputs

def cpu_environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')


def run_command(args,log,capture=False):
    # Captured stdout may be a PNG/PDF/blob. Do not decode it into the text log.
    log.write('COMMAND '+repr(args)+'\n');log.flush()
    proc=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,env=cpu_environment())
    if not capture:log.write(proc.stdout.decode(errors='replace'))
    log.write(proc.stderr.decode(errors='replace'));log.flush()
    require(proc.returncode==0,'Command failed: '+repr(args))
    return proc.stdout if capture else None


def publish(phase):
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    # Share the existing R3 transaction lock; never reset or kill another writer.
    with (ROOT/'.git/m2-recovery-publication.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (OUT/(phase+'_publication.log')).open('a',encoding='utf-8') as log:
            def command(args,capture=False):return run_command(args,log,capture)
            def git(*args):return command(['git',*args],True).decode().strip()
            def exact_blobs(bindings,revision):
                for path,digest in bindings.items():
                    rel=Path(path).relative_to(ROOT).as_posix()
                    require(hashlib.sha256(command(['git','show',revision+':'+rel],True)).hexdigest()==digest,
                            'Git bytes differ: '+rel)
            require(git('rev-parse','--show-toplevel')==str(ROOT) and git('branch','--show-current')=='main','Wrong worktree')
            require(git('remote','get-url','origin') in ('git@github.com:daiqizai/var-comm.git',
                    'https://github.com/daiqizai/var-comm.git'),'Wrong remote')
            receipt_path=OUT/(phase+'_publication.json');pending_path=OUT/(phase+'_publication_pending.json')
            previous=read(receipt_path) if receipt_path.exists() else None
            pending=read(pending_path) if pending_path.exists() else None
            command(['git','fetch','origin','main'])
            if previous:
                require(previous.get('phase')==phase and previous.get('checks')=='PASS'
                        and previous.get('status') in ('COMMITTED','PUSHED'),'Unknown prior publication')
                verify(previous['inputs']);exact_blobs(previous['published_files'],previous['commit'])
                if previous['status']=='PUSHED':
                    command(['git','merge-base','--is-ancestor',previous['commit'],'origin/main'])
                    print('ALREADY_PUSHED',previous['commit'],flush=True);return previous
                require(git('rev-parse','HEAD')==previous['commit'],'Unpushed publication is no longer HEAD')
                require(not git('diff','--cached','--name-only') and not git('diff','--name-only'),
                        'Preserve concurrent edits before push')
                record=previous
            else:
                if pending is None:
                    require(not git('diff','--cached','--name-only') and not git('diff','--name-only'),
                            'Preserve existing tracked/staged edits')
                    base=git('rev-parse','HEAD')
                    require(base==git('rev-parse','origin/main'),'Existing local/remote commits need review')
                    files,inputs=prepare(phase)
                    bind(inputs,{str(ROOT/p):sha(ROOT/p) for p in ('tools/update_repository_manifest.py',
                                'tools/verify_repository.py','tools/run_cpu_checks.py')})
                    pending=dict(phase=phase,base_commit=base,inputs=inputs,
                                 published_files={str(p):sha(p) for p in files})
                    write(pending_path,pending)
                require(pending.get('phase')==phase,'Pending transaction phase differs')
                verify(pending['inputs']);verify(pending['published_files'])
                checked=pending['published_files'];rel=[Path(p).relative_to(ROOT).as_posix() for p in checked]
                allowed=set(rel)|{'release_manifest.json'}
                for args in (('diff','--cached','--name-only'),('diff','--name-only')):
                    require(set(git(*args).splitlines())<=allowed,'Preserve unrelated repository edits')
                head=git('rev-parse','HEAD')
                if head!=pending['base_commit']:
                    # Recover only the exact checked commit after a crash
                    # between git commit and the durable COMMITTED receipt.
                    require(pending.get('checks')=='PASS' and pending.get('tree')==git('rev-parse','HEAD^{tree}')
                        and git('rev-parse','HEAD^')==pending['base_commit'],'Unexpected interrupted-commit HEAD')
                    exact_blobs(checked,head)
                else:
                    require(git('rev-parse','origin/main')==pending['base_commit'],'Remote changed before commit')
                    # Every path is already hash-reviewed in the frozen pending
                    # manifest. Explicitly include its small provenance logs,
                    # which the repository's general *.log ignore rule excludes.
                    for start in range(0,len(rel),80):command(['git','add','-f','--',*rel[start:start+80]])
                    command([sys.executable,'-B','tools/update_repository_manifest.py'])
                    command(['git','add','--','release_manifest.json'])
                    for script in ('tools/verify_repository.py','tools/run_cpu_checks.py'):
                        command([sys.executable,'-B',script])
                    verify(checked);verify(pending['inputs'])
                    require(set(git('diff','--cached','--name-only').splitlines())<=allowed
                        and not git('diff','--name-only') and git('rev-parse','HEAD')==pending['base_commit'],
                        'Concurrent repository edit')
                    checked[str(ROOT/'release_manifest.json')]=sha(ROOT/'release_manifest.json')
                    exact_blobs(checked,'')
                    pending.update(checks='PASS',tree=git('write-tree'));write(pending_path,pending)
                    message=OUT/(phase+'_commit_message.txt')
                    message.write_text('Register matched-budget Swin and HiFi evaluation; publish historical references\n'
                        if phase=='start' else 'Publish Swin and frozen HiFi results; matched controls remain pending\n',encoding='utf-8')
                    command(['git','commit','-F',str(message)]);head=git('rev-parse','HEAD')
                    require(git('rev-parse','HEAD^{tree}')==pending['tree'],'Commit tree changed')
                record=dict(status='COMMITTED',phase=phase,checks='PASS',commit=head,
                    published_files=checked,inputs=pending['inputs'],time=time.time(),full_plan_complete=False)
                write(receipt_path,record)
            command(['git','push','origin','main'])
            remote=git('ls-remote','origin','refs/heads/main').split()[0]
            require(remote==record['commit'],'Remote commit not confirmed')
            record.update(status='PUSHED',remote_commit=remote,verified_time=time.time())
            write(receipt_path,record);print('PUSHED',record['commit'],flush=True);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('start','results'),required=True);a=p.parse_args()
    try:publish(a.phase)
    except BaseException as e:
        write(OUT/f'{a.phase}_publication_failure.json',dict(status='FAILED_REQUIRES_REVIEW',
            error=repr(e),time=time.time(),preserve_index_and_worktree=True));raise
