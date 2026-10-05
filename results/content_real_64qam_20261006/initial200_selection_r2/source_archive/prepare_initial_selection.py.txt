"""Prepare closed H initial200 selection evidence; no Git, SSH or job launch.

All live/failed predecessor states are verified before exporting immutable public
text copies. A failed selection owner needs a separately implemented future
closeout contract; a worker completion alone is never accepted here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

MARKER='<!-- H_INITIAL200_SELECTION_R2 -->'
ARMS=('H16-R','H16-A','H64-R','H64-A')
SNRS=(13,19)
MAX_BYTES=10_000_000
RAW_FIELDS={'source_tokens','token_values','decoded_bits','raw_payload','payload_bytes','source_rgb',
            'pixel_values','image_pixels','raw_images','model_state_dict','private_path_aliases'}


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def bind(paths):return {str(p):sha(p) for p in paths}
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Immutable evidence changed: '+p)
def pin(row):
    p=Path(row['path']);require(p.is_absolute() and sha(p)==row['sha256'],'Pinned input differs: '+str(p));return p
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def within(path,parent):
    path,parent=Path(path).resolve(),Path(parent).resolve()
    require(path!=parent and parent in path.parents,'Path must be inside intended directory: '+str(path));return path


def no_payload(value):
    if isinstance(value,dict):
        for k,v in value.items():
            require(k.lower() not in RAW_FIELDS,'Raw image/token/payload export prohibited: '+k);no_payload(v)
    elif isinstance(value,list):
        for v in value:no_payload(v)


def verify_selection_closed(a,prior,owner_path,reg_path,launch_path,config_path,state_reader):
    """Only a successful, exited actual selection owner is currently supported."""
    cfg,reg,launch=read(owner_path),read(reg_path),read(launch_path);rsha=sha(reg_path)
    a.validate_config(cfg,reg,sha(owner_path));verify(reg['source_bindings']);verify(reg['input_bindings'])
    require(cfg['registration']==str(reg_path) and reg['allowed_stage_ids']==['freeze']
            and len(cfg['stages'])==1,'Selection scope differs')
    stage=cfg['stages'][0];require(stage['resource']=='cpu' and len(stage['jobs'])==1,'Single CPU selection required')
    job=stage['jobs'][0];argv=job['argv']
    require(argv.count('--config')==1 and argv[argv.index('--config')+1]==str(config_path)
            and argv.count('--stage')==1 and argv[argv.index('--stage')+1]=='run','Wrong selection command')
    ident=launch['identity'];expected=[launch['argv'][0],'-B',str(Path(a.__file__).resolve()),'--config',str(owner_path)]
    require(ident['argv']==launch['argv']==expected and launch['registration_sha256']==rsha
            and launch['owner_config_sha256']==sha(owner_path),'Selection owner launch changed')
    base=Path(cfg['owner_out']);identity=read(base/'owner_identity.json')
    require(a.same_identity(identity,ident) and identity['registration_sha256']==rsha
            and identity['config_sha256']==sha(owner_path),'Selection owner identity changed')
    require(prior.exited(ident,state_reader),'Selection owner still live or unreaped')
    for path in (base/'failure.json',base/'STOP',Path(job['out'])/'failure.json',Path(job['out'])/'STOP',Path(cfg['out'])/'STOP'):
        require(not path.exists(),'Selection failure/STOP requires explicit future closeout: '+str(path))
    closure=prior.verify_batch(a,str(owner_path),str(reg_path),ident['uid'],state_reader)
    require(len(closure['child_identities'])==1 and prior.exited(closure['child_identities'][0],state_reader),
            'Selection child must be fully exited')
    cp=Path(job['completion']);done=read(cp)
    require(closure['bindings'].get(str(cp))==sha(cp) and done['status']=='H_INITIAL_TRUE200_WHOLE_SELECTION_COMPLETE'
            and done['registration_sha256']==rsha and done['config_sha256']==sha(config_path)
            and done['selected_count']==8 and done['source_count']==200 and done['measured_frames']==9600
            and done['new_packet_decodes']==0 and done['new_visual_inference']==0 and done['GPU_used'] is False
            and done['original_render_owner_success'] is False and done['partial_reselected'] is False
            and done['development_used'] is False and done['holdout_used'] is False
            and done['full1000_calibration_complete'] is False,'Selection scientific receipt differs')
    verify(done['outputs']);verify(done['source_bindings']);verify(done['input_bindings'])
    return dict(owner=cfg,registration=reg,done=done,completion=cp,closure=closure,
                closed_bindings=dict(closure['bindings'],**bind((owner_path,reg_path,launch_path,config_path,base/'owner_identity.json'))))


def validate_selection_outputs(directory,done,original_shortlist):
    directory=Path(directory);selected=directory/'selected_whole.json';partial=directory/'partial_shortlist_reference.json';summary=directory/'whole_summary.csv'
    for path in (selected,partial,summary):require(done['outputs'].get(str(path))==sha(path),'Selection output not sealed')
    result,ref=read(selected),read(partial)
    require(result['status']=='H_INITIAL_TRUE200_SELECTION_COMPLETE' and result['selected_count']==8
            and result['candidate_count']==16 and result['measured_frames']==9600 and result['source_count']==200
            and result['source_ids']==original_shortlist['source_ids'] and result['development_used'] is False
            and result['full1000_calibration_complete'] is False,'Wrong initial200 selection result')
    winners=result['selected_candidates'];families={(x['arm'],x['snr_db']) for x in winners}
    require(len(winners)==8 and families=={(a,s) for a in ARMS for s in SNRS},'Eight original arm/SNR winners required')
    candidates={(x['candidate_id'],x['snr_db']):x for x in original_shortlist['whole_candidates']}
    for row in winners:
        original=candidates[row['candidate_id'],row['snr_db']]
        require(all(row[k]==original[k] for k in ('arm','q','target_m','nominal_rate','slot')),'Winner changed original policy')
        histogram=row['actual_m_histogram']
        require(histogram==original['actual_m_histogram'] and set(histogram)=={'6','7','8','9'}
                and all(type(n) is int and n>=0 for n in histogram.values()) and sum(histogram.values())==200,
                'Actual200 source-prefix histogram differs from original shortlist')
    require(ref['status']=='FROZEN_PRESCREEN_REFERENCE_ONLY_NOT_RESELECTED' and ref['candidate_count']==6
            and ref['partial_reselected'] is False and ref['actual_partial_calibration_complete'] is False
            and ref['partial_candidates']==original_shortlist['partial_candidates'],'Partial reference must remain unmeasured original shortlist')
    require(sha(ref['original_shortlist_path'])==ref['original_shortlist_sha256'],'Partial shortlist source changed')
    with summary.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==16 and {(x['candidate_id'],int(x['snr_db'])) for x in rows}==set(candidates),'Summary omitted original candidates')
    return result,ref


def evidence_envelope(source,expected,*,original_path=None):
    """Make an immutable JSON envelope for closed logs/markers/archived state.

    Raw JSON content remains structured so the sanitizer still checks sensitive
    keys. Text is UTF8, never loaded as code, images, tensors or binary payload.
    """
    source=Path(source);require(sha(source)==expected,'Archived byte SHA changed')
    raw=source.read_bytes();require(len(raw)<MAX_BYTES and b'\0' not in raw,'Oversized/binary archive prohibited')
    text=raw.decode('utf-8-sig');value=json.loads(text) if source.suffix.lower()=='.json' else text
    no_payload(value)
    return dict(status='IMMUTABLE_CLOSED_ENGINEERING_EVIDENCE_COPY',original_path=str(original_path or source),
        preserved_source_path=str(source),original_raw_sha256=expected,original_raw_bytes=len(raw),
        closed_evidence=True,content_encoding='JSON_VALUE' if source.suffix.lower()=='.json' else 'UTF8_TEXT',content=value)


def report_section(result,link):
    lines=[MARKER,'','## 初始200源选策与独立收尾认证','',
        '已完成200张固定校准源图 × 16个整尺度候选 × 3个噪声，共9600帧实际接收重建，并按登记的逐源平均PSNR规则选出8个整尺度策略。下表使用同一批200源选择并报告，不能据此作配对优势或最终泛化结论；它不是development结果，也不表示完整1000源校准或H系统优势。','',
        '| SNR | 臂 | 目标尺度 | 实际源尺度（图数） | 调制 | 码率 | 初始200源平均PSNR |','| --- | --- | --- | --- | --- | --- | --- |']
    for row in sorted(result['selected_candidates'],key=lambda x:(x['snr_db'],ARMS.index(x['arm']))):
        histogram=row['actual_m_histogram'];require(sum(histogram.values())==200,'Source histogram must cover200 images')
        prefix='，'.join(f'm{k}: {histogram[k]}' for k in sorted(histogram,key=int) if histogram[k])
        lines.append(f"| {row['snr_db']} dB | {row['arm']} | m{row['target_m']} | {prefix} | {2**row['q']}QAM | {row['nominal_rate']} | {row['initial_true200_mean_psnr_db']:.4f} dB |")
    lines.extend(['',
        '实际源尺度是发送端按登记容量回退后的前缀，按200张源图计数；目标m9不表示每张图都能发送到m9，需结合m8/m9的实际分布解释算术编码的收益。该列不是接收CRC成功率，PSNR仍由实际接收重建计算。','',
        'GPU worker已成功退出并封存全部图像和分数；原render owner在观察worker退出时遇到竞态而失败，原owner仍没有成功完成凭证。后续独立只读认证核验了原失败、worker退出0、全部输出SHA和69960次既有计费，并精确归档释放两个已登记STOP。原失败与22件归档保留，遗漏的局部STOP另有补充证据；没有补写旧owner成功记录，也没有重跑PHY或图像。','',
        '本次CPU选策owner及其worker已成功完成并退出。6个partial候选仅保留原预筛引用，尚未在该阶段完成partial校准。完整1000源校准、统一指标、development、主系统校准及C-REAL仍待各自登记执行；本次没有自动启动这些阶段。','',
        f'[完整选策与收尾证据]({link})包含原始SHA与公开脱敏副本SHA。早先提交445a07e保留为当时阶段快照。',''])
    return '\n'.join(lines)


def compact_selected(result,source):
    value={k:v for k,v in result.items() if k!='per_source_evidence'}
    value['public_export']=dict(scope='SELECTION_SUMMARY_FULL_FRAME_EVIDENCE_RETAINED_PRIVATELY',
        original_file_path=str(source),original_file_sha256=sha(source),
        omitted_fields=['per_source_evidence'],omitted_source_candidate_groups=len(result['per_source_evidence']),
        measured_frames=result['measured_frames'],original_file_modified=False)
    return value


def prepare(request_path):
    request_path=Path(request_path).resolve();request=read(request_path)
    require(request['schema']=='H_INITIAL_SELECTION_PUBLICATION_REQUEST_V1','Wrong publication request')
    require(sys.platform.startswith('linux'),'Actual preparation requires Linux exit evidence')
    root=Path(request['root']).resolve();work=Path(request['private_out']).resolve();public=Path(request['public_out']).resolve()
    require(not work.exists() and not public.exists(),'Existing publication attempt: preserve and inspect')
    within(work,root/'outputs/CONTENT-REAL-64QAM-20261006');within(public,root/'results/content_real_64qam_20261006')
    paths={k:pin(request[k]) for k in ('selection_owner_config','selection_registration','selection_launch','selection_config','sanitizer',
                                     'original_receipt_validation')}
    cfg,reg=read(paths['selection_config']),read(paths['selection_registration']);verify(reg['source_bindings']);verify(reg['input_bindings'])
    require(reg['input_bindings'].get(str(paths['selection_config']))==sha(paths['selection_config']),'Selection config unbound')
    selection_request=read(cfg['request_path']);require(reg['input_bindings'].get(cfg['request_path'])==sha(cfg['request_path']),'Selection request unbound')
    old=read(pin(selection_request['render_config']))
    for name in ('owner_module','wait_module'):
        require(reg['source_bindings'].get(old[name])==sha(old[name]),'Original verifier source is not bound: '+name)
    owner=module(old['owner_module'],'publication_closed_owner')
    prior=module(old['wait_module'],'publication_closed_prior')
    selector_path=owner.command_entry(read(paths['selection_owner_config'])['stages'][0]['jobs'][0]['argv'])
    require(reg['source_bindings'].get(str(selector_path))==sha(selector_path)
            and selector_path.name=='register_initial_selection_r2.py','Expected immutable R2 selection helper')
    selector=module(selector_path,'publication_registered_r2_selector')
    ctx=selector.predecessor(selection_request)  # Rechecks certified render/CPU/STOP bindings, no science execution.
    audited=verify_selection_closed(owner,prior,paths['selection_owner_config'],paths['selection_registration'],paths['selection_launch'],
                                     paths['selection_config'],owner.raw_process_state)
    selector.check_budget(audited['done']['budget']);require(audited['done']['budget']==ctx['before'],'Selection changed shared ledger')
    result,partial=validate_selection_outputs(cfg['out'],audited['done'],ctx['shortlist'])
    close_reg_path=ctx['paths']['closeout_registration'];close_reg=read(close_reg_path);close_request_path=pin(close_reg['request']);close_request=read(close_request_path)
    archive_path=pin(close_request['failure_archive']);archive=read(archive_path)
    require(archive['status']=='SECOND_FAILURE_PRESERVED_STOP_NOT_RELEASED' and len(archive['files'])==22,'Original22 archive differs')
    supplement_path=pin(close_request['renderer_stop_evidence']);supplement=read(supplement_path)
    diagnostic=read(paths['original_receipt_validation'])
    require(diagnostic['status']=='ORIGINAL_RECEIPT_VALIDATION_PASS_DIAGNOSTIC_ONLY'
            and diagnostic['diagnosis_sha256']==close_request['diagnosis']['sha256']
            and diagnostic['render_completion_sha256']==sha(ctx['closure']['render_completion'])
            and diagnostic['owner_source_sha256']==sha(old['owner_module'])
            and diagnostic['original_owner_success'] is False and diagnostic['selection_authorized'] is False
            and diagnostic['new_GPU'] is False and diagnostic['new_packet_decodes']==0,
            'Original validator diagnostic does not match this independently certified failure')
    main_report=root/'reports/content_real_64qam_20261006.md';old_report=main_report.read_bytes()
    require(MARKER not in old_report.decode('utf-8-sig'),'Report section already exists; do not duplicate or overwrite')
    sanitizer=module(paths['sanitizer'],'initial_selection_public_sanitizer')
    work.mkdir()
    try:
        entries=[];raw_evidence={};wrapped=work/'closed_evidence'
        def add(source,target,role='receipt'):
            source=Path(source);require(source.suffix.lower() in ('.json','.csv','.md','.py','.txt'),'Explicit text-only file required')
            if source.suffix.lower()=='.json':no_payload(read(source))
            entries.append(dict(source=str(source),source_sha256=sha(source),target=target,role=role));raw_evidence[str(source)]=sha(source)
        def closed(source,target,expected=None,original=None):
            source=Path(source);expected=expected or sha(source)
            # Immutable aliases allow previously closed launch/status/log files,
            # while the sanitizer continues refusing live dynamic filenames.
            envelope=evidence_envelope(source,expected,original_path=original)
            alias=wrapped/(hashlib.sha256(target.encode()).hexdigest()+'.json');save(alias,envelope)
            add(alias,target);raw_evidence[str(source)]=expected
        def qualification(path,prefix):
            path=Path(path);value=read(path);add(path,prefix+'/qualification.json')
            for i,row in enumerate(value['results']):
                record=row.get('log')
                if record is not None:
                    p=pin(record);closed(p,f'{prefix}/qualification_log_{i:02d}.json',record['sha256'])
        for index,(original,row) in enumerate(sorted(archive['files'].items())):
            closed(row['copy'],f'render_original_archive/{index:02d}_{Path(original).name}.json',row['sha256'],original)
        add(archive_path,'render_failure/failure_archive_receipt.json')
        add(pin(close_request['diagnosis']),'render_failure/diagnosis.json')
        add(paths['original_receipt_validation'],'render_failure/original_receipt_validation.json')
        add(supplement_path,'render_failure/renderer_stop_evidence.json')
        closed(supplement['preserved_path'],'render_failure/supplemental_renderer_STOP.json',supplement['sha256'],supplement['original_path'])
        close_dir=close_reg_path.parent
        for name in ('execution_registration.json','registration_completion.json','completion.json','release_receipt.json','closeout_completion.json'):
            add(close_dir/name,'closeout/'+name)
        add(close_request_path,'closeout/request.json','configuration')
        qualification(pin(close_request['prepared_qualification']),'closeout')
        release=read(close_dir/'release_receipt.json')
        for index,row in enumerate(release['released_markers']):
            closed(row['preserved_path'],f'closeout/released_STOP_{index}.json',row['sha256'],row['original_path'])
            if (close_dir/f'stop_release_{index}.json').exists():add(close_dir/f'stop_release_{index}.json',f'closeout/stop_release_{index}.json')
        selection_dir=Path(audited['owner']['owner_out'])
        for name in ('execution_registration.json','registration_completion.json','selection_config.json','owner_config.json','predecessor_closure.json','completion.json'):
            add(selection_dir/name,'selection_execution/'+name,'configuration' if name.endswith('_config.json') else 'receipt')
        add(Path(cfg['request_path']),'selection_execution/request.json','configuration')
        qualification(pin(selection_request['prepared_qualification']),'selection_execution')
        for name in ('launch.json','owner_identity.json'):
            closed(selection_dir/name,'selection_execution/closed_'+name)
        stage=selection_dir/'stages/freeze';add(stage/'completion.json','selection_execution/freeze_completion.json')
        worker=stage/'workers/initial_selection'
        for name in ('launch.json','exit_receipt.json','worker.log'):
            closed(worker/name,'selection_execution/closed_worker_'+name+'.json')
        for name in ('completion.json','whole_summary.csv','partial_shortlist_reference.json'):
            add(Path(cfg['out'])/name,'selection/'+name,'summary' if name.endswith('.csv') else 'receipt')
        selected_source=Path(cfg['out'])/'selected_whole.json';compact_path=work/'SELECTED_SUMMARY.json'
        save(compact_path,compact_selected(result,selected_source));raw_evidence[str(selected_source)]=sha(selected_source)
        add(compact_path,'selection/selected_whole_summary.json','summary')
        # Explicit new sources only; importing or following output maps is prohibited.
        for record in request.get('source_snapshots',[]):
            p=pin(record);require(p.suffix=='.py','Source snapshot must be Python text');add(p,'source_archive/'+p.name+'.txt','source')
        add(Path(__file__).resolve(),'source_archive/'+Path(__file__).name+'.txt','source')
        section=report_section(result,str(Path('..')/public.relative_to(root)/'README.md').replace('\\','/'))
        readme=work/'README.md';readme.write_text(report_section(result,'selection/selected_whole_summary.json').replace(
            MARKER,'# 初始200源整尺度选策交付'),encoding='utf-8')
        add(readme,'README.md','documentation')
        receipt=work/'VERIFIED_SCOPE.json';save(receipt,dict(status='INITIAL200_SELECTION_AND_CLOSEOUT_VERIFIED_FOR_EXPORT',
            original_render_owner_success=False,original_render_owner_completion_exists=False,
            certified_render_scientific_outputs_complete=True,selection_owner_success=True,selection_owner_and_worker_exited=True,
            source_count=200,frame_count=9600,whole_winners=8,partial_reselected=False,
            budget=ctx['before'],input_bindings=audited['closed_bindings'],new_packet_decodes=0,new_visual_inference=0,
            GPU_used=False,full1000_calibration_complete=False,H_full_delivery_claimed=False,C_started=False,
            historical_snapshot_commit='445a07ee88f24417de65f9b8d545505d4884d668',published=False))
        add(receipt,'VERIFIED_SCOPE.json')
        manifest=work/'export_manifest.json';save(manifest,dict(schema='EXPLICIT_PUBLIC_TEXT_EXPORT_V1',files=entries))
        sanitizer.prepare(root,manifest,public,work/'private_export_receipt.json')
        require(main_report.read_bytes()==old_report,'Report changed while preparing; do not overwrite another update')
        updated=old_report.decode('utf-8-sig').rstrip()+'\n\n'+section
        require(not any(line.endswith((' ','\t')) for line in section.splitlines()),'New report section has trailing whitespace')
        require(sanitizer.Sanitizer(root).text(section)==section,'Private path leaked into report section')
        main_report.write_text(updated,encoding='utf-8',newline='\n')
        files=sorted([str(p.relative_to(root)).replace('\\','/') for p in public.rglob('*') if p.is_file()]+[str(main_report.relative_to(root)).replace('\\','/')])
        require(all((root/p).stat().st_size<MAX_BYTES for p in files),'Public file is at least10MB')
        verify(raw_evidence)
        selector.check_budget(owner.budget_snapshot(old['ledger'],sha(old['budget_registration']),selector.PHASES,quiescent=True))
        # A final read-only closure check rejects a late/new failure or STOP.
        selector.predecessor(selection_request)
        verify_selection_closed(owner,prior,paths['selection_owner_config'],paths['selection_registration'],paths['selection_launch'],
                                paths['selection_config'],owner.raw_process_state)
        stage_manifest=work/'publication_stage_manifest.json'
        save(stage_manifest,dict(status='PREPARED_FOR_MANUAL_REVIEW_CHECKS_AND_PUSH',root=str(root),git_add_paths=files,
            files={p:sha(root/p) for p in files},report_original_sha256=hashlib.sha256(old_report).hexdigest(),
            raw_evidence_bindings=raw_evidence,request_sha256=sha(request_path),sanitizer_sha256=sha(paths['sanitizer']),
            source_sha256=sha(__file__),Git_invoked=False,committed=False,pushed=False,other_status_documents_modified=False,
            required_next='Review exact stage list; repository manifest/checks; explicit normal commit and push with remote verification'))
        done=dict(status='INITIAL200_SELECTION_PUBLICATION_PREPARED_NOT_PUSHED',outputs=bind((stage_manifest,manifest,receipt)),
            public_files={p:sha(root/p) for p in files},new_packet_decodes=0,GPU_used=False,Git_invoked=False,pushed=False)
        save(work/'completion.json',done);return done
    except BaseException as error:
        if not (work/'failure.json').exists():save(work/'failure.json',dict(status='FAILED_PRESERVE_NO_AUTOMATIC_RETRY',
            error=repr(error),traceback=traceback.format_exc(),Git_invoked=False,pushed=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    result=prepare(parser.parse_args().config);print(json.dumps({'status':result['status'],'public_files':len(result['public_files'])}))


if __name__=='__main__':main()
