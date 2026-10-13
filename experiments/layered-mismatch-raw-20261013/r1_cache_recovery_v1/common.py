"""Strict metadata for one explicitly authorized cache-initialization recovery."""
from __future__ import annotations
import hashlib, json, os, stat, time
from pathlib import Path

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/layered-mismatch-raw-20261013'
OLD=E/'r1_execution_resource_v2'
HERE=E/'r1_cache_recovery_v1'
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013')
ANCESTOR=BASE/'r1_execution_v1'
RUN=BASE/'r1_cache_recovery_v1'
OUT=R/'results/layered_mismatch_raw_20261013/r1_cache_recovery_v1'
DISPATCH=BASE/'r1_cache_recovery_dispatch_v1'
OLD_OUT=R/'results/layered_mismatch_raw_20261013/r1_execution_v1'
ORIGINAL=dict(path=str(OLD_OUT/'registered/registration.json'),sha256='70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47')
ORIGINAL_VISUAL=dict(path=str(OLD_OUT/'registered/visual_registration.json'),sha256='bdec4911370fa14cd4430dbb4cea8da7e76486242ed95c2a3629d6ff0f744828')
DEADLINE=1791954813.2221007
OLD_SHA={'shared.py':'6f80228bf4af97685172d0dff88feee865a9b01f413272e03c0238c7ac31fbe6',
 'owner.py':'1cc1ba2a64ad73303deb992438da6fae669836228894cec0613b0e9570608ab6',
 'receiver_visual.py':'55cd58863018e30ce92e501148c225fd4b412ac1c99c274d3ea0b63f964e2a9f',
 'score_selection.py':'1bd5d9b016ed9d4bd8044acf67942a2fc06bd0e23450fda5d04039543ff48947',
 'dispatch_once.py':'f4b635edc808f4337133b09eaace2e72d34df2698f8490db8e95c3d8214edd18'}
RECOVERY_SCHEMA='R1_CACHE_RECOVERY_V1'
FAILURE_AUDIT=dict(path=str(R/'results/layered_mismatch_raw_20261013/r1_cache_failure_audit_v1/failure_audit.json'),
 sha256='daad4d721815607b9b9186ed33ded6b778c9a91b91d169afbfb5000f15add7d4')
STAGES=('visual','score','selection')
PROBE_SOURCE=dict(path=str(E/'r1_cache_recovery_probe_v2/probe.py'),sha256='aa04755bbaa867e0f3ec3d694242712348104ed9634edeffec3ec81245529c25')
PROBE_OUT=BASE/'r1_cache_recovery_probe_v2'
PROBE_WAIT=BASE/'r1_cache_recovery_probe_dispatch_v2/actual_wait.json'
PROBE_COUNTERS={'tensor_requests','model_constructor_requests','weight_load_requests','cuda_init_requests','channel_import_requests'}

def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def regular(path):
    p=Path(path);require(p.is_absolute() and '..' not in p.parts,'Canonical absolute path required')
    for q in (*reversed(p.parents),p):require(not stat.S_ISLNK(q.lstat().st_mode),'Symlink forbidden')
    require(stat.S_ISREG(p.lstat().st_mode),'Regular file required');return p

def read(path):return json.loads(regular(path).read_text(encoding='utf-8-sig'))
def pin(path):return dict(path=str(regular(path)),sha256=sha(path),bytes=Path(path).stat().st_size)
def same(a,b):return a['path']==b['path'] and a['sha256']==b['sha256']
def checked(value):
    p=regular(value['path']);require(sha(p)==value['sha256'],'SHA changed: '+str(p))
    if 'bytes' in value:require(type(value['bytes']) is int and p.stat().st_size==value['bytes'],'Byte count differs')
    return p
def readpin(value):return read(checked(value))
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)

def identity(pid):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text();tail=text[text.rfind(')')+2:].split()
        return dict(pid=int(pid),start_ticks=int(tail[19]),state=tail[0])
    except FileNotFoundError:return None
def gone(value,probe=identity):
    require(type(value.get('pid')) is int and type(value.get('start_ticks')) is int,'PID/start integers required')
    now=probe(value['pid']);require(now is None or now['start_ticks']!=value['start_ticks'],'Original identity still exists, including zombie')
def actual(wait,launch,code):
    require(all(type(launch.get(k)) is int and launch[k]>0 for k in ('pid','start_ticks')),'Positive PID/start integer identity required')
    require(type(wait.get('pid')) is int and wait['pid']>0,'Positive waited PID integer required')
    require(wait.get('actual_wait') is True and type(wait.get('returncode')) is int and wait['returncode']==code
            and wait.get('pid')==launch['pid'] and not any(wait.get(k,False) for k in ('timeout','timed_out','terminated','killed','stopped')),
            'Actual known child exit required')
    if 'start_ticks' in wait:require(type(wait['start_ticks']) is int and wait['start_ticks']==launch['start_ticks'],'Wait generation differs')

def validate_probe(value):
    require(same(value['probe_source'],PROBE_SOURCE),'Different reviewed zero-model probe source')
    checked(PROBE_SOURCE)
    require(value['zero_model_probe']['path']==str(PROBE_OUT/'completion.json')
        and value['zero_model_probe_actual_wait']['path']==str(PROBE_WAIT),'Different actual probe receipt locations')
    done=readpin(value['zero_model_probe']);wait=readpin(value['zero_model_probe_actual_wait'])
    require(done['schema']=='R1_RX_CACHE_ZERO_MODEL_QUALIFICATION_V2' and done['status']=='PASS_ZERO_MODEL_CACHE_IMPORT_FLAGS'
        and same(done['probe'],PROBE_SOURCE) and done['source']['path']==str(OLD/'receiver_visual.py')
        and done['source']['sha256']==OLD_SHA['receiver_visual.py']
        and done['bundle']['path']==str(OLD_OUT/'registered/model_bundle.json')
        and done['bundle']['sha256']=='c1d25aaf76fdb36098e225c8dba1157b0d4ea92c9afccb39f348e2121c87a0a0',
        'Probe did not use frozen receiver and bundle')
    require(all(type(done.get(k)) is int and done[k]==0 for k in ('scientific_calls','model_or_channel_calls',
        'qualification_frames','source_truth_reads','model_weight_bytes_read')) and done['actual_scientific_registration_created'] is False
        and done['CUDA_VISIBLE_DEVICES']=='' and done['cuda_initialized_before_flags'] is False and done['cuda_initialized_after_flags'] is False
        and all(done[k]==[] for k in ('blocked','storage_blocked','rejected_requests'))
        and set(done['request_counts'])==PROBE_COUNTERS
        and all(type(n) is int and n==0 for n in done['request_counts'].values()),'Probe has denied or nonzero scientific activity')
    require(wait['actual_wait'] is True and type(wait['returncode']) is int and wait['returncode']==0
        and type(wait['pid']) is int and wait['pid']>0 and wait['CUDA_VISIBLE_DEVICES']==''
        and wait['automatic_retry'] is False and wait['probe_sha256']==PROBE_SOURCE['sha256']
        and not any(wait.get(k,False) for k in ('timeout','timed_out','terminated','killed','stopped')),'Probe real wait not successful')
    require(identity(wait['pid']) is None,'Probe PID still exists; actual qualification must finish')
    require(not (PROBE_OUT/'failure.json').exists(),'Probe failure cannot qualify recovery')
    return done

def frozen_runtime(old):
    import copy
    runtime=copy.deepcopy(old)
    require(runtime['deadline_unix']==DEADLINE and runtime['root']==str(ANCESTOR),'Wrong original finite runtime')
    runtime['root']=str(RUN)
    for stage in STAGES:runtime[stage+'_directory']=str(RUN/stage)
    # Keep original STOP, shared GPU/CPU, numerical environment, and total wall budget.
    return runtime

def closed_ancestors():
    """Only immutable small receipts, no arrays, SQLite or quality computation."""
    original=readpin(ORIGINAL);owner=read(ANCESTOR/'owner_launch.json')
    require(same(owner['registration'],ORIGINAL),'Wrong original owner')
    parent=read(BASE/'r1_resource_dispatch_v2/launch.json')
    waited=read(BASE/'r1_resource_dispatch_v2/actual_wait.json')
    actual(waited,owner['process'],1)
    require(same(waited['registration'],ORIGINAL),'Old owner wait registration differs')
    require(waited['identity_at_launch']['pid']==owner['process']['pid'] and waited['identity_at_launch']['start_ticks']==owner['process']['start_ticks'],
            'Old owner generation differs')
    gone(owner['process']);gone(parent['process']);stages=[]
    for stage in ('phy','projection','visual'):
        launch=read(ANCESTOR/(stage+'_launch.json'));wait=read(ANCESTOR/stage/'actual_wait.json')
        require(launch['stage']==stage and same(launch['registration'],ORIGINAL) and wait==read(ANCESTOR/(stage+'_actual_wait.json')),
                'Original dispatch/wait differs')
        actual(wait,launch['process'],1 if stage=='visual' else 0);gone(launch['process'])
        if stage!='visual':require(not (ANCESTOR/stage/'failure.json').exists(),'Successful ancestor has a failure marker')
        stages.append(dict(stage=stage,launch=pin(ANCESTOR/(stage+'_launch.json')),actual_wait=pin(ANCESTOR/stage/'actual_wait.json')))
    phy=read(ANCESTOR/'phy/completion.json');projection=read(ANCESTOR/'projection/completion.json')
    require(phy['status']=='COMPLETE_R1_PHY_ONLY' and same(phy['registration'],original['phy_registration'])
            and phy['ledger']['physical_complete']==phy['ledger']['physical_reserved']==32400 and phy['ledger']['unresolved_packets']==0
            and phy['ledger']['packet_decodes_complete']==phy['ledger']['packet_decodes_reserved']<=108000,'Inherited PHY not complete')
    require(projection['status']=='COMPLETE_R1_RX_PROJECTION' and same(projection['registration'],ORIGINAL)
            and projection['logical_count']==43200 and projection['scientific_calls']==0,'Inherited projection not complete')
    require(not (ANCESTOR/'score').exists() and not (ANCESTOR/'selection').exists(),'Original successor unexpectedly attempted')
    require((ANCESTOR/'visual/failure.json').is_file() and (ANCESTOR/'owner_failure.json').is_file(),'Original failure must be preserved')
    return dict(original_registration=ORIGINAL,original_owner_wait=pin(BASE/'r1_resource_dispatch_v2/actual_wait.json'),
        original_owner_failure=pin(ANCESTOR/'owner_failure.json'),failed_visual=pin(ANCESTOR/'visual/failure.json'),stages=stages,
        inherited_phy=pin(ANCESTOR/'phy/completion.json'),inherited_projection=pin(ANCESTOR/'projection/completion.json'))

def validate_authorization(value):
    require(value['schema']=='R1_CACHE_RECOVERY_AUTHORIZATION_V1' and value['approved'] is True
            and value['original_registration']==ORIGINAL and value['deadline_unix']==DEADLINE
            and value['allowed_stages']==list(STAGES) and value['old_forward_calls']==0
            and value['old_model_load_completed']==0 and value['old_failed_model_load_reservations']==1
            and value['additional_scientific_budget']==0 and value['automatic_retry'] is False,
            'Root zero-forward failure audit and finite recovery authorization required')
    require(same(value['failure_audit'],FAILURE_AUDIT),'Different diagnosed failure')
    audit=readpin(FAILURE_AUDIT)
    require(audit['schema']=='R1_CLOSED_VISUAL_CACHE_FAILURE_AUDIT_V1' and audit['status']=='DIAGNOSED_NO_AUTOMATIC_RETRY'
        and same(audit['original_registration'],ORIGINAL) and audit['original_deadline_unix']==DEADLINE
        and audit['failed_calls']==[dict(id=1,kind='model_load',metadata='{}',state='FAILED')]
        and audit['extra_failed_initialization_attempts']==1 and audit['completed_stages']==['phy','projection']
        and all(type(audit[k]) is int and audit[k]==0 for k in ('completed_model_loads','completed_VAR_renders','completed_prior_scales',
                     'completed_decoder_forwards','output_images','partial_logical_frames','scientific_calls_in_this_audit')),
        'Failed initialization has nonzero or unproven scientific work')
    require(audit['evidence'] and all(path==p['path'] for path,p in audit['evidence'].items()),'Malformed failure evidence inventory')
    for p in audit['evidence'].values():checked(p)
    require(read(ANCESTOR/'visual/partial_logical_frames.json')==[],'Failed segment has partial scientific frames')
    images=ANCESTOR/'visual/images'
    require(images.is_dir() and not images.is_symlink() and not any(images.iterdir()),'Failed segment unexpectedly has output images')
    validate_probe(value)
    review=readpin(value['independent_review'])
    require(review['status']=='PASS_R1_CACHE_RECOVERY_IMPLEMENTATION_V1' and review['approved'] is True
            and review['code_bindings']==value['code_bindings'],'Independent implementation review/bindings differ')
    for p,digest in value['code_bindings'].items():require(sha(regular(p))==digest,'Reviewed recovery source changed')
    require(value['original_closure']==closed_ancestors(),'Original closed/failed receipts differ')
    return value
