"""Archive frozen timing recovery evidence only; never active v6 outputs.

plan reads metadata only. build requires a pinned normal-push receipt and a
registered, reviewed v6 recovery. It does not start, stop or retry science.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
RUNTIME=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
BASE=ROOT/'experiments/baseline-delivery-20261012'
RESULTS=ROOT/'results/baseline_strength_20261011/timing'
NAME='timing_recovery_20261013_v1'
PACKAGE=TMP/(NAME+'_delivery')
V6_STATIC_NAMES=('registration.json','recovery_authorization.json','coordinator_review.json')
DEPENDENCIES=[
 '../timing_entry_20261012_v1: original six-method timing entry and source/model/native bindings',
 '../formal_delivery_20261012_v1: private Sionna dependency and final baseline results',
 '../final_training_20261012_v1: exact complete P100k/Swin200k model states',
 '../diagnostic_delivery_20261012_v1: earlier diagnostic context and assets',
 '../entropy_main_common500_20261013_v1: separate completed E1/E2 quality package; not timing completion',
 '../input_restore and ../R0_local: original image/token/source seed assets',
 '../../VAR_COMM_MIGRATION_20261008 and ../../VAR_COMM_WCL_INCREMENT_20261009_v1: original tokenizer/VAR/Dc/metric model and source recovery',
 '../../VAR_COMM_LEO_STAGE_20261010: frozen environments and H800 native lineage',
]
FORBIDDEN={'.pt','.pth','.ckpt','.safetensors','.onnx','.pyc','.so','.whl','.tar','.gz'}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(p,value):
    with Path(p).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def pin(p):return dict(path=str(p),sha256=sha(p))


def archive_name(path):
    p=Path(path).resolve()
    require(not p.is_relative_to((TMP/'six_method_timing_v6').resolve()),'Active v6 runtime excluded')
    v6=(RESULTS/'six_method_v6').resolve()
    if p.is_relative_to(v6):require(p.parent==v6 and p.name in V6_STATIC_NAMES,'Active v6 result excluded')
    for prefix,parent in (('repo',ROOT),('runtime',TMP),('legacy_assets',RUNTIME)):
        if p.is_relative_to(parent.resolve()):return prefix+'/'+p.relative_to(parent.resolve()).as_posix()
    raise RuntimeError('Outside explicit project backup roots: '+str(p))


def inventory(release_receipt=None):
    files={};missing=[]
    def add(p,required=True):
        p=Path(p)
        if not p.exists():
            if required:missing.append(str(p))
            return
        require(not p.is_symlink() and p.is_file(),'Only regular original files: '+str(p))
        require(p.suffix not in FORBIDDEN,'Forbidden model/cache/archive payload: '+str(p))
        require(p.stat().st_size < 32<<20,'Unexpected large payload: '+str(p))
        name=archive_name(p);require(name not in files or files[name]==p,'Duplicate archive name');files[name]=p
    def tree(parent):
        if not parent.is_dir():missing.append(str(parent));return
        for base,dirs,names in os.walk(parent,followlinks=False):
            dirs[:]=sorted(n for n in dirs if n not in ('__pycache__','cache','torchinductor_root','.git'))
            for name in sorted(names):
                if name.endswith(('.lock','.pyc','.pending')):continue
                add(Path(base)/name)
    for version in (2,3,4,5,6):tree(BASE/f'timing_revision_v{version}')
    tree(BASE/'timing_postprocess_v1');tree(BASE/'timing_recovery_backup_v1')
    for version in (2,3,4,5):tree(RESULTS/f'six_method_v{version}')
    for name in ('six_method_v5_failure_audit_v1','six_method_v6_preparation_v1','six_method_v5_partial_stop_v1'):
        tree(RESULTS/name)
    for name in V6_STATIC_NAMES:add(RESULTS/'six_method_v6'/name)
    # All v5 top-level journals/resources/model bindings are immutable after exit.
    for p in sorted((TMP/'six_method_timing_v5').iterdir()):
        if p.is_file() and p.suffix in ('.json','.jsonl','.log'):add(p)
    reg=read(RESULTS/'six_method_v5/registration.json')
    for field in ('recovery_evidence_sha256','v3_failure_evidence_sha256','unlaunched_v4_evidence_sha256'):
        for path in reg[field]:add(path)
    static=read(reg['visual_request']['path'])['static']
    for field in ('table','completion'):add(static[field]['path'])
    for name in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md','release_manifest.json'):add(ROOT/name)
    if release_receipt:add(release_receipt)
    return files,missing


def plan(release_receipt=None):
    files,missing=inventory(release_receipt)
    return dict(status='PLAN_ONLY_NO_ARCHIVE_NO_SCIENTIFIC_CALLS',files=len(files),
        bytes=sum(p.stat().st_size for p in files.values()),required_missing=missing,
        destination='D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/'+NAME,
        active_v6_runtime_and_results_excluded=True,complete_six_method_result=False,
        dependencies=DEPENDENCIES,independently_complete_restore=False)


def qualify(release_receipt):
    require(release_receipt is not None,'Normal push receipt required')
    release=read(release_receipt);head=release['head']
    require(release['normal_push'] and release['verified_remote']==head and release['branch']=='main','Unverified normal publication')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==head,'HEAD changed since push')
    v5=RESULTS/'six_method_v5';wait=read(v5/'actual_wait.json')
    require(wait==dict(actual_child_waited=True,exit_code=1,timed_out=False),'Actual v5 failure receipt required')
    journal=read(TMP/'six_method_timing_v5/case_reservations.json')
    require(len(journal['cases'])==577 and all(x['state']=='COMPLETE' for x in journal['cases'][:576]),'Unexpected v5 complete prefix')
    require(journal['cases'][-1]['state']=='RESERVED' and journal['cases'][-1]['actual_packet_decodes']==0,'Failure evidence changed')
    require(journal['packet_decodes']==1152,'Prior packet count changed')
    reg=read(RESULTS/'six_method_v6/registration.json');auth=read(RESULTS/'six_method_v6/recovery_authorization.json')
    require(reg['inherited_complete_frames']==576 and reg['remaining_complete_frames']==1152,'Wrong registered recovery scope')
    require(reg['known_fresh_visual_attempts_at_success']==1729 and reg['measured_frame_cap']==864,'Recovery accounting changed')
    require(auth['status']=='REVIEWED_V6_RECOVERY_AUTHORIZED' and auth['review_evidence'],'Review authorization missing')
    for item in (reg['prior_journal'],reg['recovery_authorization']):require(sha(item['path'])==item['sha256'],'Recovery pin differs')
    for field in ('v5_failure_evidence_sha256','recovery_evidence_sha256','v3_failure_evidence_sha256','unlaunched_v4_evidence_sha256'):
        for path,digest in reg[field].items():require(sha(path)==digest,'Historical evidence changed: '+path)
    static=reg['actual_received_static_family_support']['binding']
    for field in ('table','completion'):
        item=static[field];require(sha(item['path'])==item['sha256']==reg['input_sha256'][item['path']],'Static binding changed')
    return dict(normal_push_receipt=pin(release_receipt),git_head=head,
        v6_registration=pin(RESULTS/'six_method_v6/registration.json'),v6_authorization=pin(RESULTS/'six_method_v6/recovery_authorization.json'),
        v5_actual_wait=pin(v5/'actual_wait.json'),v5_journal=pin(TMP/'six_method_timing_v5/case_reservations.json'),
        inherited_complete=576,failed_reserved=1,remaining_registered=1152,
        complete_six_method_result=False,active_v6_outputs_included=False)


def build(release_receipt):
    closure=qualify(release_receipt);files,missing=inventory(release_receipt)
    require(not missing,'Missing required evidence: '+repr(missing));require(not PACKAGE.exists(),'Existing backup attempt preserved; inspect before retry')
    # All newly archived source code must be present in the verified publication.
    for name,p in files.items():
        if name.startswith('repo/experiments/'):
            data=subprocess.check_output(['git','show',closure['git_head']+':'+name[5:]],cwd=ROOT)
            require(hashlib.sha256(data).hexdigest()==sha(p),'Published source differs: '+name)
    PACKAGE.mkdir(parents=True)
    try:
        rows=[]
        for name,p in sorted(files.items()):
            before=p.stat();digest=sha(p);after=p.stat()
            require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Payload changed while hashing')
            rows.append(dict(path=name,original_path=str(p),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=digest))
        manifest=dict(schema='TIMING_RECOVERY_INCREMENT_V1',scope='FROZEN_RECOVERY_ENTRY_AND_V5_FAILURE',
            created_unix=time.time(),files=rows,file_count=len(rows),payload_bytes=sum(x['bytes'] for x in rows),
            closure=closure,dependencies=DEPENDENCIES,independently_complete_restore=False,
            active_v6_outputs_included=False,complete_six_method_result=False,new_scientific_calls=0,deleted_files=0)
        write(PACKAGE/'backup_manifest.json',manifest)
        (PACKAGE/'RESTORE.md').write_text('''# Frozen six-method timing recovery increment

This package preserves the reviewed recovery entry and immutable failure
evidence. It is not a complete six-method result or an independently complete
recovery package. The v5 journal contains 576 completed calls and one incomplete
entropy warmup. The registered v6 continuation retains those 576 and executes
only 1152 remaining complete calls. Final v6 timing data are deliberately absent
and will require a separate verified result increment after actual closure.

Restore repo/ under the repository, runtime/ under the original personal timing
TMP root, and legacy_assets/ under the legacy runtime root using the manifest's
original paths and SHA values. Preserve all older recovery packages listed in
the manifest; no checkpoint, original image dataset or full environment is
duplicated here. Two original frozen STATIC CDF files are included because the
public entropy receiver domain requires them. They are not newly fitted.

Do not replay v5's completed calls, overwrite its failed journal, automatically
launch v6 after migration, or treat CPU review as numerical timing completion.
Audit path relocation and the actual latest v6 state before any continuation.
The backup performs no deletion and no scientific calls.
''',encoding='utf8')
        archive=PACKAGE/(NAME+'.tar.gz')
        with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz',compresslevel=1) as tar:
            for row in rows:
                p=Path(row['original_path']);s=p.stat()
                require((s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']),'Payload changed before archive')
                tar.add(p,arcname=row['path'],recursive=False);require(sha(p)==row['sha256'],'Payload changed during archive')
            for name in ('backup_manifest.json','RESTORE.md'):tar.add(PACKAGE/name,arcname=name,recursive=False)
        for row in rows:require(sha(row['original_path'])==row['sha256'],'Frozen payload changed after archive')
        receipt=dict(status='REMOTE_TIMING_RECOVERY_INCREMENT_CREATED_LOCAL_VERIFICATION_PENDING',archive=str(archive),
            bytes=archive.stat().st_size,sha256=sha(archive),payload_files=len(rows),payload_bytes=manifest['payload_bytes'],
            manifest_sha256=sha(PACKAGE/'backup_manifest.json'),restore_sha256=sha(PACKAGE/'RESTORE.md'),
            independently_complete_restore=False,complete_six_method_result=False,active_v6_outputs_included=False,
            new_scientific_calls=0,deleted_files=0)
        write(PACKAGE/'backup_archive_receipt.json',receipt);return receipt
    except BaseException:
        write(PACKAGE/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('plan','build'))
    p.add_argument('--release-receipt',type=Path);a=p.parse_args()
    print(json.dumps(plan(a.release_receipt) if a.command=='plan' else build(a.release_receipt),indent=2))
