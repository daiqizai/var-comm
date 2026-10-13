"""Read-only inventory and finite EMAIN E1/E2 incremental recovery archive.

plan is safe while science runs: it reads names and sizes only. build requires
actual scoring/statistics/plot closure and stopped owned science/CPU processes.
No old models, original image/token data, environments or cache directories are
copied; these remain explicit dependencies of this incremental package.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
ART=TMP/'entropy_main_common500_20261013_v1'
OUT=ROOT/'results/entropy_main_common500_20261013'
ENTRY=ROOT/'experiments/entropy-main-common500-20261013'
PACKAGE=TMP/'entropy_main_common500_delivery_20261013_v1'
NAME='entropy_main_common500_20261013_v1'
DEPENDENCIES=[
 '../final_training_20261012_v1: exact complete P100k/Swin200k checkpoints and original500 statistics dependencies',
 '../formal_delivery_20261012_v1 and ../diagnostic_delivery_20261012_v1: final strong-reference scores/statistics and prior registered source',
 '../timing_entry_20261012_v1: existing six-method timing entry/private source bindings; actual new timing will be a separate increment',
 '../input_restore and ../R0_local: original common500 RGB/token archives and fixed baseline source assets',
 '../../VAR_COMM_MIGRATION_20261008 and ../../VAR_COMM_WCL_INCREMENT_20261009_v1: original VAR/tokenizer/Dc/four-metric model assets and historical source/result recovery',
 '../../VAR_COMM_LEO_STAGE_20261010: frozen_visual_py310/frozen_ldpc_py311 environments, native source/model seed and H800 lineage',
]
FORBIDDEN_SUFFIXES={'.pt','.pth','.ckpt','.safetensors','.onnx','.pyc'}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):result.update(block)
    return result.hexdigest()


def verify_binding(entry):
    path=Path(entry['path']);require(sha(path)==entry['sha256'],'Binding changed: '+str(path));return path


def inventory():
    files={};excluded=[]
    for parent,label in ((ENTRY,'repo'),(OUT,'repo'),(ART,'runtime')):
        require(parent.is_dir(),'Missing task root: '+str(parent))
        for base,dirs,names in os.walk(parent,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','cache','.git'))
            for name in sorted(names):
                p=Path(base)/name
                if p.suffix=='.lock' or p.suffix=='.pyc' or name.endswith('.pending'):
                    excluded.append(str(p));continue
                require(not p.is_symlink(),'Do not follow a symlink while backing up: '+str(p))
                require(p.is_file(),'Nonregular file: '+str(p))
                require(p.suffix not in FORBIDDEN_SUFFIXES,'Model/environment duplication forbidden: '+str(p))
                origin=ROOT if label=='repo' else ART
                require(p.resolve().is_relative_to(origin.resolve()),'Escaping task archive root')
                archive_name=label+'/'+p.relative_to(origin).as_posix()
                require(archive_name not in files,'Duplicate archive name')
                files[archive_name]=p
    for name in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md','release_manifest.json'):
        p=ROOT/name;require(p.is_file(),'Missing root publication context');files['repo/'+name]=p
    return files,excluded


def plan():
    files,excluded=inventory();groups=defaultdict(lambda:dict(files=0,bytes=0))
    for name,p in files.items():
        key='/'.join(name.split('/')[:2]) if name.startswith('runtime/') else 'repo'
        groups[key]['files']+=1;groups[key]['bytes']+=p.stat().st_size
    figure_npz_max=4500*(3*256*256*4+4096)
    result=dict(status='PLAN_ONLY_NO_ARCHIVE_NO_SCIENTIFIC_CALLS',observed_unix=time.time(),
                scope='E1_E2_quality_and_source_length_delivery; E3 timing separate increment',
                current_payload_files=len(files),current_payload_bytes=sum(p.stat().st_size for p in files.values()),
                current_categories=dict(sorted(groups.items())),maximum_registered_unique_reconstruction_arrays=4500,
                full_unique_image_payload_estimate_bytes=figure_npz_max,
                size_note='Current size is a live metadata snapshot, not the final package size. If all 4500 float32 RGB arrays are unique, their uncompressed payload alone is about 3.56 GB. Add actual source/PHY/metric ledgers and figures; gzip size is measured only after closure. Exact-input reuse may reduce stored image count.',
                excluded_lock_cache_bytecode=True,excluded_file_count=len(excluded),old_models_or_dataset_copies=0,
                local_destination='D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/'+NAME,
                dependencies=DEPENDENCIES,independently_complete_restore=False,build_requires_actual_closure=True)
    return result


def no_active_owner(pid):
    proc=Path('/proc')/str(pid)
    if not proc.exists():return
    try:
        state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        argv=(proc/'cmdline').read_bytes().decode(errors='replace')
    except FileNotFoundError:return
    if state=='Z':return
    require(str(ENTRY) not in argv,'Owned EMAIN process still running: '+str(pid))


def qualify():
    score=read(OUT/'scoring_completion.json')
    require(score['status']=='COMPLETE_EMAIN4500_SCORING' and score['logical_frames']==4500,'Actual 4500 scoring closure required')
    for stage in score['stages']:
        require(stage['actual_wait']['exit_code']==0 and stage['actual_wait']['actual_child_waited'],'Actual stage exit0 required')
        verify_binding(stage['completion']);no_active_owner(stage['actual_wait']['pid'])
    for key in ('frame_rows','source_streams'):verify_binding(score[key])
    main_launch=read(ART/'owner_launch.json');no_active_owner(main_launch['pid'])
    actual_registration=verify_binding(main_launch['registration'])
    reg=read(actual_registration)
    require(reg['E4_allowed'] is False and reg['N']==1024 and reg['SNRs']==[4,10,19],'Scope altered')
    stats=read(OUT/'analysis_v1/completion.json')
    require(stats['status']=='EMAIN_COMMON500_E1_E2_STATISTICS_COMPLETE' and stats['new_summary_rows']==12
            and stats['new_paired_rows']==60 and stats['old_intervals_recomputed']==0,'Actual E1/E2 statistics closure required')
    plots=read(OUT/'figures_v1/completion.json')
    require(plots['status']=='EMAIN_CSV_ONLY_PAPER_FIGURES_COMPLETE','Actual figure export closure required')
    for folder,done in ((OUT/'analysis_v1',stats),(OUT/'figures_v1',plots)):
        for path,digest in done['outputs'].items():require(sha(folder/path)==digest,'Completed output changed: '+path)
    followers=[]
    for folder in sorted(OUT.glob('analysis_followthrough_v*')):
        if not (folder/'launch.json').exists():continue
        launch=read(folder/'launch.json');no_active_owner(launch['pid'])
        if (folder/'completion.json').exists():
            done=read(folder/'completion.json')
            require(done['status']=='COMPLETE_EMAIN_STATS_AND_FIGURE_EXPORTS','Invalid completed CPU successor')
            for receipt in done['actual_waits']:
                require(receipt['exit_code']==0 and receipt['actual_child_waited'],'Failed CPU successor child')
            followers.append(str(folder))
    require(len(followers)==1,'Exactly one completed CPU successor required')
    return dict(scientific_registration=dict(path=str(actual_registration),sha256=sha(actual_registration)),
                score_completion_sha256=sha(OUT/'scoring_completion.json'),statistics_completion_sha256=sha(OUT/'analysis_v1/completion.json'),
                figures_completion_sha256=sha(OUT/'figures_v1/completion.json'),completed_cpu_owner=followers[0],
                formal_png_visual_review=plots.get('visual_review','UNKNOWN'),all_scientific_calls_closed=True,E3_timing_included=False)


def build():
    closure=qualify()
    require(not PACKAGE.exists(),'Backup attempt exists; preserve it and audit before retry')
    files,excluded=inventory()
    PACKAGE.mkdir(parents=True)
    try:
        records=[]
        for name,p in sorted(files.items()):
            stat=p.stat();digest=sha(p);after=p.stat()
            require((stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'File changed during manifest hashing')
            records.append(dict(path=name,original_path=str(p),bytes=stat.st_size,sha256=digest,mtime_ns=stat.st_mtime_ns))
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        manifest=dict(schema='EMAIN_E1_E2_INCREMENTAL_BACKUP_V1',created_unix=time.time(),scope='E1_E2',
                      files=records,payload_bytes=sum(r['bytes'] for r in records),file_count=len(records),
                      repo_original_root=str(ROOT),runtime_original_root=str(ART),git_head_at_packaging=head,
                      git_push_verified_by_this_entry=False,closure=closure,dependencies=DEPENDENCIES,
                      independently_complete_restore=False,old_models_or_original_dataset_copied=False,
                      caches_and_locks_excluded=True,new_scientific_calls=0,deleted_files=0,excluded_files=excluded)
        write(PACKAGE/'backup_manifest.json',manifest)
        restore='''# EMAIN common500 E1/E2 incremental restore

This is an incremental package, not an independently complete recovery package.
It preserves this round's code/registrations, all completed source arithmetic
streams and fallback attempts, real PHY/RX outcomes and unique reconstructed
pixels, metric ledgers, source statistics, figures, failures and actual waits.
It does not copy old model weights, original500 image/token archives, environments,
or scratch caches. Those dependencies are listed in backup_manifest.json.

Restore repo/ under the chosen VAR_COMM Git root and runtime/ under a new personal
runtime directory. Every payload carries its original absolute path and SHA256.
Use a separate audited relocation map for a new server; do not rewrite historical
identities or automatically replay completed scientific calls. Keep the existing
full recovery packages, in particular final_training_20261012_v1, original500
inputs/statistics, native VAR/tokenizer/Dc/metric weights and frozen environments.

The old wait-only CPU successor and its metadata-only retirement are preserved.
The final successor is identified in the closure record; original registrations
remain historical evidence. E3 six-method timing is excluded from this E1/E2
archive and will be backed up separately after its own actual closure. This
package does not imply that timing completed. Publication/remote-Git validation
is recorded separately; git_head_at_packaging is merely a local snapshot.
'''
        (PACKAGE/'RESTORE.md').write_text(restore,encoding='utf8')
        archive=PACKAGE/(NAME+'.tar.gz')
        with archive.open('xb') as output,tarfile.open(fileobj=output,mode='w:gz',compresslevel=1) as tar:
            for row in records:
                p=Path(row['original_path']);before=p.stat()
                require((before.st_size,before.st_mtime_ns)==(row['bytes'],row['mtime_ns']),'File changed before archive read')
                tar.add(p,arcname=row['path'],recursive=False)
                require(sha(p)==row['sha256'],'File changed during archive read')
            tar.add(PACKAGE/'backup_manifest.json',arcname='backup_manifest.json',recursive=False)
            tar.add(PACKAGE/'RESTORE.md',arcname='RESTORE.md',recursive=False)
        for row in records:
            p=Path(row['original_path']);now=p.stat()
            require((now.st_size,now.st_mtime_ns)==(row['bytes'],row['mtime_ns']),'Input changed while packaging; preserve failed archive')
        receipt=dict(status='REMOTE_E1_E2_INCREMENT_CREATED_LOCAL_VERIFICATION_PENDING',archive=str(archive),
                     bytes=archive.stat().st_size,sha256=sha(archive),payload_files=len(records),payload_bytes=manifest['payload_bytes'],
                     manifest_sha256=sha(PACKAGE/'backup_manifest.json'),restore_sha256=sha(PACKAGE/'RESTORE.md'),
                     independently_complete_restore=False,deleted_files=0,new_scientific_calls=0,E3_timing_included=False)
        write(PACKAGE/'backup_archive_receipt.json',receipt)
        return receipt
    except BaseException:
        write(PACKAGE/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,deleted_files=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','build'))
    args=parser.parse_args();print(json.dumps(plan() if args.command=='plan' else build(),indent=2))
