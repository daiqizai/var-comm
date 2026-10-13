"""Metadata-corrected CPU successor, bound to the actually launched science.

The wait performs no scientific calls. Failures stop without automatic replay.
Models, reconstruction, scoring and the independent timing owner are untouched.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
from analyze import read, write, rows, require, sha, bound, STAT_SHA

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/entropy_main_common500_20261013'
STATE = OUT/'analysis_followthrough_v2'
TMP = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
ART = TMP/'entropy_main_common500_20261013_v1'
PYTHON = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python')
E0 = OUT/'e0_reuse'


def descriptor(path):
    return dict(path=str(path), sha256=sha(path))


def status(value):
    temp = STATE/'status.json.pending'
    write(temp, value)
    os.replace(temp, STATE/'status.json')


def prepare():
    STATE.mkdir(exist_ok=True)
    require(not (STATE/'registration.json').exists(), 'Successor already registered')
    e0 = read(E0/'audit.json')
    require(e0['status']=='E0_REFERENCE_DATA_AND_SOURCE_REUSE_VERIFIED', 'Completed E0 reuse audit required')
    retired = read(OUT/'analysis_followthrough_v1/retirement_zero_science_metadata_v2.json')
    require(retired['status']=='WAIT_ONLY_OWNER_TERMINATED_IDENTITY_VERIFIED' and retired['new_bootstrap_calls']==0,
            'Original zero-science wait owner retirement required')
    science_launch = read(ART/'owner_launch.json')
    actual_registration = bound(science_launch['registration'])
    names = ('analyze.py','plot.py','analysis_owner_v2.py','test_analysis.py')
    e0names = ('source_means.csv','summary.csv','original_summary_rows.json','source_order.json','audit.json')
    registered = dict(schema='EMAIN_CPU_FOLLOWTHROUGH_V2_ACTUAL_SCIENCE_REGISTRATION',created_unix=time.time(),
                      science_registration=descriptor(actual_registration),
                      science_owner_launch=descriptor(ART/'owner_launch.json'),
                      predecessor_retirement=descriptor(OUT/'analysis_followthrough_v1/retirement_zero_science_metadata_v2.json'),
                      code_bindings={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in names},
                      reuse_bindings={str(E0/n):sha(E0/n) for n in e0names},
                      new_intervals=72,bootstrap_seed=2026100701,bootstrap_replicates=10000,
                      wait_seconds=30,automatic_retries=False,training=False,inference=False,E4_allowed=False)
    write(STATE/'registration.json', registered)
    status(dict(status='REGISTERED_WAITING_FOR_ACTUAL_SCORE_CLOSURE',scientific_calls=0))


def manifest():
    """Admit immutable completed science receipts before any bootstrap draw."""
    completed = read(OUT/'scoring_completion.json')
    require(completed['status']=='COMPLETE_EMAIN4500_SCORING' and completed['logical_frames']==4500,
            'All 4500 new logical frames must actually complete')
    waited=read(OUT/'actual_wait_scoring.json')
    require(waited['exit_code']==0 and waited['actual_child_waited'], 'Missing actual successful scoring wait')
    require([s['stage'] for s in completed['stages']]==['source','phy','visual','score'], 'Incomplete actual scientific stage chain')
    for stage in completed['stages']:
        receipt=stage['actual_wait']
        require(receipt['exit_code']==receipt['returncode']==0 and receipt['actual_child_waited'], 'Failed science child')
        bound(stage['completion'])
    require(bound(completed['frame_rows'])==OUT/'frame_rows.csv','Unexpected frame output')
    require(bound(completed['source_streams'])==OUT/'source_streams.csv','Unexpected source-stream output')
    score=read(bound(completed['stages'][-1]['completion']))
    require(score['status']=='COMPLETE_SCORE4500' and score['logical_frames']==4500 and score['metric_cells']==18000,
            'Full four-metric scoring closure required')
    require(score['counts']['unresolved']==0,'Unresolved scoring calls')
    metric_path=bound(score['metric_identity'])
    require(bound(score['frame_rows'])==OUT/'frame_rows.csv','Score completion rows differ')
    ownreg=read(STATE/'registration.json')
    science_path=bound(ownreg['science_registration'])
    actual_launch=read(bound(ownreg['science_owner_launch']))
    require(bound(actual_launch['registration'])==science_path,'Actual scientific registration changed')
    reg=read(science_path)
    require(reg['E4_allowed'] is False and reg['N']==1024 and reg['SNRs']==[4,10,19], 'Registered scope differs')
    ids=read(E0/'source_order.json')
    require([r['source_id'] for r in reg['sources']]==ids,'Science and reuse source order differ')
    for path,digest in reg['code_bindings'].items():
        require(sha(path)==digest,'Bound scientific source changed: '+path)
    data=rows(OUT/'frame_rows.csv')
    require(len({r['metric_identity'] for r in data})==1 and len({r['model_identity'] for r in data})==1,
            'Mixed new evaluator/model identity')
    table_paths=dict(new_frame_rows=OUT/'frame_rows.csv',source_streams=OUT/'source_streams.csv',
                     reference_source_means=E0/'source_means.csv',reference_summary=E0/'summary.csv',
                     reference_summary_unchanged=E0/'original_summary_rows.json',
                     reference_whole_lengths=ROOT/'results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/source_lengths.csv',
                     statistics_module=TMP/'postprocess_original500_v1/raw64_common500_source_statistics.py',
                     scoring_completion=OUT/'scoring_completion.json',actual_wait=OUT/'actual_wait_scoring.json',
                     metric_identity=metric_path,e0_audit=E0/'audit.json',science_registration=science_path)
    require(sha(table_paths['statistics_module'])==STAT_SHA,'Original bootstrap source changed')
    audit=read(E0/'audit.json')
    for key in ('reference_whole_lengths','statistics_module'):
        path=table_paths[key]
        require(audit['input_bindings'][str(path)]==sha(path),'Reference binding differs from E0')
    plan=dict(schema='EMAIN_ANALYSIS_INPUTS_V1',source_ids=ids,inputs={k:descriptor(v) for k,v in table_paths.items()},
              scoring_expected_status='COMPLETE_EMAIN4500_SCORING',frozen_policy_verified=True,
              source_pixels_verified=True,metric_identity_verified=True,reference_reuse_verified=True,
              flags_evidence='Actual four-stage exit-zero chain, original science registration, frozen code bindings, completed four-metric scorer, and closed E0 audit.',
              E4_allowed=False,old_intervals_recomputed=0)
    write(STATE/'analysis_manifest.json',plan)


def run():
    lock=(STATE/'owner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (STATE/'launch.json').exists(),'Successor already attempted; no automatic retry')
    reg=read(STATE/'registration.json')
    for path,digest in {**reg['code_bindings'],**reg['reuse_bindings']}.items():
        require(sha(path)==digest,'Registered CPU successor input changed: '+path)
    write(STATE/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),registration=descriptor(STATE/'registration.json')))
    for key in ('science_registration','science_owner_launch','predecessor_retirement'):
        bound(reg[key])
    logs=ART/'cpu_followthrough_logs_v2';logs.mkdir(parents=True,exist_ok=True)
    child=None;waits=[]
    try:
        while not ((OUT/'scoring_completion.json').exists() and (OUT/'actual_wait_scoring.json').exists()):
            require(not (OUT/'STOP').exists(),'Study STOP present')
            failures=[p for p in (ART/'owner_failure.json',ART/'source/failure.json',ART/'phy/failure.json',ART/'visual/failure.json',ART/'score/failure.json') if p.exists()]
            require(not failures,'Scientific failure retained; stop successor: '+str(failures))
            status(dict(status='WAITING_FOR_ACTUAL_4500_SCORE_CLOSURE',owner_pid=os.getpid(),last_checked_unix=time.time(),scientific_calls=0))
            time.sleep(30)
        manifest()
        for stage,command in (
            ('statistics',[str(PYTHON),str(Path(__file__).with_name('analyze.py')),'--manifest',str(STATE/'analysis_manifest.json'),'--out',str(OUT/'analysis_v1')]),
            ('figures',[str(PYTHON),str(Path(__file__).with_name('plot.py')),'--analysis',str(OUT/'analysis_v1'),'--out',str(OUT/'figures_v1')])):
            for path,digest in reg['code_bindings'].items():require(sha(path)==digest,'Bound CPU source changed before '+stage)
            require(not (OUT/'STOP').exists(),'Study STOP present')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                                            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            status(dict(status='RUNNING_'+stage.upper(),owner_pid=os.getpid(),actual_waits=waits))
            started=time.time()
            with (logs/(stage+'.stdout.log')).open('xb') as stdout,(logs/(stage+'.stderr.log')).open('xb') as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,cwd=ROOT)
                code=child.wait()
            receipt=dict(stage=stage,exit_code=code,returncode=code,actual_child_waited=True,pid=child.pid,elapsed_seconds=time.time()-started)
            child=None;write(STATE/('actual_wait_'+stage+'.json'),receipt);waits.append(receipt)
            require(code==0,'CPU child failed without retry: '+stage)
        write(STATE/'completion.json',dict(status='COMPLETE_EMAIN_STATS_AND_FIGURE_EXPORTS',actual_waits=waits,
                                           statistics=descriptor(OUT/'analysis_v1/completion.json'),figures=descriptor(OUT/'figures_v1/completion.json'),
                                           actual_png_visual_review='PENDING',publication_and_backup='PENDING',timing_complete=False))
        status(dict(status='COMPLETE_EMAIN_STATS_AND_FIGURE_EXPORTS_VISUAL_REVIEW_PENDING',actual_waits=waits))
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate();code=child.wait()
            write(STATE/'interrupted_actual_wait.json',dict(exit_code=code,actual_child_waited=True,pid=child.pid))
        write(STATE/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,actual_waits=waits))
        status(dict(status='STOPPED_NO_RETRY',actual_waits=waits));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run'))
    args=p.parse_args();prepare() if args.command=='prepare' else run()
