"""Finite, once-only CPU statistics successor; never schedules GPU work."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/baseline_strength_20261011/paired_supplement_v1'
CHAIN=ROOT/'results/baseline_strength_20261011/automatic_followthrough_v1'
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def run():
    lock=(OUT/'successor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'successor_launch.json').exists(),'Already attempted'
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'statistics.py',HERE/'after_scoring.py',HERE/'plot_evaluation.py',OUT/'registration.json']}
    write('successor_launch.json',dict(pid=os.getpid(),started_unix=time.time(),input_sha256=pins))
    os.nice(10);os.sched_setaffinity(0,{43,47})
    started=time.monotonic()
    while not (CHAIN/'completion.json').exists():
        assert not list(CHAIN.glob('*failure.json')),'Existing scoring failed; do not retry'
        assert not (CHAIN/'STOP').exists() and not (OUT/'STOP').exists(),'Stop requested'
        assert time.monotonic()-started<24*3600,'Finite wait expired'
        write('successor_status.json',dict(status='WAITING_FOR_CLOSED_SCORING',pid=os.getpid(),updated_unix=time.time()))
        time.sleep(30)
    for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert not (OUT/'STOP').exists()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONNOUSERSITE='1')
    with (TMP/'postprocess_statistics_v1.log').open('xb') as log:
        child=subprocess.Popen([PY,str(HERE/'statistics.py'),'run'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write('successor_status.json',dict(status='CPU_STATISTICS_RUNNING',pid=os.getpid(),worker_pid=child.pid))
        try:code=child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();code=child.wait()
            write('actual_wait.json',dict(actual_child_waited=True,exit_code=code,timed_out=True));raise
    write('actual_wait.json',dict(actual_child_waited=True,exit_code=code,timed_out=False))
    assert code==0,'Statistics failed; no automatic repeat'
    assert json.loads((OUT/'completion.json').read_text())['status']=='COMPLETE_BOTH_SOURCE_PAIRED_SUPPLEMENTS'
    env['MPLCONFIGDIR']=str(TMP/'matplotlib')
    for mode in ['diagnostics','final']:
        with (TMP/('postprocess_plots_'+mode+'_v1.log')).open('xb') as log:
            result=subprocess.run([PY,str(HERE/'plot_evaluation.py'),mode],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        write('plots_'+mode+'_actual_wait.json',dict(actual_child_waited=True,exit_code=result.returncode))
        assert result.returncode==0,'Figure export failed; no scientific work repeated'
    write('successor_status.json',dict(status='STATISTICS_AND_FIGURES_COMPLETE_VISUAL_REVIEW_TIMING_PUBLICATION_PENDING',pid=os.getpid()))


if __name__=='__main__':
    try:run()
    except BaseException:
        write('successor_failure.json',dict(status='STOPPED_NO_RETRY',traceback=traceback.format_exc()))
        raise
