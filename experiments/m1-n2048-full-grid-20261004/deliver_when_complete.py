"""Finish this registered run: metrics, static report, checked normal Git push.

This owner never starts training, retries a failed scientific stage, resumes a
historical queue, or edits registered inference code.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import m1_common as u

STOP=False
def stop(*_):
    global STOP
    STOP=True
def check(out):
    if STOP or (out/'STOP').exists():raise InterruptedError('User requested a safe stop')
def command(args,root,**kwargs):return subprocess.run(args,cwd=root,check=True,**kwargs)

def run(root,out):
    here=Path(__file__).resolve().parent;config_path=out/'delivery_config.json';config=u.read(config_path)
    u.verify(config['bindings']);u.require(config['root']==str(root) and config['out']==str(out),'Wrong delivery target')
    result=root/'results/m1_n2048_full_grid_20261004';report=root/'reports/m1_n2048_full_grid_20261004.md'
    def status(s,**kw):u.write(out/'delivery_status.json',dict(status=s,pid=os.getpid(),time=time.time(),**kw))
    status('WAITING_FOR_REGISTERED_RECONSTRUCTIONS')
    while not (out/'development_completion.json').exists():
        check(out)
        if (out/'owner_failure.json').exists():raise RuntimeError('Registered reconstruction owner failed; inspect owner_failure.json')
        time.sleep(30)
    # The last CUDA process releases its context only after exiting.
    status('WAITING_FOR_GPU_RELEASE')
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        check(out);time.sleep(15)
    u.verify(config['bindings']);check(out);status('SCORING_UNIFIED_METRICS')
    command([sys.executable,'-B',str(here/'score_n2048.py'),'--root',str(root),'--out',str(out)],root)
    check(out);status('BUILDING_REPORT')
    command([sys.executable,'-B',str(here/'build_report.py'),'--root',str(root),'--out',str(out),'--result',str(result)],root)
    manifest=u.read(result/'MANIFEST.json');u.require(manifest['status']=='STATIC_REPORT_COMPLETE','Static report not complete')
    for relative,digest in manifest['outputs'].items():
        path=Path(relative);path=path if path.is_absolute() else result/path
        u.require(u.sha(path)==digest and path.stat().st_size<10_000_000,'Published output hash or size differs')
    text=(result/'REPORT.md').read_text(encoding='utf-8')
    report.write_text(text.replace('](figures/','](../results/m1_n2048_full_grid_20261004/figures/')
        .replace('](SAME_K.md)','](../results/m1_n2048_full_grid_20261004/SAME_K.md)')
        .replace('](MAIN_TABLE.md)','](../results/m1_n2048_full_grid_20261004/MAIN_TABLE.md)')
        .replace('](MAIN_TABLE.csv)','](../results/m1_n2048_full_grid_20261004/MAIN_TABLE.csv)')
        .replace('](POLICIES.csv)','](../results/m1_n2048_full_grid_20261004/POLICIES.csv)')
        .replace('](metrics_paired_intervals.csv)','](../results/m1_n2048_full_grid_20261004/metrics_paired_intervals.csv)')
        .replace('](m1_vs_P_paired.csv)','](../results/m1_n2048_full_grid_20261004/m1_vs_P_paired.csv)')
        .replace('](resource_summary.csv)','](../results/m1_n2048_full_grid_20261004/resource_summary.csv)')
        .replace('](calibration_summary.csv)','](../results/m1_n2048_full_grid_20261004/calibration_summary.csv)'),encoding='utf-8')
    status('VERIFYING_REPOSITORY')
    u.require(not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip(),'Unexpected staged work; preserve it')
    command(['git','fetch','origin'],root)
    command(['git','merge-base','--is-ancestor','origin/main','HEAD'],root)
    paths=[str(p.relative_to(root)) for p in result.rglob('*') if p.is_file()]+[str(report.relative_to(root))]
    # Source is staged only from the explicitly frozen delivery inventory.
    paths+=config['publish_source_paths']
    command(['git','add','--',*paths],root)
    command(['python3','-B','tools/update_repository_manifest.py'],root)
    command(['git','add','--','release_manifest.json'],root)
    command(['python3','-B','tools/verify_repository.py'],root)
    command(['python3','-B','tools/run_cpu_checks.py'],root)
    check(out);u.verify(config['bindings'])
    command(['git','commit','-m','Report complete N2048 M1 calibration and paired evaluation'],root)
    command(['git','push','origin','HEAD:main'],root)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=root,text=True).split()[0]
    u.require(remote==commit,'Normal push remote SHA mismatch')
    u.seal(out/'publication_completion.json',dict(status='PUSHED_AND_STOPPED',commit=commit,remote_commit=remote,checks='PASS',
        score_completion_sha256=u.sha(out/'score_completion.json'),report_manifest_sha256=u.sha(result/'MANIFEST.json'),
        training_updates=0,old_queues_resumed=False,figures_structurally_verified=True,
        final_figures_manual_visual_review='NOT_YET_PERFORMED',report=str(report.relative_to(root))))
    status('PUSHED_AND_STOPPED',commit=commit)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.root).resolve();out=Path(a.out).resolve();lock=(out/'delivery.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:run(root,out)
    except BaseException as e:
        u.write(out/'delivery_failure.json',dict(error=repr(e),automatic_retry=False));raise
    finally:lock.close()
if __name__=='__main__':main()
