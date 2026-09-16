"""After the owned 5000-update trial and audit, run its already-registered evaluation and analysis."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

from evaluate_interfaces import require_uncontended_gpu
from innovation_comm.common import output_path, settings
from innovation_comm.evaluation import audited_milestone
from wetok_comm.common import PROJECT, now, sha256, verify_sources, write_json


def owned_handle(pid, script):
    process = Path(f'/proc/{pid}')
    if process.stat().st_uid != os.getuid() or str(script).encode() not in (process / 'cmdline').read_bytes().split(b'\0'):
        raise RuntimeError('completion watcher target is not the registered owned process')
    return pid, (process / 'stat').read_text().split(') ', 1)[1].split()[19]


def alive(handle):
    pid, started = handle
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == started
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--review-watcher-pid', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, base, parent = settings()
    if not arguments.execute:
        print('PLAN ONLY: wait for owned receiver5000 and audited selection, then evaluate/analyze; never starts another training branch')
        return
    trainer = owned_handle(arguments.training_pid, EXPERIMENT / 'scripts/train.py')
    reviewer = owned_handle(arguments.review_watcher_pid, EXPERIMENT / 'scripts/watch.py')
    root = PROJECT / 'outputs'
    prefix = root / 'WETOK-INNOVATION-R1-step0005000-finish'
    sources = [Path(__file__), EXPERIMENT / 'scripts/evaluate.py', EXPERIMENT / 'scripts/analyze.py',
        EXPERIMENT / 'scripts/plot_calibration.py', EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
        *sorted((EXPERIMENT / 'src/innovation_comm').glob('*.py'))]
    hashes = {str(path.relative_to(PROJECT.parent)): sha256(path) for path in sources}
    def status(value, **extra):
        write_json(prefix.with_suffix('.pipeline.json'), {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'training_pid': trainer[0], 'review_watcher_pid': reviewer[0], 'source_hashes': hashes,
            'research_goal_complete': False, **extra})
    def run_stage(script, receipt, valid_status):
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record['status'] != valid_status or record['milestone_sha256'] != sha256(milestone_path):
                raise RuntimeError('existing registered stage is not a valid completed result')
            verify_sources(record['source_hashes'])
            for relative, expected in record['output_hashes'].items():
                if sha256(receipt.parent / relative) != expected:
                    raise RuntimeError('completed registered stage artifact changed')
            return
        command = [sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts' / script), '--step', '5000', '--execute']
        if script == 'evaluate.py' and (receipt.parent / 'metadata.json').exists():
            command.append('--resume')
        with prefix.with_suffix('.stages.log').open('a') as log:
            subprocess.run(command, cwd=EXPERIMENT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads(receipt.read_text())
        if record['status'] != valid_status or record['milestone_sha256'] != sha256(milestone_path):
            raise RuntimeError('registered stage returned without its valid completion receipt')
    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('WAITING_FOR_OWNED_5000_TRAINING_AND_REVIEW')
            producer_pipeline = root / 'WETOK-INNOVATION-R1-step0005000.pipeline.json'
            while alive(trainer) or alive(reviewer):
                if producer_pipeline.exists():
                    record = json.loads(producer_pipeline.read_text())
                    if record['status'] == 'INNOVATION_PIPELINE_NEEDS_REVIEW':
                        raise RuntimeError(record)
                time.sleep(30)
            milestone, milestone_path, review_path = audited_milestone(config, 5000)
            verify_sources(hashes)
            status('WAITING_FOR_UNCONTENDED_GPU_FOR_REGISTERED_EVALUATION', milestone_sha256=sha256(milestone_path))
            while True:
                try:
                    require_uncontended_gpu()
                    break
                except RuntimeError as error:
                    status('WAITING_FOR_OTHER_GPU_WORK_WITHOUT_STOPPING_IT', detail=str(error))
                    time.sleep(60)
            status('RUNNING_REGISTERED_RECEIVER_DEVELOPMENT_EVALUATION')
            evaluation_receipt = output_path(config, 'evaluation') / 'step_0005000/completion.json'
            run_stage('evaluate.py', evaluation_receipt, 'INNOVATION_EVALUATION_COMPLETE')
            status('RUNNING_REGISTERED_RECEIVER_STATISTICS_AND_CPU_ARCHIVE_AUDIT')
            analysis_receipt = output_path(config, 'analysis') / 'development_0005000/completion.json'
            run_stage('analyze.py', analysis_receipt, 'INNOVATION_DEVELOPMENT_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE')
            run_stage('plot_calibration.py', output_path(config, 'analysis') / 'calibration_plots_0005000/completion.json',
                      'AUDITED_CALIBRATION_PLOTS_COMPLETE')
            verify_sources(hashes)
            status('REGISTERED_RECEIVER_TRIAL_EVALUATED_AND_ANALYZED_NOT_RESEARCH_COMPLETE',
                evaluation_receipt_sha256=sha256(evaluation_receipt), analysis_receipt_sha256=sha256(analysis_receipt),
                new_training_branches_started=0)
        except BaseException as error:
            status('REGISTERED_TRIAL_COMPLETION_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
