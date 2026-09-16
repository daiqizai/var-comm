"""Finish registered Grid5000: quality, CPU analysis, then separately admitted timing."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(JOINT / 'scripts'),
               str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

from evaluate_interfaces import require_uncontended_gpu
from finish_registered_trial import alive, owned_handle, validate_receipt
from grid_controls.common import output_path
from grid_controls.evaluation import audited_grid, load_evaluation
from grid_controls.hardware import gpu_memory
from wetok_comm.common import PROJECT, now, sha256, verify_sources, write_json


def retryable_guard(stage, returncode, log, record, child_pid):
    if returncode != 1 or record.get('pid') != child_pid:
        return False
    lines = [line for line in log.splitlines() if line.strip()]
    if not lines:
        return False
    if stage == 'quality':
        messages = {'RuntimeError: insufficient free GPU memory for the registered quality admission',
                    'RuntimeError: insufficient free GPU memory between quality source images'}
        return record.get('status') == 'GRID_QUALITY_FAILED_OR_INTERRUPTED' and lines[-1] in messages
    if stage == 'timing':
        return record.get('status') == 'GRID_TIMING_FAILED_OR_INTERRUPTED' and re.fullmatch(
            r'RuntimeError: other GPU compute workloads are present: \[\d+(?:, \d+)*\]; not stopping them', lines[-1]) is not None
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--review-watcher-pid', type=int, required=True)
    parser.add_argument('--shared-quality', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: owned5000 -> audited quality -> CPU analysis -> separate timing; never starts training')
        return
    trainer = owned_handle(arguments.training_pid, EXPERIMENT / 'scripts/train.py', {'--until-update': 5000})
    reviewer = owned_handle(arguments.review_watcher_pid, EXPERIMENT / 'scripts/watch.py',
        {'--training-pid': arguments.training_pid, '--step': 5000})
    evaluation, config, original, reference, base, parent = load_evaluation()
    prefix = PROJECT / 'outputs/WETOK-JOINT-GRID-CONTROLS-R1-step0005000-finish'
    files = [Path(__file__), EXPERIMENT / 'scripts/evaluate_quality.py', EXPERIMENT / 'scripts/analyze_quality.py',
        EXPERIMENT / 'scripts/evaluate_timing.py', EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
        *sorted((EXPERIMENT / 'src/grid_controls').glob('*.py'))]
    sources = {str(path.relative_to(PROJECT.parent)): sha256(path) for path in files}

    def status(value, **extra):
        write_json(prefix.with_suffix('.pipeline.json'), {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'training_pid': trainer[0], 'review_watcher_pid': reviewer[0], 'shared_quality_requested': arguments.shared_quality,
            'source_hashes': sources, 'research_goal_complete': False, **extra})

    def wait_admission(stage):
        while True:
            if stage == 'quality' and arguments.shared_quality:
                free, total = gpu_memory()
                if free >= int(evaluation['shared_start_free_GiB'] * 1024 ** 3):
                    return
                status('WAITING_FOR_SHARED_QUALITY_MEMORY_WITHOUT_STOPPING_OTHER_TASKS', free_bytes=free)
            else:
                try:
                    require_uncontended_gpu()
                    return
                except RuntimeError as error:
                    if not str(error).startswith('other GPU compute workloads are present:'):
                        raise
                    status('WAITING_FOR_UNCONTENDED_' + stage.upper(), detail=str(error))
            time.sleep(30)

    def run(stage, script, receipt, expected_status, bindings, admission=True):
        if receipt.exists():
            return validate_receipt(receipt, expected_status, bindings)
        attempt = 0
        while not receipt.exists():
            if admission:
                wait_admission(stage)
            verify_sources(sources)
            attempt += 1
            log_path = prefix.with_suffix(f'.{stage}_{time.time_ns()}_{attempt}.log')
            command = [sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts' / script), '--step', '5000', '--execute']
            if stage == 'quality' and arguments.shared_quality:
                command.append('--shared-gpu')
            if stage in ('quality', 'timing') and (receipt.parent / 'metadata.json').exists():
                command.append('--resume')
            with log_path.open('x') as log:
                process = subprocess.Popen(command, cwd=EXPERIMENT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status('RUNNING_REGISTERED_GRID_' + stage.upper(), child_pid=process.pid, attempt=attempt, log=str(log_path))
                returncode = process.wait()
            if returncode == 0:
                return validate_receipt(receipt, expected_status, bindings)
            state_path = receipt.parent / 'status.json'
            current = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not retryable_guard(stage, returncode, log_path.read_text(), current, process.pid):
                raise RuntimeError({'stage': stage, 'returncode': returncode, 'state': current, 'log': str(log_path)})
            status('REGISTERED_GRID_RESOURCE_GUARD_EXIT_PRESERVED', stage=stage, log=str(log_path), state=current)
            time.sleep(10)

    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('WAITING_FOR_OWNED_GRID5000_TRAINING_AND_REVIEW')
            pipeline = PROJECT / 'outputs/WETOK-JOINT-GRID-CONTROLS-R1-step0005000.pipeline.json'
            while alive(trainer) or alive(reviewer):
                if pipeline.exists() and json.loads(pipeline.read_text())['status'] == 'GRID_PIPELINE_NEEDS_REVIEW':
                    raise RuntimeError('owned5000 grid training/review requires inspection')
                time.sleep(30)
            milestone, milestone_path, review_path, references = audited_grid(config, original, reference, 5000)
            grid_sha = sha256(milestone_path)
            quality = output_path(config, 'evaluation') / 'quality_0005000/completion.json'
            run('quality', 'evaluate_quality.py', quality, 'GRID_QUALITY_COMPLETE',
                {'grid_milestone_sha256': grid_sha, 'grid_review_sha256': sha256(review_path)})
            analysis = output_path(config, 'analysis') / 'quality_0005000/completion.json'
            run('analysis', 'analyze_quality.py', analysis, 'GRID_QUALITY_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE',
                {'grid_milestone_sha256': grid_sha, 'quality_receipt_sha256': sha256(quality)}, admission=False)
            timing = output_path(config, 'evaluation') / 'timing_0005000/completion.json'
            run('timing', 'evaluate_timing.py', timing, 'GRID_TIMING_COMPLETE',
                {'grid_milestone_sha256': grid_sha, 'quality_receipt_sha256': sha256(quality)})
            verify_sources(sources)
            status('REGISTERED_GRID_QUALITY_AND_SEPARATE_TIMING_COMPLETE_NOT_RESEARCH_COMPLETE',
                quality_receipt_sha256=sha256(quality), analysis_receipt_sha256=sha256(analysis), timing_receipt_sha256=sha256(timing),
                new_training_branches_started=0)
        except BaseException as error:
            status('GRID_COMPLETION_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
