"""Finish the registered R3 trial only: quality, CPU analysis, independent timing; never starts another training branch."""

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
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT.parent / 'wetok-joint-sender-r1/scripts'),
    str(EXPERIMENT.parent / 'wetok-comm-v2-20260912/scripts')]

from evaluate_interfaces import require_uncontended_gpu
from finish_registered_trial import alive, owned_handle, validate_receipt
from grid_controls.hardware import gpu_memory
from vector_control.evaluation import audited_endpoint, evaluation_output, load_evaluation
from wetok_comm.common import PROJECT, now, sha256, verify_sources, write_json


def retryable_guard(stage, returncode, log, record, child_pid):
    if returncode != 1 or record.get('pid') != child_pid:
        return False
    lines = [line for line in log.splitlines() if line.strip()]
    if not lines:
        return False
    if stage == 'quality':
        messages = {'RuntimeError: insufficient free GPU memory for the registered quality admission',
            'RuntimeError: insufficient free GPU memory between quality source images',
            'RuntimeError: quality sharing needs explicit --shared-gpu when another GPU task is present'}
        return record.get('status') == 'R3_QUALITY_FAILED_OR_INTERRUPTED' and lines[-1] in messages
    if stage == 'timing':
        return record.get('status') == 'R3_TIMING_FAILED_OR_INTERRUPTED' and re.fullmatch(
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
        print('PLAN ONLY: owned R3 10000 -> quality -> CPU audit -> independent timing; no new training branch')
        return
    trainer = owned_handle(arguments.training_pid, EXPERIMENT / 'scripts/train.py', {'--until-update': 10000})
    reviewer = owned_handle(arguments.review_watcher_pid, EXPERIMENT / 'scripts/watch.py',
        {'--training-pid': arguments.training_pid, '--step': 10000})
    evaluation, config, r2, original, grid, reference, base, parent = load_evaluation()
    prefix = PROJECT / 'outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-step0010000-finish'
    files = [Path(__file__), EXPERIMENT / 'scripts/evaluate_quality.py', EXPERIMENT / 'scripts/analyze_quality.py',
        EXPERIMENT / 'scripts/evaluate_timing.py', EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
        *sorted((EXPERIMENT / 'src/vector_control').glob('*.py'))]
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
                status('WAITING_FOR_SHARED_R3_QUALITY_MEMORY', free_bytes=free)
            else:
                try:
                    require_uncontended_gpu()
                    return
                except RuntimeError as error:
                    if not str(error).startswith('other GPU compute workloads are present:'):
                        raise
                    status('WAITING_FOR_UNCONTENDED_R3_' + stage.upper(), detail=str(error))
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
            command = [sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts' / script), '--step', '10000', '--execute']
            if stage == 'quality' and arguments.shared_quality:
                command.append('--shared-gpu')
            if stage in ('quality', 'timing') and (receipt.parent / 'metadata.json').exists():
                command.append('--resume')
            with log_path.open('x') as log:
                process = subprocess.Popen(command, cwd=EXPERIMENT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status('RUNNING_REGISTERED_R3_' + stage.upper(), child_pid=process.pid, attempt=attempt, log=str(log_path))
                returncode = process.wait()
            if returncode == 0:
                return validate_receipt(receipt, expected_status, bindings)
            state_path = receipt.parent / 'status.json'
            current = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not retryable_guard(stage, returncode, log_path.read_text(), current, process.pid):
                raise RuntimeError({'stage': stage, 'returncode': returncode, 'state': current, 'log': str(log_path)})
            status('R3_RESOURCE_GUARD_EXIT_PRESERVED', stage=stage, log=str(log_path), state=current)
            time.sleep(10)

    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('WAITING_FOR_OWNED_R3_10000_TRAINING_AND_REVIEW')
            pipeline = PROJECT / 'outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-step0010000.pipeline.json'
            while alive(trainer) or alive(reviewer):
                if pipeline.exists() and json.loads(pipeline.read_text())['status'] == 'R3_PIPELINE_NEEDS_REVIEW':
                    raise RuntimeError('owned R3 training/review requires inspection')
                time.sleep(30)
            milestone, milestone_path, review_path, qualified = audited_endpoint(config, r2, grid, 10000)
            milestone_sha = sha256(milestone_path)
            quality = evaluation_output(evaluation, 'quality') / 'completion.json'
            run('quality', 'evaluate_quality.py', quality, 'R3_QUALITY_COMPLETE',
                {'r3_milestone_sha256': milestone_sha, 'r3_review_sha256': sha256(review_path)})
            analysis = evaluation_output(evaluation, 'quality_analysis') / 'completion.json'
            run('analysis', 'analyze_quality.py', analysis, 'R3_QUALITY_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE',
                {'r3_milestone_sha256': milestone_sha, 'quality_receipt_sha256': sha256(quality)}, admission=False)
            timing = evaluation_output(evaluation, 'timing') / 'completion.json'
            run('timing', 'evaluate_timing.py', timing, 'R3_TIMING_COMPLETE',
                {'r3_milestone_sha256': milestone_sha, 'quality_receipt_sha256': sha256(quality)})
            verify_sources(sources)
            status('R3_QUALITY_AND_SEPARATE_TIMING_COMPLETE_NOT_RESEARCH_COMPLETE', quality_receipt_sha256=sha256(quality),
                analysis_receipt_sha256=sha256(analysis), timing_receipt_sha256=sha256(timing), new_training_branches_started=0)
        except BaseException as error:
            status('R3_COMPLETION_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
