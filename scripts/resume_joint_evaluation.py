"""Resume only the registered Joint5000 evaluation after an observed GPU-contention exit."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / 'experiments/wetok-joint-sender-r1'
sys.path.insert(0, str(EXPERIMENT / 'scripts'))

from finish_registered_trial import run_stage, validate_receipt
from evaluate_interfaces import require_uncontended_gpu
from joint_sender.evaluation import load_evaluation, matched_milestone
from joint_sender.references import References
from wetok_comm.common import now, sha256, snapshot, verify_sources, write_json


BUSY = re.compile(r'other GPU compute workloads are present: \[\d+(?:, \d+)*\]; not stopping them')


def busy_error(message):
    return BUSY.fullmatch(message) is not None


def known_contention_status(record):
    if record.get('status') != 'JOINT_EVALUATION_FAILED_OR_INTERRUPTED':
        return False
    return re.fullmatch(r"RuntimeError\('(" + BUSY.pattern + r")'\)", record.get('error', '')) is not None


def retryable_exit(returncode, log_text, state, child_pid):
    lines = [line for line in log_text.splitlines() if line.strip()]
    if returncode != 1 or not lines or not lines[-1].startswith('RuntimeError: '):
        return False
    if not busy_error(lines[-1].removeprefix('RuntimeError: ')):
        return False
    return state.get('pid') != child_pid or known_contention_status(state)


def committed_receipts(evaluation):
    return {str(path.relative_to(evaluation)): sha256(path) for path in sorted((evaluation / 'images').glob('*/receipt.json'))}


def check_commits(evaluation, committed):
    if any(sha256(evaluation / relative) != digest for relative, digest in committed.items()):
        raise RuntimeError('a previously committed source receipt changed')


def refuse_live_trial():
    scripts = {str(EXPERIMENT / 'scripts' / name).encode() for name in ('train.py', 'watch.py', 'finish_registered_trial.py', 'evaluate.py', 'analyze.py')}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            command = set((process / 'cmdline').read_bytes().split(b'\0'))
            state = (process / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
            if state != 'Z' and command.intersection(scripts):
                raise RuntimeError(f'an original trial process is still active: {process.name}')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: preserve GPU-contention failure; wait; resume original evaluation; analyze; never train')
        return
    output = arguments.output.resolve()
    if not output.is_relative_to(PROJECT / 'outputs'):
        raise ValueError('recovery output must stay within project outputs')
    evaluation = PROJECT / 'outputs/WETOK-JOINT-SENDER-R1-EVALUATION/step_0005000'
    state_path = evaluation / 'status.json'
    original = json.loads(state_path.read_text())
    if not known_contention_status(original):
        raise RuntimeError('only an explicitly recorded GPU-contention failure is eligible for this recovery')
    output.mkdir(parents=True, exist_ok=False)
    started = now()
    attempt = 0
    waiting_seconds = 0.

    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'started_local': started,
            'local_time': now(), 'evaluation_attempts': attempt, 'admission_wait_seconds': waiting_seconds,
            'research_goal_complete': False, **extra})

    with (PROJECT / 'outputs/WETOK-JOINT-SENDER-R1-step0005000-finish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            refuse_live_trial()
            write_json(output / 'original_evaluation_status.json', original)
            finish_path = PROJECT / 'outputs/WETOK-JOINT-SENDER-R1-step0005000-finish.pipeline.json'
            write_json(output / 'original_finisher_status.json', json.loads(finish_path.read_text()))
            settings, config, reference, base, parent = load_evaluation()
            milestone, milestone_path, review_path, control, control_path, control_review = matched_milestone(config, reference, 5000)
            references = References(settings)
            bindings = {'milestone_sha256': sha256(milestone_path), 'review_sha256': sha256(review_path),
                'control_milestone_sha256': sha256(control_path), 'control_review_sha256': sha256(control_review),
                'reference_qualification_sha256': sha256(references.qualification_path)}
            metadata = json.loads((evaluation / 'metadata.json').read_text())
            if any(metadata.get(key) != value for key, value in bindings.items()):
                raise RuntimeError('evaluation prerequisites changed')
            verify_sources(metadata['source_hashes'])
            committed = committed_receipts(evaluation)
            if len(committed) != int(original['completed_images']) or len(committed) >= 100:
                raise RuntimeError('the saved source count differs from the contention failure')
            initial_committed_count = len(committed)
            files = [Path(__file__), EXPERIMENT / 'scripts/finish_registered_trial.py',
                     EXPERIMENT / 'scripts/evaluate.py', EXPERIMENT / 'scripts/analyze.py']
            sources = snapshot(output, files)
            write_json(output / 'qualification.json', {'status': 'SAME_PROTOCOL_EVALUATION_RESUME_QUALIFIED',
                'local_time': now(), **bindings, 'original_metadata_sha256': sha256(evaluation / 'metadata.json'),
                'committed_source_receipts': committed, 'source_hashes': sources, 'new_training': False,
                'receiver_or_selection_changed': False, 'research_goal_complete': False})
            receipt = evaluation / 'completion.json'
            while not receipt.exists():
                quiet_since = None
                waiting_started = time.monotonic()
                status('WAITING_FOR_30_SECONDS_OBSERVED_GPU_QUIET', committed_sources=len(committed))
                while True:
                    try:
                        require_uncontended_gpu()
                    except RuntimeError as error:
                        if not busy_error(str(error)):
                            raise
                        quiet_since = None
                        status('WAITING_FOR_OTHER_GPU_WORK_WITHOUT_STOPPING_IT', detail=str(error), committed_sources=len(committed))
                    else:
                        if quiet_since is None:
                            quiet_since = time.monotonic()
                        if time.monotonic() - quiet_since >= 30:
                            break
                    time.sleep(10)
                waiting_seconds += time.monotonic() - waiting_started
                refuse_live_trial()
                check_commits(evaluation, committed)
                verify_sources(sources)
                verify_sources(metadata['source_hashes'])
                attempt += 1
                log_path = output / f'evaluation_attempt_{attempt:03d}.log'
                command = [sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts/evaluate.py'), '--step', '5000', '--resume', '--execute']
                with log_path.open('x') as log:
                    process = subprocess.Popen(command, cwd=EXPERIMENT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    status('RESUMING_ORIGINAL_JOINT_EVALUATION', evaluation_pid=process.pid, committed_sources=len(committed))
                    returncode = process.wait()
                current = json.loads(state_path.read_text())
                new_committed = committed_receipts(evaluation)
                check_commits(evaluation, committed)
                write_json(output / f'evaluation_attempt_{attempt:03d}.json', {'returncode': returncode, 'pid': process.pid,
                    'status': current, 'log_sha256': sha256(log_path), 'prior_committed_sources': len(committed),
                    'committed_sources_after': len(new_committed), 'completed_local': now()})
                if returncode == 0:
                    validate_receipt(receipt, 'JOINT_EVALUATION_COMPLETE', bindings)
                    break
                if not retryable_exit(returncode, log_path.read_text(), current, process.pid):
                    raise RuntimeError('evaluation failed for a reason other than a confirmed GPU-contention guard')
                committed = new_committed
                status('ANOTHER_GPU_CONTENTION_EXIT_PRESERVED_WAITING_TO_RESUME', committed_sources=len(committed))
            validate_receipt(receipt, 'JOINT_EVALUATION_COMPLETE', bindings)
            analysis_receipt = PROJECT / 'outputs/WETOK-JOINT-SENDER-R1-ANALYSIS/development_0005000/completion.json'
            status('RUNNING_ORIGINAL_JOINT_ANALYSIS_AND_CPU_AUDIT')
            run_stage('analyze.py', analysis_receipt, 'JOINT_DEVELOPMENT_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE',
                {'milestone_sha256': bindings['milestone_sha256'], 'control_milestone_sha256': bindings['control_milestone_sha256'],
                 'evaluation_receipt_sha256': sha256(receipt)}, output / 'analysis.log', sources)
            verify_sources(sources)
            check_commits(evaluation, committed)
            write_json(output / 'completion.json', {'status': 'REGISTERED_JOINT_EVALUATION_RESUMED_AND_ANALYZED_NOT_RESEARCH_COMPLETE',
                'completed_local': now(), **bindings, 'evaluation_receipt_sha256': sha256(receipt),
                'analysis_receipt_sha256': sha256(analysis_receipt), 'evaluation_attempts': attempt,
                'admission_wait_seconds': waiting_seconds, 'initial_committed_source_receipts_unchanged': True,
                'initial_committed_sources': initial_committed_count,
                'new_training': False, 'source_hashes': sources, 'research_goal_complete': False})
            status('REGISTERED_JOINT_EVALUATION_RESUMED_AND_ANALYZED_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('JOINT_EVALUATION_RECOVERY_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
