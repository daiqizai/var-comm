"""Finish the registered Joint5000 evaluation after its owned trainer and review exit."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

from evaluate_interfaces import require_uncontended_gpu
from joint_sender.common import output_path, settings
from joint_sender.evaluation import matched_milestone
from wetok_comm.common import PROJECT, now, sha256, verify_sources, write_json


def owned_handle(pid, script, required_arguments, process_root=Path('/proc')):
    process = process_root / str(pid)
    command = (process / 'cmdline').read_bytes().split(b'\0')
    if process.stat().st_uid != os.getuid() or str(script).encode() not in command or b'--execute' not in command:
        raise RuntimeError('completion target is not the registered owned executable')
    for option, value in required_arguments.items():
        encoded = option.encode()
        if command.count(encoded) != 1 or command[command.index(encoded) + 1] != str(value).encode():
            raise RuntimeError('completion target has a different milestone or trainer')
    fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
    if fields[0] == 'Z':
        raise RuntimeError('completion target has already exited')
    return pid, fields[19]


def alive(handle, process_root=Path('/proc')):
    pid, started = handle
    try:
        fields = (process_root / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == started
    except FileNotFoundError:
        return False


def validate_receipt(receipt, valid_status, bindings):
    record = json.loads(receipt.read_text())
    if record['status'] != valid_status or any(record.get(key) != value for key, value in bindings.items()):
        raise RuntimeError('registered stage receipt has a different result or prerequisite')
    verify_sources(record['source_hashes'])
    for relative, expected in record['output_hashes'].items():
        artifact = (receipt.parent / relative).resolve()
        if not artifact.is_relative_to(receipt.parent.resolve()) or sha256(artifact) != expected:
            raise RuntimeError('registered stage artifact changed or escaped its output directory')
    return record


def run_stage(script, receipt, valid_status, bindings, log_path, source_hashes):
    verify_sources(source_hashes)
    if not receipt.exists():
        command = [sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts' / script), '--step', '5000', '--execute']
        if script == 'evaluate.py' and (receipt.parent / 'metadata.json').exists():
            command.append('--resume')
        with log_path.open('a') as log:
            subprocess.run(command, cwd=EXPERIMENT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    return validate_receipt(receipt, valid_status, bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--review-watcher-pid', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: owned Joint5000 -> calibration review -> registered evaluation -> CPU analysis; no new training')
        return
    trainer = owned_handle(arguments.training_pid, EXPERIMENT / 'scripts/train.py', {'--until-update': 5000})
    reviewer = owned_handle(arguments.review_watcher_pid, EXPERIMENT / 'scripts/watch.py',
        {'--step': 5000, '--training-pid': arguments.training_pid})
    config, reference, base, parent = settings()
    root = PROJECT / 'outputs'
    prefix = root / 'WETOK-JOINT-SENDER-R1-step0005000-finish'
    sources = [Path(__file__), EXPERIMENT / 'scripts/evaluate.py', EXPERIMENT / 'scripts/analyze.py',
        EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
        *sorted((EXPERIMENT / 'src/joint_sender').glob('*.py'))]
    hashes = {str(path.relative_to(PROJECT.parent)): sha256(path) for path in sources}

    def status(value, **extra):
        write_json(prefix.with_suffix('.pipeline.json'), {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'training_pid': trainer[0], 'review_watcher_pid': reviewer[0], 'source_hashes': hashes,
            'research_goal_complete': False, **extra})

    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('WAITING_FOR_OWNED_JOINT5000_TRAINING_AND_REVIEW')
            producer = root / 'WETOK-JOINT-SENDER-R1-step0005000.pipeline.json'
            while alive(trainer) or alive(reviewer):
                if producer.exists():
                    current = json.loads(producer.read_text())
                    if current['status'] == 'JOINT_PIPELINE_NEEDS_REVIEW':
                        raise RuntimeError(current)
                time.sleep(30)
            milestone, milestone_path, review_path, control, control_path, control_review = matched_milestone(config, reference, 5000)
            verify_sources(hashes)
            bindings = {'milestone_sha256': sha256(milestone_path), 'control_milestone_sha256': sha256(control_path)}
            evaluation_receipt = output_path(config, 'evaluation') / 'step_0005000/completion.json'
            if not evaluation_receipt.exists():
                while True:
                    try:
                        require_uncontended_gpu()
                        break
                    except RuntimeError as error:
                        status('WAITING_FOR_OTHER_GPU_WORK_WITHOUT_STOPPING_IT', detail=str(error))
                        time.sleep(60)
            log_path = prefix.with_suffix('.stages.log')
            status('RUNNING_REGISTERED_JOINT_DEVELOPMENT_EVALUATION', **bindings)
            run_stage('evaluate.py', evaluation_receipt, 'JOINT_EVALUATION_COMPLETE',
                {**bindings, 'review_sha256': sha256(review_path)}, log_path, hashes)
            status('RUNNING_REGISTERED_JOINT_STATISTICS_AND_CPU_ARCHIVE_AUDIT', **bindings)
            analysis_receipt = output_path(config, 'analysis') / 'development_0005000/completion.json'
            run_stage('analyze.py', analysis_receipt, 'JOINT_DEVELOPMENT_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE',
                {**bindings, 'evaluation_receipt_sha256': sha256(evaluation_receipt)}, log_path, hashes)
            verify_sources(hashes)
            status('REGISTERED_JOINT_TRIAL_EVALUATED_AND_ANALYZED_NOT_RESEARCH_COMPLETE', **bindings,
                evaluation_receipt_sha256=sha256(evaluation_receipt), analysis_receipt_sha256=sha256(analysis_receipt),
                new_training_branches_started=0)
        except BaseException as error:
            status('REGISTERED_JOINT_COMPLETION_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
