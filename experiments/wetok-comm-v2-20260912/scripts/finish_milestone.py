"""Wait for an owned training milestone, then evaluate and analyze without extra training."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))
from wetok_comm.common import PROJECT, now, output_path, settings, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config = settings()
    if not arguments.execute:
        print('PLAN ONLY: wait for an owned milestone, then evaluate/analyze; do not extend training without review')
        return
    logs = PROJECT / 'outputs/WETOK-COMM-V2-20260912-LOGS'
    logs.mkdir(parents=True, exist_ok=True)
    training = output_path(config, 'training')
    stage = 'waiting_for_training_milestone'
    def status(value, **details):
        write_json(logs / f'milestone_{arguments.milestone:07d}_pipeline.json',
                   {'status': value, 'stage': stage, 'local_time': now(), 'pid': os.getpid(),
                    'milestone': arguments.milestone, 'research_goal_complete': False, **details})
    with (logs / f'milestone_{arguments.milestone:07d}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('this milestone already has a finishing supervisor')
        try:
            status('WAITING_FOR_OWNED_TRAINING')
            milestone = training / 'milestones' / f'step_{arguments.milestone:07d}.json'
            while not milestone.exists():
                current = json.loads((training / 'status.json').read_text())
                if current['status'] == 'TRAINING_FAILED_OR_INTERRUPTED':
                    raise RuntimeError(current)
                os.kill(arguments.training_pid, 0)
                time.sleep(30)
            while True:
                process = Path(f'/proc/{arguments.training_pid}/stat')
                if not process.exists() or process.read_text().split(') ', 1)[1].split()[0] == 'Z':
                    break
                time.sleep(2)
            for stage, script in (('evaluate', 'evaluate_milestone.py'), ('analyze', 'analyze_milestone.py')):
                status('RUNNING')
                with (logs / f'{stage}_{arguments.milestone:07d}.log').open('a') as handle:
                    command = [sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts' / script), '--milestone', str(arguments.milestone), '--execute']
                    subprocess.run(command, cwd=EXPERIMENT, stdout=handle, stderr=subprocess.STDOUT, check=True,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            stage = 'research_review'
            status('MILESTONE_REVIEW_READY_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('FAILED_NEEDS_ENGINEERING_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
