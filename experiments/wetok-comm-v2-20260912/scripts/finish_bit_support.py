"""Finish one owned bit-support milestone, leaving the research goal active."""

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
from wetok_comm.bit_support import load_repair, repair_output
from wetok_comm.common import PROJECT, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    repair, base, parent, parent_receipt = load_repair()
    if not arguments.execute:
        print('PLAN ONLY: finish this owned recipe milestone, then await scientific review')
        return
    training = repair_output(repair, 'training')
    logs = PROJECT / 'outputs/WETOK-COMM-V2-20260912-LOGS'
    logs.mkdir(parents=True, exist_ok=True)
    stage = 'waiting_for_recipe_training'
    def status(value, **extra):
        write_json(logs / f'bit_support_{arguments.milestone:07d}_pipeline.json', {'status': value, 'stage': stage,
                   'pid': os.getpid(), 'local_time': now(), 'additional_milestone': arguments.milestone,
                   'research_goal_complete': False, **extra})
    with (logs / f'bit_support_{arguments.milestone:07d}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('recipe finisher already active')
        try:
            status('WAITING_FOR_OWNED_TRAINING')
            milestone = training / 'milestones' / f'additional_{arguments.milestone:07d}.json'
            while not milestone.exists():
                current = json.loads((training / 'status.json').read_text())
                if current['status'] == 'REPAIR_FAILED_OR_INTERRUPTED':
                    raise RuntimeError(current)
                process = Path(f'/proc/{arguments.training_pid}/stat')
                if not process.exists() or process.read_text().split(') ', 1)[1].split()[0] == 'Z':
                    raise RuntimeError('training process exited without the milestone receipt')
                time.sleep(30)
            while True:
                process = Path(f'/proc/{arguments.training_pid}/stat')
                if not process.exists() or process.read_text().split(') ', 1)[1].split()[0] == 'Z':
                    break
                time.sleep(2)
            for stage, script in (('evaluate', 'evaluate_bit_support.py'), ('analyze', 'analyze_bit_support.py')):
                status('RUNNING')
                with (logs / f'bit_support_{stage}_{arguments.milestone:07d}.log').open('a') as handle:
                    command = [sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts' / script),
                               '--milestone', str(arguments.milestone), '--execute']
                    subprocess.run(command, cwd=EXPERIMENT, stdout=handle, stderr=subprocess.STDOUT, check=True,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            stage = 'research_review'
            status('RECIPE_REVIEW_READY_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('RECIPE_PIPELINE_FAILED_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
