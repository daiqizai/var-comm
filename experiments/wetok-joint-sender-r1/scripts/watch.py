"""Watch only the owned Joint trainer and review its completed calibration milestone."""

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
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from joint_sender.common import output_path, settings
from wetok_comm.common import PROJECT, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, reference, base, parent = settings()
    if not arguments.execute:
        print('PLAN ONLY: owned Joint trainer -> full calibration audit, no automatic new research branch')
        return
    if arguments.step not in config['full_calibration_steps']:
        raise ValueError('unregistered Joint milestone')
    process = Path(f'/proc/{arguments.training_pid}')
    if (process.stat().st_uid != os.getuid() or
        str(EXPERIMENT / 'scripts/train.py').encode() not in (process / 'cmdline').read_bytes().split(b'\0')):
        raise RuntimeError('watcher target is not the owned Joint trainer')
    started = (process / 'stat').read_text().split(') ', 1)[1].split()[19]
    def alive():
        try:
            fields = (process / 'stat').read_text().split(') ', 1)[1].split()
            return fields[0] != 'Z' and fields[19] == started
        except FileNotFoundError:
            return False
    prefix = PROJECT / 'outputs' / f'WETOK-JOINT-SENDER-R1-step{arguments.step:07d}'
    def status(value, **extra):
        write_json(prefix.with_suffix('.pipeline.json'), {'status': value, 'pid': os.getpid(), 'training_pid': arguments.training_pid,
            'local_time': now(), 'step': arguments.step, 'research_goal_complete': False, **extra})
    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            training = output_path(config, 'training')
            milestone = training / 'milestones' / f'step_{arguments.step:07d}.json'
            status('WAITING_FOR_OWNED_JOINT_TRAINING')
            while not milestone.exists():
                state_path = training / 'status.json'
                current = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'WAITING_FOR_FIRST_TRAINER_STATUS'}
                if current['status'] == 'JOINT_SENDER_FAILED_OR_INTERRUPTED' or not alive():
                    raise RuntimeError(current)
                time.sleep(30)
            while alive():
                time.sleep(2)
            status('RUNNING_JOINT_CALIBRATION_REVIEW')
            with prefix.with_suffix('.review.log').open('a') as handle:
                subprocess.run([sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts/review.py'), '--step', str(arguments.step), '--execute'],
                    cwd=EXPERIMENT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            status('JOINT_REVIEW_READY_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('JOINT_PIPELINE_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
