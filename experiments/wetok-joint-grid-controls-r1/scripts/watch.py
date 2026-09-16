"""Watch an owned grid-control trainer and run only its completed milestone audit."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(JOINT / 'scripts'), str(INNOVATION / 'src'), str(BASE / 'src')]

from finish_registered_trial import alive, owned_handle
from grid_controls.common import output_path, settings
from wetok_comm.common import PROJECT, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: owned grid training -> completed full-calibration audit; no automatic new training')
        return
    config, original, reference, base, parent = settings()
    if arguments.step not in config['full_calibration_steps'] or arguments.step <= 0:
        raise ValueError('unregistered grid milestone')
    trainer = owned_handle(arguments.training_pid, EXPERIMENT / 'scripts/train.py', {'--until-update': arguments.step})
    prefix = PROJECT / 'outputs' / f'WETOK-JOINT-GRID-CONTROLS-R1-step{arguments.step:07d}'

    def status(value, **extra):
        write_json(prefix.with_suffix('.pipeline.json'), {'status': value, 'pid': os.getpid(),
            'training_pid': trainer[0], 'local_time': now(), 'step': arguments.step,
            'research_goal_complete': False, **extra})

    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            training = output_path(config, 'training')
            milestone = training / 'milestones' / f'step_{arguments.step:07d}.json'
            status('WAITING_FOR_OWNED_GRID_TRAINING')
            while not milestone.exists():
                current = json.loads((training / 'status.json').read_text())
                if current['status'] == 'JOINT_GRID_FAILED_OR_INTERRUPTED' or not alive(trainer):
                    raise RuntimeError(current)
                time.sleep(30)
            while alive(trainer):
                time.sleep(2)
            status('RUNNING_GRID_CALIBRATION_REVIEW')
            with prefix.with_suffix('.review.log').open('a') as log:
                subprocess.run([sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts/review.py'),
                    '--step', str(arguments.step), '--execute'], cwd=EXPERIMENT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            status('GRID_REVIEW_READY_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('GRID_PIPELINE_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
