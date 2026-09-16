"""Wait only for the owned R2 trainer and audit its registered completed milestone."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(JOINT / 'scripts'), str(INNOVATION / 'src'), str(BASE / 'src')]

from finish_registered_trial import alive, owned_handle
from sufficiency.common import output_path, settings
from wetok_comm.common import PROJECT, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: owned R2 training -> complete-calibration/Adam/data audit; never starts more training')
        return
    config, original, grid, reference, base, parent = settings()
    if arguments.step not in config['new_full_steps']:
        raise ValueError('unregistered R2 milestone')
    trainer = owned_handle(arguments.training_pid, EXPERIMENT / 'scripts/train.py', {'--until-update': arguments.step})
    prefix = PROJECT / 'outputs' / f'WETOK-JOINT-SUFFICIENCY-R2-step{arguments.step:07d}'

    def status(value, **extra):
        write_json(prefix.with_suffix('.pipeline.json'), {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'training_pid': trainer[0], 'step': arguments.step, 'research_goal_complete': False, **extra})

    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            training = output_path(config, 'training')
            milestone = training / 'milestones' / f'step_{arguments.step:07d}.json'
            status('WAITING_FOR_OWNED_R2_TRAINING')
            while not milestone.exists():
                current = json.loads((training / 'status.json').read_text())
                if current['status'] == 'SUFFICIENCY_TRAINING_FAILED_OR_INTERRUPTED' or not alive(trainer):
                    raise RuntimeError(current)
                time.sleep(30)
            while alive(trainer):
                time.sleep(2)
            status('RUNNING_R2_CALIBRATION_AND_ADAM_REVIEW')
            with prefix.with_suffix('.review.log').open('a') as log:
                subprocess.run([sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts/review.py'), '--step', str(arguments.step), '--execute'],
                    cwd=EXPERIMENT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
            status('R2_REVIEW_READY_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('R2_PIPELINE_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
