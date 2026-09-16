"""Wait for an explicitly owned R3 trainer and audit its completed milestone, without starting another training."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT.parent / 'wetok-joint-sender-r1/scripts'),
    str(EXPERIMENT.parent / 'wetok-comm-v2-20260912/scripts')]

from finish_registered_trial import alive, owned_handle
from vector_control.common import output_path, settings
from wetok_comm.common import PROJECT, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: owned R3 trainer -> complete milestone audit; no new training or inference')
        return
    config, r2, original, grid, reference, base, parent_record = settings()
    if arguments.step not in config['full_calibration_steps'] or arguments.step <= 0:
        raise ValueError('unregistered R3 milestone')
    handle = owned_handle(arguments.training_pid, EXPERIMENT / 'scripts/train.py', {'--until-update': arguments.step})
    prefix = PROJECT / 'outputs' / f'WETOK-REENCODING-VECTOR-CONTROL-R3-step{arguments.step:07d}'

    def status(value, **extra):
        write_json(prefix.with_suffix('.pipeline.json'), {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'training_pid': handle[0], 'step': arguments.step, 'research_goal_complete': False, **extra})

    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('WAITING_FOR_OWNED_R3_TRAINING')
            root = output_path(config, 'training')
            milestone = root / 'milestones' / f'step_{arguments.step:07d}.json'
            while not milestone.exists():
                state_path = root / 'status.json'
                current = json.loads(state_path.read_text()) if state_path.exists() else {}
                if current.get('status') == 'R3_TRAINING_FAILED_OR_INTERRUPTED' or not alive(handle):
                    raise RuntimeError(current)
                time.sleep(20)
            while alive(handle):
                time.sleep(2)
            status('RUNNING_R3_CALIBRATION_AND_HISTORY_REVIEW')
            with prefix.with_suffix('.review.log').open('a') as log:
                subprocess.run([sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts/review.py'), '--step', str(arguments.step), '--execute'],
                    cwd=EXPERIMENT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
            status('R3_REVIEW_READY_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('R3_PIPELINE_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
