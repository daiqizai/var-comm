"""Observe only the owned geometry trainer and finish its CPU calibration audit."""

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

from wetok_comm.common import PROJECT, now, write_json
from wetok_comm.geometry_study import geometry_output, load_geometry_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--total', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, base, qualification = load_geometry_study()
    if arguments.total - 2000 not in config['calibration']['full_image_steps']:
        raise ValueError('no matched control calibration at this image update budget')
    if not arguments.execute:
        print('PLAN ONLY: watch the owned geometry milestone, then run matched CPU review')
        return
    process = Path(f'/proc/{arguments.training_pid}')
    if (str(EXPERIMENT / 'scripts/train_geometry.py').encode() not in (process / 'cmdline').read_bytes().split(b'\0')
        or process.stat().st_uid != os.getuid()):
        raise RuntimeError('watcher target is not the owned geometry trainer')
    started = (process / 'stat').read_text().split(') ', 1)[1].split()[19]
    def alive():
        try:
            fields = (process / 'stat').read_text().split(') ', 1)[1].split()
            return fields[0] != 'Z' and fields[19] == started
        except FileNotFoundError:
            return False
    logs = PROJECT / 'outputs/WETOK-COMM-V2-20260912-LOGS'
    path = logs / f'geometry_total_{arguments.total:07d}_pipeline.json'
    def status(value, **extra):
        write_json(path, {'status': value, 'pid': os.getpid(), 'training_pid': arguments.training_pid,
            'total_milestone': arguments.total, 'local_time': now(), 'research_goal_complete': False, **extra})
    with (logs / f'geometry_total_{arguments.total:07d}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            training = geometry_output(config, 'training')
            milestone = training / 'milestones' / f'total_{arguments.total:07d}.json'
            status('WAITING_FOR_OWNED_GEOMETRY_TRAINING')
            while not milestone.exists():
                current = json.loads((training / 'status.json').read_text())
                if current['status'] == 'GEOMETRY_FAILED_OR_INTERRUPTED' or not alive():
                    raise RuntimeError(current)
                time.sleep(30)
            while alive():
                time.sleep(2)
            status('RUNNING_MATCHED_CALIBRATION_REVIEW')
            with (logs / f'geometry_review_total{arguments.total:07d}.log').open('a') as handle:
                subprocess.run([sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts/review_geometry.py'),
                    '--total', str(arguments.total), '--execute'], cwd=EXPERIMENT,
                    stdout=handle, stderr=subprocess.STDOUT, check=True)
            status('GEOMETRY_REVIEW_READY_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('GEOMETRY_PIPELINE_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
