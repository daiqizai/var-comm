"""Wait for one owned training milestone and run its CPU-only calibration review."""

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
from wetok_comm.interface_study import interface_output, load_interface_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    study, base, parent = load_interface_study()
    if not arguments.execute:
        print('PLAN ONLY: wait for owned trainer, audit the checkpoint, no extra GPU or development evaluation')
        return
    training = interface_output(study, 'training')
    logs = PROJECT / 'outputs/WETOK-COMM-V2-20260912-LOGS'
    path = logs / f'interface_{arguments.milestone:07d}_pipeline.json'
    process = Path(f'/proc/{arguments.training_pid}')
    command = (process / 'cmdline').read_bytes().split(b'\0')
    if str(EXPERIMENT / 'scripts/train_interfaces.py').encode() not in command or process.stat().st_uid != os.getuid():
        raise RuntimeError('watcher target is not the owned interface trainer')
    started = (process / 'stat').read_text().split(') ', 1)[1].split()[19]
    def alive():
        try:
            fields = (process / 'stat').read_text().split(') ', 1)[1].split()
            return fields[0] != 'Z' and fields[19] == started
        except FileNotFoundError:
            return False
    def status(value, **extra):
        write_json(path, {'status': value, 'pid': os.getpid(), 'training_pid': arguments.training_pid,
            'milestone': arguments.milestone, 'local_time': now(), 'research_goal_complete': False, **extra})
    with (logs / f'interface_{arguments.milestone:07d}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('WAITING_FOR_OWNED_INTERFACE_TRAINING')
            milestone = training / 'milestones' / f'additional_{arguments.milestone:07d}.json'
            while not milestone.exists():
                current = json.loads((training / 'status.json').read_text())
                if current['status'] == 'INTERFACE_FAILED_OR_INTERRUPTED' or not alive():
                    raise RuntimeError(current)
                time.sleep(30)
            while alive():
                time.sleep(2)
            status('RUNNING_CPU_CALIBRATION_REVIEW')
            with (logs / f'interface_review_{arguments.milestone:07d}.log').open('a') as handle:
                subprocess.run([sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts/review_interface_milestone.py'),
                    '--milestone', str(arguments.milestone), '--execute'], stdout=handle, stderr=subprocess.STDOUT,
                    cwd=EXPERIMENT, check=True)
            status('INTERFACE_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE')
        except BaseException as error:
            status('INTERFACE_PIPELINE_NEEDS_REVIEW', error=repr(error))
            raise


if __name__ == '__main__':
    main()
