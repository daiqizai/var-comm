"""Validate a stopped trial's actual checkpoint, preserve evidence, and restore its owned pipeline."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch

from evaluate_interfaces import require_uncontended_gpu
from innovation_comm.common import load_parent, new_system, output_path, settings
from wetok_comm.common import PROJECT, configure_torch, now, sha256, verify_sources, write_json
from wetok_comm.training import module_sha256, paired_batches


def require_no_trial_processes():
    scripts = {str(EXPERIMENT / 'scripts' / name).encode() for name in ('train.py', 'watch.py', 'finish_registered_trial.py')}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = set((process / 'cmdline').read_bytes().split(b'\0'))
            if command & scripts:
                raise RuntimeError(f'trial process {process.name} is still present; do not duplicate it')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--stamp', required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: audit a stopped checkpoint and restart the unchanged 5000-update pipeline')
        return
    config, base, parent_milestone = settings()
    if arguments.expected_step <= 0 or arguments.expected_step >= config['training']['planned_updates'] or not arguments.stamp.isdigit():
        raise ValueError('invalid checkpoint position or recovery date stamp')
    configure_torch()
    root, training = PROJECT / 'outputs', output_path(config, 'training')
    with (root / 'WETOK-INNOVATION-R1-recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_no_trial_processes()
        require_uncontended_gpu()
        bindings = [training / 'metadata.json', output_path(config, 'profile') / 'profile.json',
                    root / 'WETOK-INNOVATION-R1-step0005000-finish.pipeline.json']
        for path in bindings:
            verify_sources(json.loads(path.read_text())['source_hashes'])
        resume_path = training / 'resume.pt'
        if sha256(resume_path) != arguments.expected_sha256:
            raise RuntimeError('checkpoint changed since the recovery inspection')
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        completed = saved['completed_receiver_updates']
        if completed != arguments.expected_step or saved['global_data_step'] != 7000 + completed:
            raise RuntimeError('actual checkpoint update position differs')
        profile = json.loads((output_path(config, 'profile') / 'profile.json').read_text())
        if saved['frozen'] != profile['frozen'] or set(saved['models']) != set(config['variants']):
            raise RuntimeError('checkpoint frozen models or six-arm membership changed')
        parent = load_parent(config, base, 'cpu')
        for name in config['variants']:
            system = new_system(parent, name, config, 'cpu')
            system.load_state_dict(saved['models'][name], strict=True)
            if module_sha256(system.encoder) != saved['frozen']['encoder']:
                raise RuntimeError('receiver checkpoint changed its frozen sender')
            if len(saved['histories'][name]) != completed or any(int(state['step']) != completed for state in saved['optimizers'][name]['state'].values()):
                raise RuntimeError('model/history/Adam positions are not paired')
            for value in saved['models'][name].values():
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise RuntimeError('checkpoint model contains nonfinite values')
            for state in saved['optimizers'][name]['state'].values():
                for value in state.values():
                    if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                        raise RuntimeError('checkpoint optimizer contains nonfinite values')
            choice = saved['selected'][name]
            if sha256(training / choice['checkpoint']) != choice['checkpoint_sha256']:
                raise RuntimeError('previously selected full-calibration model changed')
            del system
        for index, batch in enumerate(paired_batches(base, 20000, 7000, 7000 + completed)):
            observations = set()
            for name in config['variants']:
                row = saved['histories'][name][index]
                if row['step'] != index + 1 or row['global_data_step'] != batch['step'] + 1 or row['batch_sha256'] != batch['fingerprint']:
                    raise RuntimeError('saved source/augmentation/SNR/noise order differs from its registered history')
                if not np.isfinite(row['loss']):
                    raise RuntimeError('saved training objective is nonfinite')
                observations.add(row['shared_received_sha256'])
            if len(observations) != 1:
                raise RuntimeError('the saved six-arm history did not share actual observations')
        recovery = root / f'WETOK-INNOVATION-R1-RECOVERY-{arguments.stamp}'
        recovery.mkdir(parents=True, exist_ok=False)
        checkpoint_link = recovery / f'resume_from_{completed:07d}.pt'
        os.link(resume_path, checkpoint_link)
        archived = []
        candidates = [training / 'status.json', root / 'WETOK-INNOVATION-R1-step0005000.pipeline.json',
            root / 'WETOK-INNOVATION-R1-step0005000-finish.pipeline.json', root / 'WETOK-INNOVATION-R1-step0005000.launch.json',
            root / 'WETOK-INNOVATION-R1-train-0005000.log', root / 'WETOK-INNOVATION-R1-watch-0005000.log',
            root / 'WETOK-INNOVATION-R1-finish-0005000.log']
        for path in candidates:
            if path.exists():
                destination = recovery / 'previous_status_and_logs' / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                archived.append({'source': str(path), 'archive': str(destination), 'sha256': sha256(destination)})
        report = {'status': 'STOPPED_CHECKPOINT_RECOVERY_AUDIT_PASS', 'local_time': now(),
            'checkpoint_step': completed, 'global_data_step': 7000 + completed, 'checkpoint_sha256': arguments.expected_sha256,
            'retained_checkpoint_hardlink': str(checkpoint_link), 'calibrated': saved['calibrated'],
            'all_models_Adam_source_noise_shared_y_and_frozen_E_verified': True,
            'previous_observed_elapsed_seconds': saved['elapsed_seconds'],
            'unobserved_work_after_last_save_not_reconstructed': True, 'downtime_is_not_GPU_training_time': True,
            'current_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'current_boot_started': subprocess.check_output(['uptime', '-s'], text=True).strip(),
            'archived': archived, 'script_sha256': sha256(Path(__file__)), 'research_goal_complete': False}
        write_json(recovery / 'audit.json', report)
        require_no_trial_processes()
        require_uncontended_gpu()
        if sha256(resume_path) != arguments.expected_sha256:
            raise RuntimeError('checkpoint changed during the audited recovery')
        handles = {}
        commands = [('trainer', ['train.py', '--until-update', '5000', '--resume', '--execute'])]
        for role, command in commands:
            with (recovery / f'{role}.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts' / command[0]), *command[1:]],
                    cwd=EXPERIMENT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            handles[role] = process.pid
        write_json(recovery / 'launch.json', {'created_local': now(), **handles, 'audit_sha256': sha256(recovery / 'audit.json')})
        followups = [('review_watcher', ['watch.py', '--training-pid', str(handles['trainer']), '--step', '5000', '--execute']),
                     ('finisher', ['finish_registered_trial.py', '--training-pid', str(handles['trainer']), '--review-watcher-pid', 'PENDING', '--execute'])]
        for role, command in followups:
            if role == 'finisher':
                command[command.index('PENDING')] = str(handles['review_watcher'])
            with (recovery / f'{role}.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', '-B', str(EXPERIMENT / 'scripts' / command[0]), *command[1:]],
                    cwd=EXPERIMENT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            handles[role] = process.pid
            write_json(recovery / 'launch.json', {'created_local': now(), **handles, 'audit_sha256': sha256(recovery / 'audit.json')})
        print(json.dumps({'recovery': str(recovery), 'from_actual_step': completed, 'until_step': 5000, **handles}, indent=2))


if __name__ == '__main__':
    main()
