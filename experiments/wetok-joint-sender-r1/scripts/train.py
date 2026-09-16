"""Matched Joint E/R continuation; requires completed R-only controls and a zero-update GPU profile."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch

from evaluate_interfaces import require_uncontended_gpu
from train_milestone import choose, write_csv
from joint_sender.common import complete_control, initial_system, load_parent, output_path, settings
from joint_sender.runtime import backward_paired_batch, calibration
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, monitoring_indices, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until-update', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, reference, base, parent_milestone = settings()
    until = arguments.until_update
    if until not in config['full_calibration_steps'] or until <= 0:
        raise ValueError('use a registered complete-calibration Joint milestone')
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'variants': config['variants'], 'new_joint_updates': until,
            'requires_completed_frozen_TX_control_updates': 5000, 'requires_real_zero_update_GPU_profile': True,
            'does_not_interrupt_or_start_alongside_current_receiver_trial': True}, indent=2))
        return
    require_uncontended_gpu()
    configure_torch()
    controls, control_path, review_path = complete_control(config, reference)
    profile_path = output_path(config, 'profile') / 'profile.json'
    profile = json.loads(profile_path.read_text())
    if profile['status'] != 'JOINT_IMAGE_GRADIENT_PROFILE_PASS' or profile['optimizer_updates'] != 0:
        raise RuntimeError('real image-only Joint sender/receiver gradients have not been checked')
    verify_sources(profile['source_hashes'])
    output = output_path(config, 'training')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        control_root = control_path.parent.parent
        control_inputs = [control_root / 'initialization.json',
            *[control_root / name / 'training.csv' for name in config['variants']],
            *[control_root / name / f'full_{step:07d}.csv' for name in config['variants'] for step in config['full_calibration_steps']]]
        control_hashes = {str(path): sha256(path) for path in control_inputs}
        files = [Path(__file__), EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/protocol.md',
            *sorted((EXPERIMENT / 'src/joint_sender').glob('*.py')),
            INNOVATION / 'src/innovation_comm/model.py', INNOVATION / 'src/innovation_comm/runtime.py',
            BASE / 'src/wetok_comm/interface_study.py', BASE / 'src/wetok_comm/training.py', BASE / 'src/wetok_comm/native.py']
        metadata = {'created_local': now(), 'source_hashes': snapshot(output, files), 'profile_sha256': sha256(profile_path),
            'control_milestone_sha256': sha256(control_path), 'control_review_sha256': sha256(review_path),
            'control_input_hashes': control_hashes,
            'parent_checkpoint_sha256': reference['parent_model_sha256'], 'shared_standard_noise_not_shared_learned_waveform': True}
        write_json(output / 'metadata.json', metadata)
    if (metadata['profile_sha256'] != sha256(profile_path) or metadata['control_milestone_sha256'] != sha256(control_path) or
        metadata['control_review_sha256'] != sha256(review_path)):
        raise RuntimeError('Joint control qualification or profile changed')
    for path, expected in metadata['control_input_hashes'].items():
        if sha256(path) != expected:
            raise RuntimeError('frozen control history/calibration changed during the Joint trial')
    completed, started, elapsed_prior = 0, time.time(), 0.

    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'completed_joint_updates': completed, 'global_data_step': 7000 + completed, 'milestone': until,
            'research_goal_complete': False, **extra})

    try:
        status('LOADING_MATCHED_JOINT_COMMUNICATION')
        device = torch.device('cuda:0')
        parent = load_parent(reference, base, device)
        systems = {name: initial_system(parent, name, reference, device) for name in config['variants']}
        decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
        if frozen != profile['frozen']:
            raise RuntimeError('frozen Joint visual models or shared parent changed')
        optimizers = {name: torch.optim.AdamW([value for value in system.parameters() if value.requires_grad],
            lr=config['learning_rate'], betas=tuple(base['training']['betas']), weight_decay=base['training']['weight_decay'])
            for name, system in systems.items()}
        histories, selected = {name: [] for name in systems}, {name: None for name in systems}
        calibrated, summaries = [], []
        timings = {'training': {name: 0. for name in systems}, 'calibration': {name: 0. for name in systems}}
        if arguments.resume:
            saved = torch.load(output / 'resume.pt', map_location=device, weights_only=True)
            completed = saved['completed_joint_updates']
            if completed >= until or saved['frozen'] != frozen or saved['control_milestone_sha256'] != sha256(control_path):
                raise RuntimeError('invalid actual-endpoint Joint resume')
            for name in systems:
                systems[name].load_state_dict(saved['models'][name], strict=True)
                optimizers[name].load_state_dict(saved['optimizers'][name])
            histories, selected = saved['histories'], saved['selected']
            calibrated, summaries, timings = saved['calibrated'], saved['summaries'], saved['timings']
            elapsed_prior = saved['elapsed_seconds']
        else:
            initial_hashes = {name: module_sha256(system) for name, system in systems.items()}
            if initial_hashes != profile['model_hashes_before']:
                raise RuntimeError('Joint training did not start from the profiled shared weights')
            write_json(output / 'initialization.json', {'model_hashes': initial_hashes, 'frozen': frozen,
                'all_communication_parameters_fresh_Adam': True, 'control_opportunity_updates': 5000,
                'profile_sha256': sha256(profile_path)})
        for name in systems:
            (output / name / 'checkpoints').mkdir(parents=True, exist_ok=True)
        population, calibration_population = read_population(base, 'train'), read_population(base, 'calibration')
        monitor = monitoring_indices(calibration_population[2], base)

        def persist():
            for name in systems:
                write_csv(output / name / 'training.csv', histories[name])
                write_json(output / name / 'selected.json', selected[name])
            write_csv(output / 'calibration_summary.csv', summaries)
            state = {'completed_joint_updates': completed, 'global_data_step': 7000 + completed,
                'models': {name: system.state_dict() for name, system in systems.items()},
                'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                'histories': histories, 'selected': selected, 'calibrated': calibrated, 'summaries': summaries,
                'timings': timings, 'frozen': frozen, 'control_milestone_sha256': sha256(control_path),
                'elapsed_seconds': elapsed_prior + time.time() - started}
            temporary = output / 'resume.pending.pt'
            torch.save(state, temporary)
            temporary.replace(output / 'resume.pt')

        def calibration_point():
            if completed in calibrated or completed % config['monitor_every']:
                return
            full = completed in config['full_calibration_steps']
            scope = 'full' if full else 'monitor'
            for name, system in systems.items():
                status('CALIBRATING_MATCHED_JOINT_COMMUNICATION', variant=name, scope=scope)
                tick = time.perf_counter()
                rows = calibration(system, calibration_population, decoder, perceptual, reference, base, device, None if full else monitor)
                write_csv(output / name / f'{scope}_{completed:07d}.csv', rows)
                metrics = [key for key in rows[0] if key not in ('image_id', 'snr_db') and rows[0][key] != '']
                means = {key: float(np.mean([row[key] for row in rows])) for key in metrics}
                for snr in base['channel']['snrs_db']:
                    subset = [row for row in rows if row['snr_db'] == snr]
                    summaries.append({'step': completed, 'variant': name, 'scope': scope, 'snr_db': snr, 'source_images': len(subset),
                        **{key: (float(np.mean([row[key] for row in subset])) if key in metrics else '')
                           for key in rows[0] if key not in ('image_id', 'snr_db')}})
                if full:
                    path = output / name / 'checkpoints' / f'step_{completed:07d}.pt'
                    torch.save({'model': system.state_dict(), 'variant': name, 'step': completed, 'encoder_trained': True}, path)
                    selected[name] = choose(selected[name], {'step': completed, 'scope': 'full', 'source_images': 1000,
                        'checkpoint': str(path.relative_to(output)), 'checkpoint_sha256': sha256(path), **means})
                timings['calibration'][name] += time.perf_counter() - tick
                print(f'{scope} Joint {name} step={completed} LPIPS={means["lpips"]:.6f} PSNR={means["psnr_db"]:.5f}', flush=True)
            calibrated.append(completed)
            persist()

        calibration_point()
        persist()
        status('TRAINING_MATCHED_JOINT_COMMUNICATION')
        for batch in paired_batches(base, 20000, 7000 + completed, 7000 + until):
            if batch['step'] != 7000 + completed:
                raise RuntimeError('Joint/frozen paired data sampler moved')
            inputs = batch_inputs(population, batch, device)
            common_noise_hashes = set()
            for name, system in systems.items():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                totals = backward_paired_batch(system, inputs, decoder, perceptual, config['unchanged_loss'])
                trainable = [value for value in system.parameters() if value.requires_grad]
                norm = float(torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'], error_if_nonfinite=True))
                optimizers[name].step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                common_noise_hashes.add(totals['paired_standard_noise_sha256'])
                timings['training'][name] += seconds
                histories[name].append({'step': completed + 1, 'global_data_step': batch['step'] + 1,
                    'batch_sha256': batch['fingerprint'], 'gradient_norm': norm, 'step_seconds': seconds,
                    'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), **totals})
            if len(common_noise_hashes) != 1:
                raise RuntimeError('Joint variants did not use the same actual standard noise')
            completed += 1
            if completed == 1 or completed % 100 == 0:
                persist()
                status('TRAINING_MATCHED_JOINT_COMMUNICATION')
                print(f'Joint paired updates {completed}/{until} ' + ' '.join(f'{name}={histories[name][-1]["loss"]:.6f}' for name in systems), flush=True)
            calibration_point()
        if completed != until or completed not in calibrated:
            raise RuntimeError('Joint milestone is incomplete')
        for name in systems:
            if any(int(state['step']) != completed for state in optimizers[name].state.values()):
                raise RuntimeError('Joint optimizer update budgets differ')
        if frozen != {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('Joint continuation changed a frozen visual model or parent')
        verify_sources(metadata['source_hashes'])
        persist()
        milestones = output / 'milestones'
        milestones.mkdir(exist_ok=True)
        checkpoint = milestones / f'step_{completed:07d}_optimizer.pt'
        os.link(output / 'resume.pt', checkpoint)
        write_json(milestones / f'step_{completed:07d}.json', {'status': 'JOINT_SENDER_MILESTONE_COMPLETE', 'completed_local': now(),
            'joint_updates_per_arm': completed, 'selected': selected, 'frozen': frozen, 'timings': timings,
            'source_hashes': metadata['source_hashes'], 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'control_milestone_sha256': sha256(control_path), 'control_review_sha256': sha256(review_path),
            'observed_wall_GPU_hours': (elapsed_prior + time.time() - started) / 3600, 'research_goal_complete': False})
        status('JOINT_SENDER_MILESTONE_READY_NOT_RESEARCH_COMPLETE')
    except BaseException as error:
        status('JOINT_SENDER_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
