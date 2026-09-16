"""Paired receiver-only learning on exactly shared frozen-transmitter observations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src'), str(REFERENCE / 'scripts')]

import numpy as np
import torch

from evaluate_interfaces import require_uncontended_gpu
from train_milestone import choose, write_csv
from innovation_comm.common import load_parent, new_system, output_path, settings
from innovation_comm.runtime import actual_observation, calibrate, receiver_loss
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, monitoring_indices, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until-update', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, base, parent_milestone = settings()
    until = arguments.until_update
    if until not in config['calibration']['full_steps'] or until <= 0:
        raise ValueError('use a registered full-calibration receiver milestone')
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'variants': config['variants'], 'new_updates': until, 'TX': 'same frozen parent'}, indent=2))
        return
    require_uncontended_gpu()
    configure_torch()
    profile_path = output_path(config, 'profile') / 'profile.json'
    profile = json.loads(profile_path.read_text())
    if profile['status'] != 'INNOVATION_GRADIENT_PROFILE_PASS' or profile['optimizer_updates'] != 0:
        raise RuntimeError('receiver information/gradient checks have not passed')
    verify_sources(profile['source_hashes'])
    output = output_path(config, 'training')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/protocol.md',
                 *sorted((EXPERIMENT / 'src/innovation_comm').glob('*.py')),
                 REFERENCE / 'src/wetok_comm/interface_study.py', REFERENCE / 'src/wetok_comm/training.py',
                 REFERENCE / 'src/wetok_comm/native.py', REFERENCE / 'src/wetok_comm/model.py']
        metadata = {'created_local': now(), 'source_hashes': snapshot(output, files), 'profile_sha256': sha256(profile_path),
                    'parent_model_sha256': config['parent_model_sha256']}
        write_json(output / 'metadata.json', metadata)
    if metadata['profile_sha256'] != sha256(profile_path):
        raise RuntimeError('receiver profile changed')
    completed, started, elapsed_prior = 0, time.time(), 0.
    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'completed_receiver_updates': completed, 'global_data_step': 7000 + completed,
            'milestone': until, 'research_goal_complete': False, **extra})
    try:
        status('LOADING_FIXED_TX_RECEIVERS')
        device = torch.device('cuda:0')
        parent = load_parent(config, base, device)
        systems = {name: new_system(parent, name, config, device) for name in config['variants']}
        decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'encoder': module_sha256(parent.encoder), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
        if frozen != profile['frozen']:
            raise RuntimeError('frozen parent or image models changed')
        optimizers = {name: torch.optim.AdamW([value for value in system.parameters() if value.requires_grad],
            lr=config['training']['learning_rate'], betas=tuple(base['training']['betas']), weight_decay=base['training']['weight_decay'])
            for name, system in systems.items()}
        histories, selected = {name: [] for name in systems}, {name: None for name in systems}
        calibrated, summaries = [], []
        timings = {'training': {name: 0. for name in systems}, 'calibration': {name: 0. for name in systems}, 'shared_TX': 0.}
        if arguments.resume:
            saved = torch.load(output / 'resume.pt', map_location=device, weights_only=True)
            completed = saved['completed_receiver_updates']
            if completed >= until or saved['frozen'] != frozen:
                raise RuntimeError('invalid fixed-TX receiver resume')
            for name in systems:
                systems[name].load_state_dict(saved['models'][name], strict=True)
                optimizers[name].load_state_dict(saved['optimizers'][name])
            histories, selected, calibrated = saved['histories'], saved['selected'], saved['calibrated']
            summaries, timings, elapsed_prior = saved['summaries'], saved['timings'], saved['elapsed_seconds']
            del saved
        else:
            write_json(output / 'initialization.json', {'parent_receiver_hash': module_sha256(parent.receiver), 'frozen': frozen,
                'model_hashes': {name: module_sha256(system) for name, system in systems.items()},
                'trainable_parameters': {name: sum(value.numel() for value in system.parameters() if value.requires_grad) for name, system in systems.items()},
                'all_receivers_fresh_Adam': True, 'source_transmitter_shared_in_value': True})
        for name in systems:
            (output / name / 'checkpoints').mkdir(parents=True, exist_ok=True)
        population, calibration = read_population(base, 'train'), read_population(base, 'calibration')
        monitor = monitoring_indices(calibration[2], base)

        def persist():
            for name in systems:
                write_csv(output / name / 'training.csv', histories[name])
                write_json(output / name / 'selected.json', selected[name])
            write_csv(output / 'calibration_summary.csv', summaries)
            state = {'completed_receiver_updates': completed, 'global_data_step': 7000 + completed,
                'models': {name: system.state_dict() for name, system in systems.items()},
                'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                'histories': histories, 'selected': selected, 'calibrated': calibrated, 'summaries': summaries,
                'timings': timings, 'frozen': frozen, 'elapsed_seconds': elapsed_prior + time.time() - started}
            temporary = output / 'resume.pending.pt'
            torch.save(state, temporary)
            temporary.replace(output / 'resume.pt')

        def calibration_point():
            if completed in calibrated or completed % config['calibration']['monitor_every']:
                return
            full = completed in config['calibration']['full_steps']
            scope = 'full' if full else 'monitor'
            for name, system in systems.items():
                status('CALIBRATING_FIXED_TX_RECEIVERS', variant=name, scope=scope)
                tick = time.perf_counter()
                rows = calibrate(system, parent, calibration, decoder, perceptual, config, base, device, None if full else monitor)
                write_csv(output / name / f'{scope}_{completed:07d}.csv', rows)
                metrics = [key for key in rows[0] if key not in ('image_id', 'snr_db') and rows[0][key] != '']
                means = {key: float(np.mean([row[key] for row in rows])) for key in metrics}
                for snr in base['channel']['snrs_db']:
                    subset = [row for row in rows if row['snr_db'] == snr]
                    summaries.append({'step': completed, 'variant': name, 'scope': scope, 'snr_db': snr, 'source_images': len(subset),
                        **{key: (float(np.mean([row[key] for row in subset])) if key in metrics else '') for key in rows[0] if key not in ('image_id', 'snr_db')}})
                if full:
                    path = output / name / 'checkpoints' / f'step_{completed:07d}.pt'
                    torch.save({'model': system.state_dict(), 'variant': name, 'step': completed}, path)
                    selected[name] = choose(selected[name], {'step': completed, 'scope': 'full', 'source_images': 1000,
                        'checkpoint': str(path.relative_to(output)), 'checkpoint_sha256': sha256(path), **means})
                timings['calibration'][name] += time.perf_counter() - tick
                print(f'{scope} {name} step={completed} LPIPS={means["lpips"]:.6f} PSNR={means["psnr_db"]:.5f}', flush=True)
            calibrated.append(completed)
            persist()

        calibration_point()
        persist()
        status('TRAINING_FIXED_TX_RECEIVERS')
        for batch in paired_batches(base, 20000, 7000 + completed, 7000 + until):
            if batch['step'] != 7000 + completed:
                raise RuntimeError('fixed-TX paired sampler moved')
            inputs = batch_inputs(population, batch, device)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            signal, observed = actual_observation(parent, inputs)
            torch.cuda.synchronize()
            timings['shared_TX'] += time.perf_counter() - tick
            if float((signal.square().sum(-1).mean(-1) - 2).abs().max()) > 1e-5:
                raise RuntimeError('common frozen transmitter violates N/E')
            observed_hash = hashlib.sha256(observed.cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, system in systems.items():
                system.train().zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                totals = {key: 0. for key in ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate')}
                for index in range(4):
                    part = {key: value[index:index + 1] for key, value in inputs.items()}
                    objective, components, diagnostics, result = receiver_loss(system, observed[index:index + 1], part, decoder, perceptual, config['training']['weights'])
                    if not bool(torch.isfinite(objective)):
                        raise FloatingPointError('nonfinite receiver objective')
                    (objective / 4).backward()
                    totals['loss'] += float(objective.detach()) / 4
                    for key, value in components.items():
                        totals[key] += float(value.detach().sum()) / 4
                    totals['bit_error_rate'] += float(diagnostics['bit_error_rate'].detach().sum()) / 4
                    del objective, components, diagnostics, result, part
                trainable = [value for value in system.parameters() if value.requires_grad]
                norm = float(torch.nn.utils.clip_grad_norm_(trainable, config['training']['gradient_clip'], error_if_nonfinite=True))
                if any(value.grad is not None for value in system.encoder.parameters()):
                    raise RuntimeError('receiver training leaked gradients into the frozen transmitter parameters')
                optimizers[name].step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                timings['training'][name] += seconds
                histories[name].append({'step': completed + 1, 'global_data_step': batch['step'] + 1,
                    'batch_sha256': batch['fingerprint'], 'shared_received_sha256': observed_hash,
                    'gradient_norm': norm, 'step_seconds': seconds, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), **totals})
            completed += 1
            if completed == 1 or completed % config['training']['autosave_every'] == 0:
                persist()
                status('TRAINING_FIXED_TX_RECEIVERS')
                print(f'fixed-TX paired updates {completed}/{until} ' + ' '.join(f'{name}={histories[name][-1]["loss"]:.6f}' for name in systems), flush=True)
            calibration_point()
        if completed != until or completed not in calibrated:
            raise RuntimeError('receiver milestone incomplete')
        for name, system in systems.items():
            if module_sha256(system.encoder) != frozen['encoder']:
                raise RuntimeError('a copied frozen transmitter changed during receiver learning')
            if any(int(state['step']) != completed for state in optimizers[name].state.values()):
                raise RuntimeError('receiver optimizer update budgets differ')
        if frozen != {'encoder': module_sha256(parent.encoder), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('frozen shared models changed')
        verify_sources(metadata['source_hashes'])
        persist()
        milestones = output / 'milestones'
        milestones.mkdir(exist_ok=True)
        checkpoint = milestones / f'step_{completed:07d}_optimizer.pt'
        os.link(output / 'resume.pt', checkpoint)
        write_json(milestones / f'step_{completed:07d}.json', {'status': 'INNOVATION_MILESTONE_COMPLETE', 'completed_local': now(),
            'receiver_updates_per_arm': completed, 'selected': selected, 'frozen': frozen, 'timings': timings,
            'source_hashes': metadata['source_hashes'], 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'observed_wall_GPU_hours': (elapsed_prior + time.time() - started) / 3600, 'research_goal_complete': False})
        status('INNOVATION_MILESTONE_READY_NOT_RESEARCH_COMPLETE')
    except BaseException as error:
        status('INNOVATION_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
