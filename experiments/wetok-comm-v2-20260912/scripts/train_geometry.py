"""Train only the new geometry, using qualified immutable controls for comparison."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np
import torch

from evaluate_interfaces import require_uncontended_gpu
from train_milestone import choose, write_csv
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.geometry_study import geometry_network, geometry_output, load_geometry_study
from wetok_comm.interface_study import interface_calibrate, interface_losses
from wetok_comm.model import communicate
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, monitoring_indices, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until-total', type=int, default=3000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, base, qualification = load_geometry_study()
    until = arguments.until_total
    if until < 2000 or (until - 2000) % 500:
        raise ValueError('use a complete registered calibration boundary')
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'new_geometry': config['candidate_geometry'], 'variants': config['variants'],
                          'until_total': until, 'image_updates': until - 2000, 'controls': 'qualified historical 153x40'}, indent=2))
        return
    require_uncontended_gpu()
    configure_torch()
    profile_path = geometry_output(config, 'profile') / 'profile.json'
    profile = json.loads(profile_path.read_text())
    if profile['status'] != 'GEOMETRY_GRADIENT_PROFILE_PASS' or profile['optimizer_updates'] != 0:
        raise RuntimeError('the real geometry interface/gradient profile is not complete')
    verify_sources(profile['source_hashes'])
    output = geometry_output(config, 'training')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if metadata['profile_sha256'] != sha256(profile_path):
            raise RuntimeError('geometry profile changed after training initialization')
    else:
        output.mkdir(parents=True, exist_ok=False)
        metadata = {'created_local': now(), 'profile_sha256': sha256(profile_path),
            'control_qualification_sha256': config['control_qualification_sha256'],
            'source_hashes': snapshot(output, [Path(__file__), EXPERIMENT / 'scripts/train_milestone.py',
                EXPERIMENT / 'scripts/evaluate_interfaces.py', EXPERIMENT / 'configs/geometry_study.yaml',
                EXPERIMENT / 'docs/geometry_training_protocol.md', *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])}
        write_json(output / 'metadata.json', metadata)
    completed, started, elapsed_prior = 0, time.time(), 0.
    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'total_updates_per_new_arm': completed, 'image_updates_per_new_arm': max(0, completed - 2000),
            'milestone_total': until, 'research_goal_complete': False, **extra})
    try:
        status('LOADING_GEOMETRY_TRAINING')
        device = torch.device('cuda:0')
        decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
        if frozen != profile['frozen']:
            raise RuntimeError('frozen visual models differ from the geometry profile')
        networks, matches = {}, {}
        for variant in config['variants']:
            networks[variant], matches[variant] = geometry_network(config, base, qualification, variant, device)
            (output / variant / 'checkpoints').mkdir(parents=True, exist_ok=True)
        initials = {variant: module_sha256(network) for variant, network in networks.items()}
        if len(set(initials.values())) != 1:
            raise RuntimeError('receiver structures do not share their new-geometry initial parameters')
        def new_optimizer(network, rate):
            return torch.optim.AdamW(network.parameters(), lr=rate, betas=tuple(base['training']['betas']),
                                     weight_decay=base['training']['weight_decay'])
        optimizers = {name: new_optimizer(network, .0003) for name, network in networks.items()}
        histories, selected = {name: [] for name in networks}, {name: None for name in networks}
        calibrated, summaries, image_reset = [], [], False
        timings = {kind: {name: 0. for name in networks} for kind in ('representation', 'image', 'calibration')}
        if arguments.resume:
            saved = torch.load(output / 'resume.pt', map_location=device, weights_only=True)
            completed = saved['completed_total_updates']
            if completed >= until or saved['frozen'] != frozen:
                raise RuntimeError('invalid geometry resume boundary or frozen state')
            for name in networks:
                networks[name].load_state_dict(saved['models'][name], strict=True)
                optimizers[name].load_state_dict(saved['optimizers'][name])
            histories, selected, calibrated = saved['histories'], saved['selected'], saved['calibrated']
            summaries, image_reset, timings = saved['summaries'], saved['image_optimizer_reset'], saved['timings']
            elapsed_prior = saved['elapsed_seconds']
            del saved
        else:
            write_json(output / 'initialization.json', {'candidate_initial_hashes': initials, 'matches': matches,
                'CPU_threads': torch.get_num_threads(), 'controls_retrained': False,
                'control_qualification_sha256': config['control_qualification_sha256']})
        population, calibration = read_population(base, 'train'), read_population(base, 'calibration')
        monitor = monitoring_indices(calibration[2], base)

        def persist():
            for name in networks:
                write_csv(output / name / 'training.csv', histories[name])
                write_json(output / name / 'selected.json', selected[name])
            write_csv(output / 'calibration_summary.csv', summaries)
            state = {'completed_total_updates': completed, 'completed_image_updates': max(0, completed - 2000),
                'models': {name: network.state_dict() for name, network in networks.items()},
                'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                'histories': histories, 'selected': selected, 'calibrated': calibrated, 'summaries': summaries,
                'image_optimizer_reset': image_reset, 'timings': timings, 'frozen': frozen,
                'elapsed_seconds': elapsed_prior + time.time() - started}
            temporary = output / 'resume.pending.pt'
            torch.save(state, temporary)
            temporary.replace(output / 'resume.pt')

        def reset_image_optimizer():
            nonlocal image_reset, optimizers
            if completed == 2000 and not image_reset:
                optimizers = {name: new_optimizer(network, .0001) for name, network in networks.items()}
                image_reset = True
                print('REPRESENTATION_COMPLETE_FRESH_ADAM_FOR_ALL_NEW_GEOMETRY_ARMS', flush=True)

        def calibration_point():
            image_step = max(0, completed - 2000)
            eligible = completed in config['calibration']['monitor_representation_at'] or (
                completed >= 2000 and image_step % 500 == 0)
            if completed in calibrated or not eligible:
                return
            full = completed >= 2000 and (image_step in config['calibration']['full_image_steps'] or completed == until)
            scope = 'full' if full else ('representation_monitor' if completed < 2000 else 'image_monitor')
            for name, network in networks.items():
                status('CALIBRATING_GEOMETRY', variant=name, scope=scope)
                tick = time.perf_counter()
                rows = interface_calibrate(network, calibration, decoder, perceptual, base,
                    config['training']['image_weights'], device, None if full else monitor)
                write_csv(output / name / f'{scope}_total_{completed:07d}.csv', rows)
                metrics = [key for key in rows[0] if key not in ('image_id', 'snr_db', 'interface')]
                means = {key: float(np.mean([row[key] for row in rows])) for key in metrics}
                for snr in base['channel']['snrs_db']:
                    subset = [row for row in rows if row['snr_db'] == snr]
                    summaries.append({'total_step': completed, 'image_step': image_step, 'variant': name, 'scope': scope,
                        'snr_db': snr, 'source_images': len(subset),
                        **{key: float(np.mean([row[key] for row in subset])) for key in metrics}})
                if full:
                    path = output / name / 'checkpoints' / f'image_{image_step:07d}.pt'
                    torch.save({'model': network.state_dict(), 'variant': name, 'geometry': [204, 30],
                        'total_step': completed, 'image_step': image_step}, path)
                    selected[name] = choose(selected[name], {'step': image_step, 'total_step': completed,
                        'scope': 'full', 'source_images': 1000, 'checkpoint': str(path.relative_to(output)),
                        'checkpoint_sha256': sha256(path), **means})
                timings['calibration'][name] += time.perf_counter() - tick
                print(f'{scope} {name} total={completed} image={image_step} LPIPS={means["lpips"]:.6f} PSNR={means["psnr_db"]:.5f}', flush=True)
            calibrated.append(completed)
            persist()

        reset_image_optimizer()
        calibration_point()
        persist()
        status('TRAINING_GEOMETRY')
        for batch in paired_batches(base, 20000, completed, until):
            if batch['step'] != completed:
                raise RuntimeError('global paired sampler position changed')
            phase = 'representation' if completed < 2000 else 'image'
            weights = config['training'][phase + '_weights']
            rate = config['training'][phase + '_learning_rate']
            inputs = batch_inputs(population, batch, device)
            for name, network in networks.items():
                network.train().zero_grad(set_to_none=True)
                if any(group['lr'] != rate for group in optimizers[name].param_groups):
                    raise RuntimeError('geometry optimizer schedule differs from the qualified controls')
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                totals = {key: 0. for key in ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate', 'feature_mse')}
                power_error = 0.
                for index in range(4):
                    part = {key: value[index:index + 1] for key, value in inputs.items()}
                    result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
                    objective, components, diagnostics = interface_losses(result, part['source_fq'], part['images'], decoder, perceptual, weights)
                    if not bool(torch.isfinite(objective)):
                        raise FloatingPointError('nonfinite geometry objective')
                    (objective / 4).backward()
                    totals['loss'] += float(objective.detach()) / 4
                    for key, value in components.items():
                        totals[key] += float(value.detach().sum()) / 4
                    for key in ('bit_error_rate', 'feature_mse'):
                        totals[key] += float(diagnostics[key].detach().sum()) / 4
                    power_error = max(power_error, float((result['transmitted'].detach().square().sum(-1).mean(-1) - 2).abs().max()))
                    del result, objective, components, diagnostics, part
                norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True))
                if power_error > 1e-5:
                    raise RuntimeError('geometry changed the paid per-image signal energy')
                optimizers[name].step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                timings[phase][name] += seconds
                if phase == 'representation':
                    totals['mse'], totals['lpips'] = '', ''
                histories[name].append({'total_step': completed + 1, 'image_step': max(0, completed + 1 - 2000),
                    'optimizer_step': completed + 1 if phase == 'representation' else completed + 1 - 2000,
                    'phase': phase, 'learning_rate': rate, 'batch_sha256': batch['fingerprint'],
                    'gradient_norm': norm, 'step_seconds': seconds, 'power_max_error': power_error,
                    'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), **totals})
            completed += 1
            reset_image_optimizer()
            if completed == 1 or completed % config['training']['autosave_every'] == 0:
                persist()
                status('TRAINING_GEOMETRY', phase=phase)
                print(f'geometry paired total {completed}/{until} ' + ' '.join(f'{name}={histories[name][-1]["loss"]:.6f}' for name in networks), flush=True)
            calibration_point()
        expected_step = completed if completed < 2000 else completed - 2000
        for optimizer in optimizers.values():
            if any(int(state['step']) != expected_step for state in optimizer.state.values()):
                raise RuntimeError('geometry Adam reset/counters differ across phases')
        if completed != until or completed not in calibrated:
            raise RuntimeError('geometry milestone incomplete')
        if frozen != {'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('frozen visual model changed')
        verify_sources(metadata['source_hashes'])
        persist()
        milestones = output / 'milestones'
        milestones.mkdir(exist_ok=True)
        checkpoint = milestones / f'total_{completed:07d}_optimizer.pt'
        os.link(output / 'resume.pt', checkpoint)
        write_json(milestones / f'total_{completed:07d}.json', {'status': 'GEOMETRY_MILESTONE_COMPLETE', 'completed_local': now(),
            'total_updates_per_new_arm': completed, 'image_updates_per_new_arm': completed - 2000,
            'selected': selected, 'frozen': frozen, 'source_hashes': metadata['source_hashes'], 'timings': timings,
            'control_qualification_sha256': config['control_qualification_sha256'], 'checkpoint': str(checkpoint),
            'optimizer_checkpoint_sha256': sha256(checkpoint), 'observed_wall_GPU_hours': (elapsed_prior + time.time() - started) / 3600,
            'research_goal_complete': False})
        status('GEOMETRY_MILESTONE_READY_FOR_REVIEW_NOT_RESEARCH_COMPLETE')
    except BaseException as error:
        status('GEOMETRY_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
