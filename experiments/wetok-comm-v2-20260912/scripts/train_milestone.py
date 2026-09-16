"""Paired native WeTok communication training with resumable research milestones."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np
import torch

from wetok_comm.common import WORKSPACE, artifact_hashes, configure_torch, now, output_path, settings, sha256, snapshot, verify_sources, write_json
from wetok_comm.model import communicate
from wetok_comm.native import FrozenWeTok
from wetok_comm.objective import losses
from wetok_comm.training import batch_inputs, calibrate, load_lpips, module_sha256, monitoring_indices, new_network, paired_batches, read_population, stage


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def choose(previous, candidate):
    if candidate['scope'] != 'full' or candidate['source_images'] != 1000:
        raise ValueError('only complete calibration can choose models')
    if not np.isfinite(candidate['lpips']):
        raise ValueError('nonfinite calibration selection score')
    if previous is None or (candidate['lpips'], candidate['step']) < (previous['lpips'], previous['step']):
        return candidate
    return previous


def component_gradient_rows(network, inputs, native, perceptual, weights):
    result = communicate(network, inputs['source_fq'][:1], inputs['snrs'][:1], inputs['noise'][:1])
    objective, components, diagnostic = losses(result, inputs['source_fq'][:1], inputs['images'][:1], native, perceptual, weights)
    parameters = tuple(network.parameters())
    gradients, rows = {}, []
    for name, values in components.items():
        if not values.requires_grad:
            continue
        computed = torch.autograd.grad(values.mean(), parameters, retain_graph=True, allow_unused=True)
        flat = torch.cat([(value.detach().cpu().reshape(-1) if value is not None else torch.zeros(parameter.numel()))
                          for value, parameter in zip(computed, parameters)])
        gradients[name] = flat * weights[name]
        norm = float(flat.norm())
        rows.append({'component': name, 'unweighted_norm': norm, 'weight': weights[name], 'weighted_norm': norm * weights[name],
                     'image_auxiliary_cosine': ''})
    if 'mse' in gradients and 'lpips' in gradients:
        image = gradients['mse'] + gradients['lpips']
        auxiliary = gradients['bits'] + gradients['state']
        denominator = float(image.norm() * auxiliary.norm())
        cosine = float(torch.dot(image, auxiliary)) / denominator if denominator > 0 else 0.
        for row in rows:
            row['image_auxiliary_cosine'] = cosine
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/study.yaml')
    parser.add_argument('--until-step', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config = settings(arguments.config)
    until = arguments.until_step or config['training']['first_milestone_updates']
    if until <= 0 or until % config['training']['milestone_stride']:
        raise ValueError('use a registered milestone boundary, not a new total research cap')
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'until_step': until, 'arms': config['arms'], 'not_a_total_training_cap': True}))
        return
    configure_torch()
    output = output_path(config, 'training')
    profile_root = output_path(config, 'profile')
    profile = json.loads((profile_root / 'profile.json').read_text())
    if profile['status'] != 'NATIVE_GRADIENT_PROFILE_PASS' or profile['optimizer_updates'] != 0:
        raise RuntimeError('native identity/input-gradient/power engineering checks must pass')
    verify_sources(profile['source_hashes'])
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if metadata['config_sha256'] != sha256(arguments.config):
            raise RuntimeError('use a separately registered recipe for configuration changes')
    else:
        output.mkdir(parents=True, exist_ok=False)
        paths = [Path(__file__), arguments.config, EXPERIMENT / 'docs/protocol.md',
                 *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))]
        metadata = {'created_local': now(), 'source_hashes': snapshot(output, paths), 'config_sha256': sha256(arguments.config),
                    'profile_sha256': sha256(profile_root / 'profile.json'), 'config': config, 'milestone_not_research_limit': True}
        write_json(output / 'metadata.json', metadata)
    for arm in config['arms']:
        (output / arm / 'checkpoints').mkdir(parents=True, exist_ok=True)
    completed, started = 0, time.time()
    def status(state, **fields):
        write_json(output / 'status.json', {'status': state, 'local_time': now(), 'pid': os.getpid(),
            'completed_updates_per_arm': completed, 'current_milestone': until, 'no_total_research_cap': True, **fields})
    try:
        status('LOADING_TRAINING_COMPONENTS')
        device = torch.device('cuda:0')
        native = FrozenWeTok(device, 'decoder')
        perceptual = load_lpips(device)
        frozen = {'native_codec': module_sha256(native.codec), 'lpips': module_sha256(perceptual)}
        networks = {arm: new_network(config, arm, device) for arm in config['arms']}
        initial = {arm: module_sha256(network) for arm, network in networks.items()}
        if len(set(initial.values())) != 1:
            raise RuntimeError('comparison arms did not receive identical initialization')
        optimizers = {arm: torch.optim.AdamW(network.parameters(), lr=config['training']['representation_learning_rate'],
                      betas=tuple(config['training']['betas']), weight_decay=config['training']['weight_decay']) for arm, network in networks.items()}
        population = read_population(config, 'train')
        calibration = read_population(config, 'calibration')
        monitor_positions = monitoring_indices(calibration[2], config)
        write_json(output / 'initialization.json', {'initial_model_hashes': initial,
                   'parameters': {arm: sum(value.numel() for value in network.parameters()) for arm, network in networks.items()},
                   'frozen': frozen, 'optimizer_initialization': 'fresh_AdamW_all_arms', 'source_class_condition': False})
        histories = {arm: [] for arm in config['arms']}
        selected = {arm: None for arm in config['arms']}
        calibrated, calibration_history = [], []
        gradient_history, monitored = [], []
        joint_rate = config['training']['image_learning_rate']
        no_gain_milestones = 0
        elapsed_prior = 0.
        timings = {'training': {arm: 0. for arm in config['arms']}, 'calibration': {arm: 0. for arm in config['arms']}}
        if arguments.resume:
            saved = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True)
            completed = saved['completed_updates']
            if completed >= until:
                raise ValueError('the requested milestone is already reached; select the next research milestone')
            for arm in config['arms']:
                networks[arm].load_state_dict(saved['models'][arm], strict=True)
                optimizers[arm].load_state_dict(saved['optimizers'][arm])
            histories, selected = saved['histories'], saved['selected']
            calibrated, calibration_history = saved['calibrated'], saved['calibration_history']
            gradient_history, monitored = saved['gradient_history'], saved['monitored']
            joint_rate, no_gain_milestones = saved['joint_rate'], saved['no_gain_milestones']
            timings, elapsed_prior = saved['timings'], saved['elapsed_seconds']
            if saved['frozen'] != frozen:
                raise RuntimeError('resumed visual models differ from the original run')
            del saved
        resumed_from = completed

        def persist():
            for arm in config['arms']:
                write_csv(output / arm / 'training.csv', histories[arm])
                write_json(output / arm / 'selected.json', selected[arm])
            write_csv(output / 'calibration_summary.csv', calibration_history)
            write_csv(output / 'gradient_monitor.csv', gradient_history)
            saved = {'completed_updates': completed, 'models': {arm: network.state_dict() for arm, network in networks.items()},
                     'optimizers': {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()}, 'histories': histories,
                     'selected': selected, 'calibrated': calibrated, 'calibration_history': calibration_history, 'frozen': frozen,
                     'gradient_history': gradient_history, 'monitored': monitored,
                     'joint_rate': joint_rate, 'no_gain_milestones': no_gain_milestones, 'timings': timings,
                     'elapsed_seconds': elapsed_prior + time.time() - started}
            temporary = output / 'resume.pending.pt'
            torch.save(saved, temporary)
            temporary.replace(output / 'resume.pt')

        def run_calibration():
            nonlocal joint_rate, no_gain_milestones
            full = completed == config['training']['representation_updates'] or (completed > 0 and completed % config['training']['milestone_stride'] == 0)
            monitor = completed > 0 and completed % config['calibration']['monitor_every_updates'] == 0
            if not (full or monitor) or completed in calibrated:
                return
            scope = 'full' if full else 'monitor'
            improved = False
            for arm, network in networks.items():
                status('CALIBRATING', scope=scope, arm=arm)
                tick = time.perf_counter()
                rows = calibrate(network, calibration, native, perceptual, config, device, None if full else monitor_positions)
                timings['calibration'][arm] += time.perf_counter() - tick
                write_csv(output / arm / f'{scope}_{completed:07d}.csv', rows)
                means = {name: float(np.mean([row[name] for row in rows])) for name in ('psnr_db', 'ssim', 'lpips', 'bits_BCE', 'state_error', 'bit_error_rate', 'group_error_rate')}
                for snr in config['channel']['snrs_db']:
                    current = [row for row in rows if row['snr_db'] == snr]
                    calibration_history.append({'step': completed, 'arm': arm, 'scope': scope, 'snr_db': snr,
                        'source_images': len(current), **{name: float(np.mean([row[name] for row in current])) for name in means}})
                if full:
                    checkpoint = output / arm / 'checkpoints' / f'step_{completed:07d}.pt'
                    torch.save({'model': network.state_dict(), 'arm': arm, 'step': completed}, checkpoint)
                    candidate = {'step': completed, 'scope': 'full', 'source_images': 1000, 'checkpoint': str(checkpoint.relative_to(output)),
                                 'checkpoint_sha256': sha256(checkpoint), **means}
                    prior = selected[arm]
                    if prior is None or candidate['lpips'] < prior['lpips'] - max(1e-4, .002 * prior['lpips']):
                        improved = True
                    selected[arm] = choose(prior, candidate)
                print(f'{scope} {arm} step={completed} PSNR={means["psnr_db"]:.5f} LPIPS={means["lpips"]:.6f} BER={means["bit_error_rate"]:.6f}', flush=True)
            if full and completed > config['training']['representation_updates']:
                no_gain_milestones = 0 if improved else no_gain_milestones + 1
                if no_gain_milestones >= 2:
                    joint_rate = max(config['training']['minimum_learning_rate'], joint_rate * .5)
                    no_gain_milestones = 0
                    print(f'Common LR plateau reduction: {joint_rate}; no arm receives a private schedule', flush=True)
            calibrated.append(completed)
            persist()

        run_calibration()
        persist()
        micro = config['training']['micro_batch_size']
        status('TRAINING', phase=stage(config, completed, joint_rate)[0])
        for batch in paired_batches(config, 20000, completed, until):
            if batch['step'] != completed:
                raise RuntimeError('paired training position changed')
            inputs = batch_inputs(population, batch, device)
            phase, weights, rate = stage(config, completed, joint_rate)
            if completed in (0, config['training']['representation_updates']) and completed not in monitored:
                for arm, network in networks.items():
                    network.train().zero_grad(set_to_none=True)
                    gradient_history.extend({'arm': arm, 'step': completed, 'phase': phase, **row}
                        for row in component_gradient_rows(network, inputs, native, perceptual, weights))
                monitored.append(completed)
            for arm, network in networks.items():
                network.train()
                optimizer = optimizers[arm]
                for group in optimizer.param_groups:
                    group['lr'] = rate
                optimizer.zero_grad(set_to_none=True)
                totals = {name: 0. for name in ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate', 'group_error_rate')}
                power_error = 0.
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                for offset in range(0, 4, micro):
                    stop = min(offset + micro, 4)
                    part = {key: value[offset:stop] for key, value in inputs.items()}
                    result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
                    objective, components, diagnostic = losses(result, part['source_fq'], part['images'], native, perceptual, weights)
                    if not torch.isfinite(objective):
                        raise FloatingPointError('nonfinite communication objective')
                    (objective * ((stop - offset) / 4)).backward()
                    totals['loss'] += float(objective.detach()) * ((stop - offset) / 4)
                    for name, value in components.items():
                        totals[name] += float(value.detach().sum()) / 4
                    for name in ('bit_error_rate', 'group_error_rate'):
                        totals[name] += float(diagnostic[name].detach().sum()) / 4
                    power_error = max(power_error, float((result['transmitted'].detach().square().sum(-1).mean(-1) - 2).abs().max()))
                    del part, result, objective, components, diagnostic
                norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config['training']['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                if power_error > 1e-5:
                    raise RuntimeError('per-image transmission energy exceeded the registered tolerance')
                timings['training'][arm] += seconds
                record = {'step': completed + 1, 'optimizer_step': completed + 1, 'phase': phase, 'epoch': batch['epoch'],
                          'batch_sha256': batch['fingerprint'], 'learning_rate': rate, 'gradient_norm': float(norm),
                          'power_max_error': power_error, 'step_seconds': seconds,
                          'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), **totals}
                if phase == 'representation':
                    record['mse'], record['lpips'] = '', ''
                histories[arm].append(record)
            completed += 1
            if completed == resumed_from + 1 or completed % config['training']['autosave_every'] == 0:
                persist()
                status('TRAINING', phase=phase)
                print(f'paired updates {completed}/{until} phase={phase} ' + ' '.join(f'{arm}={histories[arm][-1]["loss"]:.6f}' for arm in config['arms']), flush=True)
            run_calibration()
        if completed != until or completed not in calibrated:
            raise RuntimeError('milestone update/calibration sequence incomplete')
        for optimizer in optimizers.values():
            if any(int(value['step']) != completed for value in optimizer.state.values()):
                raise RuntimeError('Adam state was reset or update history is mismatched')
        after = {'native_codec': module_sha256(native.codec), 'lpips': module_sha256(perceptual)}
        if after != frozen or any(value.grad is not None or value.requires_grad for value in native.codec.parameters()):
            raise RuntimeError('visual codec parameters changed')
        verify_sources(metadata['source_hashes'])
        persist()
        status('MILESTONE_COMPLETE_NOT_RESEARCH_FINISHED', phase='joint', next_candidate_milestone=until + config['training']['milestone_stride'])
        write_json(output / 'milestones' / f'step_{completed:07d}.json', {'status': 'MILESTONE_COMPLETE', 'completed_local': now(),
            'updates_per_arm': completed, 'images_seen_per_arm': completed * 4, 'selected': selected, 'joint_rate': joint_rate,
            'no_gain_milestones': no_gain_milestones, 'frozen_before': frozen, 'frozen_after': after,
            'source_hashes': metadata['source_hashes'], 'checkpoint_sha256': sha256(output / 'resume.pt'),
            'timings': timings, 'observed_training_gpu_hours': (elapsed_prior + time.time() - started) / 3600,
            'note': 'Research milestone, not an automatic stopping or success decision.'})
        print('PAIRED_MILESTONE_COMPLETE_REVIEW_CALIBRATION_AND_CONTINUE', flush=True)
    except BaseException as error:
        status('TRAINING_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
