#!/usr/bin/env python3
"""Matched, full-budget continuation and two-stage prefix recovery training."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

from var_comm.learned_prefix import empty_cache
from var_comm.next_scale_prior import state_sha256
from var_comm.prefix_learning_support import build_codec, load_frozen_training_models
from var_comm.prefix_refinement import engineering_check, gradient_monitor, optimizer_fingerprint, refinement_forward, refinement_losses, restore_optimizer, selection_summary, stage_weights, target_prefix_states
from var_comm.prefix_training_data import HeaderProtocol, read_image_population
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, snapshot, verify_artifacts, verify_snapshot, write_csv, write_json


def now():
    return datetime.now().astimezone().isoformat()


def write_status(output, status, **details):
    write_json(output / 'status.json', {'status': status, 'local_time': now(), 'pid': os.getpid(), **details})


def wait_for_gpu(output, config, allow_wait):
    required = config['execution']['minimum_free_GPU_MiB']
    while True:
        query = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '--id=0'], text=True)
        available = int(query.strip())
        if available >= required:
            return available
        write_status(output, 'WAITING_FOR_GPU_NOT_TRAINING', free_GPU_MiB=available, required_GPU_MiB=required,
                     optimizer_updates_this_experiment=0 if not (output / 'resume.pt').exists() else 'see_resume')
        if not allow_wait:
            raise RuntimeError(f'GPU has {available} MiB free; requires {required}; no other process will be stopped')
        print(f'{now()} waiting for GPU: free={available} MiB, required={required} MiB; no model loaded on GPU', flush=True)
        time.sleep(config['execution']['GPU_wait_poll_seconds'])


@torch.no_grad()
def calibrate(codec, population, header, vae, var, perceptual, base, config, device):
    codec.eval()
    rows = []
    images, tokens, labels, identifiers, noises = population
    for snr in config['selection']['snrs_db']:
        decoded, usable, false = header.decode(labels.numpy(), np.full(len(images), snr), noises[:, :68])
        for start in range(0, len(images), config['training']['micro_batch_size']):
            stop = min(start + config['training']['micro_batch_size'], len(images))
            image_batch = images[start:stop].to(device).float() / 255
            token_batch = tokens[start:stop].to(device)
            valid = torch.tensor(usable[start:stop], device=device)
            result = refinement_forward(codec, token_batch, torch.tensor(decoded[start:stop], device=device),
                                        torch.full((stop - start,), snr, device=device), torch.tensor(noises[start:stop, 68:], device=device),
                                        valid, vae, var)
            targets = target_prefix_states(token_batch, codec.codebook, vae)
            loss, components, diagnostics = refinement_losses(result, image_batch, token_batch, targets, valid, perceptual,
                                                              codec.embedding_std, config['training']['joint_weights'])
            metrics = {name: values.cpu().numpy() for name, values in components.items()}
            diagnostic = {name: values.cpu().numpy() for name, values in diagnostics.items()}
            for index in range(stop - start):
                row = {'image_id': identifiers[start + index], 'snr_db': snr, 'mse': float(metrics['mse'][index]),
                       'psnr_db': float(-10 * np.log10(max(float(metrics['mse'][index]), 1e-12))), 'lpips': float(metrics['lpips'][index]),
                       'token_ce': float(diagnostic['token_ce_raw'][index]), 'state_error': float(diagnostic['state_raw'][index]),
                       'token_error_rate': float(diagnostic['token_error_rate'][index]),
                       'header_usable': int(usable[start + index]), 'header_false_acceptance': int(false[start + index])}
                for scale in range(8):
                    for name, key in [('ce', 'ce_by_scale'), ('ter', 'ter_by_scale'), ('state', 'state_by_scale')]:
                        row[f'{name}_r{scale + 1}'] = float(diagnostic[key][index, scale])
                rows.append(row)
    if len(rows) != 5000 or len({row['image_id'] for row in rows}) != 1000:
        raise RuntimeError('incomplete fixed calibration population')
    return rows


def calibration_summaries(rows, branch, step):
    numeric = [key for key in rows[0] if key not in ('image_id', 'snr_db')]
    summaries = []
    for snr in sorted({row['snr_db'] for row in rows}):
        selected = [row for row in rows if row['snr_db'] == snr]
        summaries.append({'branch': branch, 'additional_step': step, 'snr_db': snr, 'source_images': len(selected),
                          **{name: float(np.mean([row[name] for row in selected])) for name in numeric}})
    return summaries


def read_csv(path):
    import csv

    with path.open() as handle:
        return list(csv.DictReader(handle))


def compare_starting_calibration(rows, previous):
    reference = {(row['image_id'], float(row['snr_db'])): row for row in read_csv(previous / 'next_scale/calibration_epoch_02.csv')}
    errors = {'mse': 0.0, 'lpips': 0.0}
    if len(reference) != len(rows):
        raise RuntimeError('starting calibration grid changed')
    for row in rows:
        saved = reference[row['image_id'], row['snr_db']]
        for metric in errors:
            errors[metric] = max(errors[metric], abs(row[metric] - float(saved[metric])))
        if row['header_usable'] != int(saved['header_usable']) or row['header_false_acceptance'] != int(saved['header_false_acceptance']):
            raise RuntimeError('starting calibration header changed')
    if errors['mse'] > 1e-6 or errors['lpips'] > 1e-5:
        raise RuntimeError(f'starting hard calibration does not reproduce the frozen result: {errors}')
    return errors


def checkpoint_stage(branch, step, config):
    if step == 0:
        return 'starting_checkpoint'
    if branch == 'continuation':
        return 'continuation'
    return 'prefix' if step <= config['training']['prefix_only_updates'] else 'joint'


def save_checkpoint(output, branch, step, codec, config):
    path = output / branch / 'checkpoints' / f'step_{step:05d}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': codec.state_dict(), 'branch': branch, 'additional_step': step,
                'optimizer_step': config['starting_step'] + step, 'stage': checkpoint_stage(branch, step, config),
                'starting_checkpoint_sha256': config['starting_checkpoint_sha256']}, path)
    return str(path.relative_to(output)), sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/prefix_refinement.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wait-for-gpu', action='store_true')
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text())
    base = yaml.safe_load((ROOT / config['base_config']).read_text())
    previous = ROOT / config['starting_training']
    previous_receipt = verify_artifacts(previous, 'completion.json', config['starting_training_receipt_sha256'])
    verify_snapshot(previous_receipt['source_hashes'])
    for filename, expected in [(config['starting_checkpoint'], config['starting_checkpoint_sha256']), ('resume.pt', config['starting_resume_sha256'])]:
        if sha256(previous / filename) != expected:
            raise RuntimeError(f'starting artifact changed: {filename}')
    requested = (arguments.output_dir or ROOT / config['outputs']['training']).resolve()
    if not requested.is_relative_to(ROOT / 'outputs') or (requested / 'completion.json').exists():
        raise ValueError('output must be an unfinished directory under VAR_COMM/outputs')
    if arguments.resume:
        output = requested
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_snapshot(metadata['source_hashes'])
    else:
        output = create_output(requested)
        sources = [Path(__file__), arguments.config, ROOT / config['base_config'], ROOT / config['protocol'],
                   ROOT / 'src/var_comm/prefix_refinement.py', ROOT / 'src/var_comm/learned_prefix.py',
                   ROOT / 'src/var_comm/prefix_learning_support.py', ROOT / 'src/var_comm/prefix_training_data.py',
                   ROOT / 'src/var_comm/next_scale_prior.py', ROOT / 'src/var_comm/study.py']
        metadata = {'local_authorized_run_created': now(), 'command': sys.argv, 'source_hashes': snapshot(output, sources),
                    'torch': str(torch.__version__), 'numpy': str(np.__version__), 'cuda_build': torch.version.cuda,
                    'starting_training_receipt_sha256': sha256(previous / 'completion.json'),
                    'starting_checkpoint_sha256': config['starting_checkpoint_sha256'], 'starting_resume_sha256': config['starting_resume_sha256'],
                    'DINO_training_or_selection': False, 'config': config}
        write_json(output / 'metadata.json', metadata)
    completed = 0
    try:
        wait_for_gpu(output, config, arguments.wait_for_gpu)
        verify_snapshot(metadata['source_hashes'])
        write_status(output, 'LOADING_TRAINING_MODELS', additional_updates=0)
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.mha.set_fastpath_enabled(False)
        device = torch.device('cuda:0')
        vae, var, perceptual = load_frozen_training_models(base, device)
        backbones = {'vae': vae, 'var': var, 'lpips': perceptual}
        before = {name: state_sha256(model) for name, model in backbones.items()}
        if before != previous_receipt['frozen_after']:
            raise RuntimeError('frozen backbones differ from original training')
        starting = torch.load(previous / 'resume.pt', map_location='cpu', weights_only=True)
        checkpoint = torch.load(previous / config['starting_checkpoint'], map_location='cpu', weights_only=True)
        if starting['global_step'] != config['starting_step'] or not all(torch.equal(tensor, checkpoint['model'][name]) for name, tensor in starting['models']['next_scale'].items()):
            raise RuntimeError('checkpoint and optimizer are not a matched starting state')
        codecs, optimizers = {}, {}
        for branch in config['branches']:
            codec = build_codec(base, vae, 'next_scale', device)
            codec.load_state_dict(checkpoint['model'], strict=True)
            codecs[branch] = codec
            optimizers[branch] = restore_optimizer(codec, starting['optimizers']['next_scale'], config['training']['learning_rate'])
        state_hashes = {name: state_sha256(model) for name, model in codecs.items()}
        optimizer_hashes = {name: optimizer_fingerprint(optimizer.state_dict()) for name, optimizer in optimizers.items()}
        expected_optimizer_hash = optimizer_fingerprint(starting['optimizers']['next_scale'])
        if len(set(state_hashes.values())) != 1 or set(optimizer_hashes.values()) != {expected_optimizer_hash}:
            raise RuntimeError('branches do not start from identical model and optimizer states')
        for optimizer in optimizers.values():
            for parameter, state in optimizer.state.items():
                if state['exp_avg'].shape != parameter.shape or state['exp_avg_sq'].shape != parameter.shape:
                    raise RuntimeError('optimizer parameter ordering does not match the frozen model')
        write_json(output / 'initialization.json', {'model_state_sha256': state_hashes, 'optimizer_state_sha256': optimizer_hashes,
                   'trainable_parameters': {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in codecs.items()},
                   'optimizer_starting_step': config['starting_step'], 'stale_old_selection_ignored': True})
        data_receipt = verify_artifacts(ROOT / base['data_cache'], 'completion.json')
        images, labels, identifiers, bindings = read_image_population('train')
        calibration_images, calibration_labels, calibration_ids, calibration_bindings = read_image_population('calibration')
        with np.load(ROOT / base['data_cache'] / 'train.npz', allow_pickle=False) as cache:
            source_indices = torch.tensor(cache['source_indices'])
            source_tokens = torch.tensor(cache['tokens'].astype(np.int64))
            np.testing.assert_array_equal(labels[source_indices].numpy(), cache['labels'])
        with np.load(ROOT / base['data_cache'] / 'calibration.npz', allow_pickle=False) as cache:
            positions = torch.tensor(cache['source_indices'])
            calibration_tokens = torch.tensor(cache['tokens'][:, 0].astype(np.int64))
            np.testing.assert_array_equal(calibration_labels[positions].numpy(), cache['labels'])
        calibration_ids = [calibration_ids[index] for index in positions.tolist()]
        calibration_noise = np.stack([seeded_noise(identifier, config['selection']['noise_seed'], (3060, 2)).astype(np.float32) for identifier in calibration_ids])
        population = (calibration_images[positions], calibration_tokens, calibration_labels[positions], calibration_ids, calibration_noise)
        write_json(output / 'population_binding.json', {'training_images': len(source_indices), 'calibration_images': len(positions),
                   'data_receipt_sha256': sha256(ROOT / base['data_cache'] / 'completion.json'),
                   'training_ids_sha256': hashlib.sha256(json.dumps([identifiers[index] for index in source_indices.tolist()]).encode()).hexdigest(),
                   'calibration_ids_sha256': hashlib.sha256(json.dumps(calibration_ids).encode()).hexdigest(),
                   'calibration_noise_sha256': hashlib.sha256(calibration_noise.tobytes()).hexdigest()})
        header = HeaderProtocol()
        probe_noise = torch.randn((2, 3060, 2), generator=torch.Generator().manual_seed(config['training']['gradient_monitor_seed']))
        probe_snrs = torch.tensor([4., 7.], device=device)
        probe_positions = source_indices[:2]
        decoded, usable, false = header.decode(labels[probe_positions].numpy(), np.array([4., 7.]), probe_noise[:, :68].numpy())
        probe = {'tokens': source_tokens[:2, 0].to(device), 'labels': torch.tensor(decoded, device=device), 'snrs': probe_snrs,
                 'noise': probe_noise[:, 68:].to(device), 'valid': torch.tensor(usable, device=device),
                 'images': images[probe_positions].to(device).float() / 255}
        histories = {branch: [] for branch in config['branches']}
        selections = {branch: None for branch in config['branches']}
        calibration_history, monitor_history, calibrated_steps, monitored_steps = [], [], [], []
        base_psnr = None
        resumed = None
        if arguments.resume and (output / 'resume.pt').exists():
            resumed = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True)
            completed = resumed['completed_updates']
            for branch in config['branches']:
                codecs[branch].load_state_dict(resumed['models'][branch], strict=True)
                optimizers[branch] = restore_optimizer(codecs[branch], resumed['optimizers'][branch], config['training']['learning_rate'])
            histories, selections = resumed['histories'], resumed['selections']
            calibration_history, monitor_history = resumed['calibration_history'], resumed['monitor_history']
            calibrated_steps, monitored_steps, base_psnr = resumed['calibrated_steps'], resumed['monitored_steps'], resumed['base_psnr']
        else:
            write_status(output, 'GPU_ENGINEERING_CHECK_NO_UPDATES')
            check_started = time.perf_counter()
            check = engineering_check(codecs['continuation'], **probe, vae=vae, var=var, perceptual=perceptual, base_config=base)
            check['seconds'] = time.perf_counter() - check_started
            write_json(output / 'engineering_check.json', check)
        timing = resumed['timing'] if resumed else {'training_seconds': 0.0, 'calibration_seconds': 0.0, 'monitor_seconds': 0.0, 'engineering_seconds': check['seconds']}

        def persist():
            for branch in config['branches']:
                if histories[branch]:
                    write_csv(output / branch / 'training.csv', histories[branch])
            if calibration_history:
                write_csv(output / 'calibration_summary.csv', calibration_history)
            if monitor_history:
                write_csv(output / 'gradient_monitor.csv', monitor_history)
            state = {'completed_updates': completed, 'models': {name: codec.state_dict() for name, codec in codecs.items()},
                     'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()}, 'histories': histories,
                     'selections': selections, 'calibration_history': calibration_history, 'monitor_history': monitor_history,
                     'calibrated_steps': calibrated_steps, 'monitored_steps': monitored_steps, 'base_psnr': base_psnr, 'timing': timing}
            temporary = output / 'resume.pending.pt'
            torch.save(state, temporary)
            temporary.replace(output / 'resume.pt')

        def monitor(step):
            if step in monitored_steps or step not in config['training']['gradient_monitor_steps']:
                return
            tick = time.perf_counter()
            for branch, codec in codecs.items():
                codec.train()
                phase, weights = stage_weights(config, branch, step)
                codec.zero_grad(set_to_none=True)
                for row in gradient_monitor(codec, **probe, vae=vae, var=var, perceptual=perceptual, weights=weights):
                    monitor_history.append({'branch': branch, 'completed_additional_step': step, 'monitored_next_phase': phase, **row})
            timing['monitor_seconds'] += time.perf_counter() - tick
            monitored_steps.append(step)

        def run_calibration(step):
            nonlocal base_psnr
            if step in calibrated_steps:
                return
            tick = time.perf_counter()
            shared_initial_rows = None
            for branch, codec in codecs.items():
                write_status(output, 'CALIBRATING', branch=branch, completed_additional_updates=step)
                checkpoint_path, checkpoint_sha = save_checkpoint(output, branch, step, codec, config)
                rows = shared_initial_rows if step == 0 and shared_initial_rows is not None else calibrate(codec, population, header, vae, var, perceptual, base, config, device)
                if step == 0 and shared_initial_rows is None:
                    shared_initial_rows = rows
                    write_json(output / 'starting_calibration_reproduction.json', compare_starting_calibration(rows, previous))
                    base_psnr = selection_summary(rows, None, config['selection']['psnr_guard_db'])['psnr_by_snr']
                write_csv(output / branch / f'calibration_step_{step:05d}.csv', rows)
                calibration_history.extend(calibration_summaries(rows, branch, step))
                result = selection_summary(rows, base_psnr, config['selection']['psnr_guard_db'])
                candidate = {'additional_step': step, 'stage': checkpoint_stage(branch, step, config), **result,
                             'checkpoint': checkpoint_path, 'checkpoint_sha256': checkpoint_sha}
                if result['admissible'] and (selections[branch] is None or result['mean_lpips'] < selections[branch]['mean_lpips']):
                    selections[branch] = candidate
                write_json(output / branch / 'selected.json', selections[branch])
                print(f"calibration {branch} step={step} LPIPS={result['mean_lpips']:.8f} guard={result['admissible']} selected={selections[branch]['additional_step']}", flush=True)
            calibrated_steps.append(step)
            timing['calibration_seconds'] += time.perf_counter() - tick
            persist()

        monitor(completed)
        if completed % config['selection']['every_updates'] == 0:
            run_calibration(completed)
        batch_size, micro = config['training']['effective_batch_size'], config['training']['micro_batch_size']
        updates_per_epoch = len(source_indices) // batch_size
        if len(source_indices) != 20000 or len(source_indices) % batch_size:
            raise RuntimeError('training population cannot satisfy the registered update budget')
        training_started = now()
        write_status(output, 'TRAINING', completed_additional_updates=completed, additional_images_per_branch=completed * batch_size)
        for local_epoch, data_epoch in enumerate(config['training']['data_epoch_indices']):
            ordering = torch.Generator().manual_seed(base['training']['order_seed'] + data_epoch)
            order = torch.randperm(len(source_indices), generator=ordering)
            flips = torch.rand(len(source_indices), generator=ordering) < 0.5
            channel = torch.Generator().manual_seed(base['training']['channel_seed'] + data_epoch)
            for start in range(0, len(order), batch_size):
                step = local_epoch * updates_per_epoch + start // batch_size
                selected = order[start:start + batch_size]
                flip = flips[start:start + batch_size]
                snr_indices = torch.randint(len(base['training']['snrs_db']), (len(selected),), generator=channel)
                snrs_cpu = torch.tensor(base['training']['snrs_db'])[snr_indices]
                noise = torch.randn((len(selected), 3060, 2), generator=channel)
                if step < completed:
                    continue
                if step != completed:
                    raise RuntimeError('sampler cannot reproduce the saved update position')
                positions = source_indices[selected]
                tokens = source_tokens[selected, flip.long()]
                decoded, usable, false = header.decode(labels[positions].numpy(), snrs_cpu.numpy(), noise[:, :68].numpy())
                digest = hashlib.sha256()
                for tensor in (selected, flip, snrs_cpu, noise):
                    digest.update(tensor.numpy().tobytes())
                fingerprint = digest.hexdigest()
                for branch, codec in codecs.items():
                    codec.train()
                    phase, weights = stage_weights(config, branch, step)
                    render = bool(weights['mse'] or weights['lpips'])
                    optimizer = optimizers[branch]
                    optimizer.zero_grad(set_to_none=True)
                    totals = {name: 0.0 for name in ('loss', 'mse', 'lpips', 'token_ce', 'state', 'token_error_rate')}
                    maximum_power_error = 0.0
                    tick = time.perf_counter()
                    for offset in range(0, batch_size, micro):
                        end = min(offset + micro, batch_size)
                        target = images[positions[offset:end]].to(device).float() / 255
                        flipping = flip[offset:end].to(device)
                        target = torch.where(flipping[:, None, None, None], target.flip(-1), target)
                        token_batch = tokens[offset:end].to(device)
                        valid = torch.tensor(usable[offset:end], device=device)
                        result = refinement_forward(codec, token_batch, torch.tensor(decoded[offset:end], device=device),
                                                    snrs_cpu[offset:end].to(device), noise[offset:end, 68:].to(device), valid, vae, var, render)
                        targets = target_prefix_states(token_batch, codec.codebook, vae)
                        loss, components, diagnostics = refinement_losses(result, target, token_batch, targets, valid, perceptual, codec.embedding_std, weights)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite refinement loss')
                        (loss * ((end - offset) / batch_size)).backward()
                        totals['loss'] += float(loss.detach()) * (end - offset)
                        for name, values in components.items():
                            totals[name] += float(values.detach().sum())
                        totals['token_error_rate'] += float(diagnostics['token_error_rate'].detach().sum())
                        maximum_power_error = max(maximum_power_error, float((result['symbols'].detach().square().sum(-1).mean(-1) - 2).abs().max()))
                        del loss, components, diagnostics, result, targets
                    norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), config['training']['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    if maximum_power_error > 1e-5 or not empty_cache(var):
                        raise RuntimeError('power or shared VAR cache boundary failed')
                    elapsed = time.perf_counter() - tick
                    timing['training_seconds'] += elapsed
                    record = {'additional_step': step + 1, 'optimizer_step': config['starting_step'] + step + 1, 'data_epoch_index': data_epoch,
                              'additional_images_seen': (step + 1) * batch_size, 'phase': phase, 'batch_sha256': fingerprint,
                              **{name: value / batch_size for name, value in totals.items()}, 'image_loss_active': int(render), 'teacher_probability': 0.0,
                              'header_usable_fraction': float(np.mean(usable)), 'header_false_acceptances': int(sum(false)),
                              'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm': float(norm), 'power_max_error': maximum_power_error, 'step_seconds': elapsed}
                    if not render:
                        record['mse'], record['lpips'] = '', ''
                    histories[branch].append(record)
                completed = step + 1
                if completed % config['training']['autosave_every'] == 0:
                    persist()
                    write_status(output, 'TRAINING', completed_additional_updates=completed, additional_images_per_branch=completed * batch_size)
                    print(f"updates {completed}/10000 " + ' '.join(f"{branch}: loss={histories[branch][-1]['loss']:.6f}" for branch in config['branches']), flush=True)
                monitor(completed)
                if completed % config['selection']['every_updates'] == 0:
                    run_calibration(completed)
        if completed != config['training']['additional_updates'] or calibrated_steps != list(range(0, completed + 1, 1000)):
            raise RuntimeError('full update/calibration budget was not completed')
        for optimizer in optimizers.values():
            if any(float(state['step']) != config['starting_step'] + completed for state in optimizer.state.values()):
                raise RuntimeError('optimizer moments or update counters were reset or skipped')
        after = {name: state_sha256(model) for name, model in backbones.items()}
        if after != before or any(parameter.grad is not None or parameter.requires_grad for model in backbones.values() for parameter in model.parameters()):
            raise RuntimeError('frozen training backbones changed')
        verify_snapshot(metadata['source_hashes'])
        verify_snapshot(previous_receipt['source_hashes'])
        persist()
        write_status(output, 'BOTH_BRANCHES_TRAINING_COMPLETE', completed_additional_updates=completed)
        write_json(output / 'completion.json', {'status': 'MATCHED_PREFIX_REFINEMENT_TRAINING_COMPLETE', 'source_hashes': metadata['source_hashes'],
                   'local_training_started_this_session': training_started, 'local_completed': now(), 'additional_updates_per_branch': completed,
                   'additional_images_per_branch': completed * batch_size, 'calibration_steps': calibrated_steps, 'selected': selections,
                   'base_psnr_by_snr': base_psnr, 'frozen_before': before, 'frozen_after': after, 'timing': timing,
                   'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(device), 'DINO_training_or_selection': False,
                   'starting_training_receipt_sha256': config['starting_training_receipt_sha256'], 'output_hashes': artifact_hashes(output)})
        print('MATCHED_PREFIX_REFINEMENT_TRAINING_COMPLETE', flush=True)
    except BaseException as error:
        write_status(output, 'INTERRUPTED_OR_FAILED_NOT_COMPLETE', completed_additional_updates=completed, error=repr(error))
        raise


if __name__ == '__main__':
    main()
