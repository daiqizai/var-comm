#!/usr/bin/env python3
"""Train both global prefix communication variants on identical data and channel draws."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
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
from var_comm.prefix_learning_support import build_codec, communication_forward, image_losses, load_frozen_training_models
from var_comm.prefix_training_data import HeaderProtocol, read_image_population
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, snapshot, verify_artifacts, verify_snapshot, write_csv, write_json


@torch.no_grad()
def calibrate(codec, images, tokens, labels, identifiers, header, noises, vae, var, perceptual, config, device):
    codec.eval()
    rows = []
    for snr in config['selection']['snrs_db']:
        snrs = np.full(len(images), snr)
        received_labels, usable, false = header.decode(labels.numpy(), snrs, noises[:, :68])
        for start in range(0, len(images), config['training']['micro_batch_size']):
            stop = min(start + config['training']['micro_batch_size'], len(images))
            source = images[start:stop].to(device).float() / 255
            indices = tokens[start:stop].to(device)
            condition = torch.tensor(received_labels[start:stop], device=device, dtype=torch.long)
            valid = torch.tensor(usable[start:stop], device=device)
            snr_tensor = torch.full((stop - start,), snr, device=device)
            noise = torch.tensor(noises[start:stop, 68:], device=device)
            result = communication_forward(codec, indices, condition, snr_tensor, noise, valid, vae, var)
            mse = (result['image'] - source).square().flatten(1).mean(1)
            lpips_values = perceptual(result['image'] * 2 - 1, source * 2 - 1).flatten()
            accuracy = (result['indices'] == indices).float().mean(1)
            for index in range(stop - start):
                rows.append({'image_id': identifiers[start + index], 'snr_db': snr, 'mse': float(mse[index]),
                             'lpips': float(lpips_values[index]), 'token_accuracy': float(accuracy[index]),
                             'header_usable': int(usable[start + index]), 'header_false_acceptance': int(false[start + index]),
                             'objective': float(mse[index] + 0.01 * lpips_values[index])})
    return rows, float(np.mean([row['objective'] for row in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/learned_prefix_jscc.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    requested = (args.output_dir or ROOT / config['outputs']['training']).resolve()
    if not requested.is_relative_to(ROOT / 'outputs'):
        raise ValueError('training output must remain under VAR_COMM')
    if requested.exists() and not args.resume:
        raise FileExistsError(requested)
    if (requested / 'completion.json').exists():
        raise FileExistsError('completed training is frozen')
    data = verify_artifacts(ROOT / config['data_cache'], 'completion.json')
    check = verify_artifacts(ROOT / config['outputs']['selfcheck'], 'selfcheck.json')
    verify_snapshot(check['source_hashes'])
    if check['status'] != 'IMAGE_GRADIENT_AND_FROZEN_MODEL_CHECK_PASS':
        raise RuntimeError('image gradient engineering check has not passed')
    if args.resume:
        output = requested
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_snapshot(metadata['source_hashes'])
    else:
        output = create_output(requested)
        paths = [Path(__file__), args.config, ROOT / 'reports/learned_prefix_training_protocol_2026-09-07.md',
                 ROOT / 'src/var_comm/learned_prefix.py', ROOT / 'src/var_comm/prefix_learning_support.py',
                 ROOT / 'src/var_comm/prefix_training_data.py', ROOT / 'src/var_comm/next_scale_prior.py']
        metadata = {'local_started': datetime.now().astimezone().isoformat(), 'command': sys.argv, 'source_hashes': snapshot(output, paths),
                    'data_completion_sha256': sha256(ROOT / config['data_cache'] / 'completion.json'),
                    'gradient_check_sha256': sha256(ROOT / config['outputs']['selfcheck'] / 'selfcheck.json'),
                    'torch': torch.__version__, 'numpy': np.__version__, 'DINO_training_or_selection': False}
        write_json(output / 'metadata.json', metadata)
        for variant in config['variants']:
            (output / variant / 'checkpoints').mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda:0')
    vae, var, perceptual = load_frozen_training_models(config, device)
    backbones = {'vae': vae, 'var': var, 'lpips': perceptual}
    before = {name: state_sha256(model) for name, model in backbones.items()}
    if before != check['frozen_state_sha256']:
        raise RuntimeError('frozen backbones differ from engineering check')
    codecs, optimizers, initial_hashes = {}, {}, {}
    for variant in config['variants']:
        torch.manual_seed(config['training']['initialization_seed'])
        codec = build_codec(config, vae, variant, device)
        initial_hashes[variant] = state_sha256(codec)
        codecs[variant] = codec
        optimizers[variant] = torch.optim.AdamW(codec.parameters(), lr=config['training']['learning_rate'],
                                               weight_decay=config['training']['weight_decay'], betas=tuple(config['training']['betas']))
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError('communication variants do not start from identical parameters')
    write_json(output / 'initialization.json', {'state_sha256': initial_hashes, 'trainable_counts': {name: sum(parameter.numel() for parameter in codec.parameters()) for name, codec in codecs.items()},
                                                'fresh_initialization_no_selfcheck_weights_reused': True, 'frozen_backbones': before})
    images, labels, identifiers, _bindings = read_image_population('train')
    calibration_images, calibration_labels, calibration_ids, _cal_bindings = read_image_population('calibration')
    with np.load(ROOT / config['data_cache'] / 'train.npz', allow_pickle=False) as cache:
        source_indices = torch.tensor(cache['source_indices'])
        source_tokens = torch.tensor(cache['tokens'].astype(np.int64))
        assert np.array_equal(labels[source_indices].numpy(), cache['labels'])
    with np.load(ROOT / config['data_cache'] / 'calibration.npz', allow_pickle=False) as cache:
        calibration_positions = torch.tensor(cache['source_indices'])
        calibration_tokens = torch.tensor(cache['tokens'][:, 0].astype(np.int64))
        assert np.array_equal(calibration_labels[calibration_positions].numpy(), cache['labels'])
    calibration_images = calibration_images[calibration_positions]
    calibration_labels = calibration_labels[calibration_positions]
    calibration_ids = [calibration_ids[index] for index in calibration_positions.tolist()]
    header = HeaderProtocol()
    calibration_noise = np.stack([seeded_noise(identifier, config['selection']['noise_seed'], (3060, 2)).astype(np.float32) for identifier in calibration_ids])
    write_json(output / 'population_binding.json', {'train_count': len(source_indices), 'calibration_count': len(calibration_positions),
                                                  'train_source_indices': source_indices.tolist(), 'calibration_source_indices': calibration_positions.tolist(),
                                                  'calibration_noise_sha256': hashlib.sha256(calibration_noise.tobytes()).hexdigest()})
    batch_size = config['training']['effective_batch_size']
    micro = config['training']['micro_batch_size']
    updates_per_epoch = math.ceil(len(source_indices) / batch_size)
    total_updates = updates_per_epoch * config['training']['epochs']
    histories, selections = {name: [] for name in codecs}, {name: None for name in codecs}
    start_epoch, next_position, global_step = 0, 0, 0
    channel_generator, teacher_generator = torch.Generator(), torch.Generator()
    resumed = None
    if args.resume:
        resumed = torch.load(output / 'resume.pt', map_location=device, weights_only=False)
        for name in codecs:
            codecs[name].load_state_dict(resumed['models'][name])
            optimizers[name].load_state_dict(resumed['optimizers'][name])
        start_epoch, next_position, global_step = resumed['epoch'], resumed['next_position'], resumed['global_step']
        histories, selections = resumed['histories'], resumed['selections']
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            ordering = torch.Generator().manual_seed(config['training']['order_seed'] + epoch)
            order = torch.randperm(len(source_indices), generator=ordering)
            flips = torch.rand(len(source_indices), generator=ordering) < 0.5
            channel_generator.manual_seed(config['training']['channel_seed'] + epoch)
            teacher_generator.manual_seed(config['training']['teacher_seed'] + epoch)
            position_start = next_position if epoch == start_epoch else 0
            if resumed is not None and epoch == start_epoch:
                channel_generator.set_state(resumed['channel_generator'].cpu())
                teacher_generator.set_state(resumed['teacher_generator'].cpu())
            for codec in codecs.values():
                codec.train()
            for start in range(position_start, len(order), batch_size):
                stop = min(start + batch_size, len(order))
                selected = order[start:stop]
                flip = flips[start:stop]
                tokens = source_tokens[selected, flip.long()]
                source_positions = source_indices[selected]
                source_labels = labels[source_positions]
                snr_indices = torch.randint(len(config['training']['snrs_db']), (len(selected),), generator=channel_generator)
                snrs_cpu = torch.tensor(config['training']['snrs_db'])[snr_indices]
                noise = torch.randn((len(selected), 3060, 2), generator=channel_generator)
                decoded, usable, false = header.decode(source_labels.numpy(), snrs_cpu.numpy(), noise[:, :68].numpy())
                teacher_probability = max(0.0, 1 - global_step / (0.4 * total_updates))
                teacher = (torch.rand(len(selected), generator=teacher_generator) < teacher_probability) & torch.tensor(usable)
                progress = global_step / max(1, total_updates - 1)
                learning_rate = config['training']['minimum_learning_rate'] + 0.5 * (config['training']['learning_rate'] - config['training']['minimum_learning_rate']) * (1 + math.cos(math.pi * progress))
                for variant, codec in codecs.items():
                    optimizer = optimizers[variant]
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate
                    optimizer.zero_grad(set_to_none=True)
                    totals = {'mse': 0.0, 'lpips': 0.0, 'token_ce': 0.0, 'token_accuracy': 0.0, 'gate': 0.0, 'teacher_samples': 0}
                    maximum_power_error = 0.0
                    tick = time.perf_counter()
                    for offset in range(0, len(selected), micro):
                        end = min(offset + micro, len(selected))
                        target = images[source_positions[offset:end]].to(device).float() / 255
                        flipping = flip[offset:end].to(device)
                        target = torch.where(flipping[:, None, None, None], target.flip(-1), target)
                        token_batch = tokens[offset:end].to(device)
                        snr_batch = snrs_cpu[offset:end].to(device)
                        label_batch = torch.tensor(decoded[offset:end], dtype=torch.long, device=device)
                        valid = torch.tensor(usable[offset:end], device=device)
                        teacher_mask = teacher[offset:end].to(device) if variant == 'next_scale' else None
                        teacher_active = teacher_mask is not None and bool(teacher_mask.any())
                        result = communication_forward(codec, token_batch, label_batch, snr_batch, noise[offset:end, 68:].to(device),
                                                       valid, vae, var, token_batch if teacher_active else None, teacher_mask if teacher_active else None)
                        loss, components = image_losses(result, target, token_batch, valid, perceptual, config['training'])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite training loss')
                        (loss * ((end - offset) / len(selected))).backward()
                        for name in ('mse', 'lpips', 'token_ce'):
                            totals[name] += float(components[name].detach().sum())
                        totals['token_accuracy'] += float((result['indices'] == token_batch).float().mean(1).detach().sum())
                        totals['gate'] += float(result['gates'].detach().flatten(1).mean(1).sum())
                        totals['teacher_samples'] += result['teacher_samples']
                        maximum_power_error = max(maximum_power_error, float((result['symbols'].detach().square().sum(-1).mean(-1) - 2).abs().max()))
                        del result, loss, components
                    norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), config['training']['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    if maximum_power_error > 1e-5 or not empty_cache(var):
                        raise RuntimeError('power or independent receiver state boundary failed')
                    record = {'epoch': epoch + 1, 'step': global_step + 1, 'samples_seen': epoch * len(order) + stop,
                              **{name: value / len(selected) for name, value in totals.items()}, 'teacher_probability': teacher_probability if variant == 'next_scale' else 0.0,
                              'header_usable_fraction': float(np.mean(usable)), 'header_false_acceptances': int(sum(false)),
                              'learning_rate': learning_rate, 'gradient_norm': float(norm), 'power_max_error': maximum_power_error,
                              'step_seconds': time.perf_counter() - tick}
                    histories[variant].append(record)
                global_step += 1
                if global_step % 100 == 0:
                    for name in codecs:
                        write_csv(output / name / 'training.csv', histories[name])
                if global_step % 250 == 0:
                    text = []
                    for name in codecs:
                        window = histories[name][-250:]
                        text.append(f"{name}: mse={np.mean([row['mse'] for row in window]):.5f} lpips={np.mean([row['lpips'] for row in window]):.4f} acc={np.mean([row['token_accuracy'] for row in window]):.3f}")
                    print(f'training {global_step}/{total_updates} teacher_p={teacher_probability:.3f} ' + ' | '.join(text), flush=True)
                if global_step % 1000 == 0:
                    torch.save({'epoch': epoch, 'next_position': stop, 'global_step': global_step,
                                'models': {name: codec.state_dict() for name, codec in codecs.items()}, 'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                                'channel_generator': channel_generator.get_state(), 'teacher_generator': teacher_generator.get_state(),
                                'histories': histories, 'selections': selections}, output / 'resume.pt')
            for name, codec in codecs.items():
                path = output / name / 'checkpoints' / f'epoch_{epoch + 1:02d}.pt'
                torch.save({'model': codec.state_dict(), 'variant': name, 'epoch': epoch + 1, 'global_step': global_step,
                            'initial_state_sha256': initial_hashes[name], 'config': config}, path)
                print(f'calibration epoch {epoch + 1} {name}: 1000 images × 5 SNR; own history only', flush=True)
                scores, objective = calibrate(codec, calibration_images, calibration_tokens, calibration_labels, calibration_ids, header,
                                              calibration_noise, vae, var, perceptual, config, device)
                write_csv(output / name / f'calibration_epoch_{epoch + 1:02d}.csv', scores)
                if selections[name] is None or objective < selections[name]['objective']:
                    selections[name] = {'epoch': epoch + 1, 'objective': objective, 'checkpoint': str(path.relative_to(output)), 'checkpoint_sha256': sha256(path)}
                write_json(output / name / 'selected.json', selections[name])
                write_csv(output / name / 'training.csv', histories[name])
                print(f'calibration {name} epoch={epoch + 1} objective={objective:.6f} selected={selections[name]["epoch"]}', flush=True)
            if before != {name: state_sha256(model) for name, model in backbones.items()}:
                raise RuntimeError('a frozen backbone changed during training')
            resumed = None
            next_position = 0
        if global_step != total_updates:
            raise RuntimeError('training stopped short of the authorized fixed budget')
        if any(parameter.grad is not None or parameter.requires_grad for model in backbones.values() for parameter in model.parameters()):
            raise RuntimeError('frozen model accumulated trainable gradients')
        verify_snapshot(metadata['source_hashes'])
        result = {'status': 'BOTH_PREFIX_MODELS_TRAINED_AND_CALIBRATED', 'optimizer_updates_per_variant': global_step,
                  'training_images_seen_per_variant': len(source_indices) * config['training']['epochs'], 'selected': selections,
                  'initial_state_sha256': initial_hashes, 'frozen_before': before, 'frozen_after': {name: state_sha256(model) for name, model in backbones.items()},
                  'last_epoch_teacher_samples': sum(row['teacher_samples'] * batch_size for row in histories['next_scale'] if row['epoch'] == config['training']['epochs']),
                  'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(device), 'elapsed_this_session_seconds': time.perf_counter() - started}
        write_json(output / 'completion.json', {**metadata, **result, 'local_completed': datetime.now().astimezone().isoformat(), 'output_hashes': artifact_hashes(output)})
        print(result, flush=True)
    except Exception as error:
        write_json(output / 'failure.json', {'error': repr(error), 'global_step': global_step, 'local_time': datetime.now().astimezone().isoformat()})
        raise


if __name__ == '__main__':
    main()
