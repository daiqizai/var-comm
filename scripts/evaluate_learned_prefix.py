#!/usr/bin/env python3
"""Evaluate completed learned-prefix models against frozen digital and perceptual JSCC controls."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
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
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.prefix_learning_support import build_codec
from var_comm.prefix_training_data import HeaderProtocol, LEGACY
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.study import artifact_hashes, create_output, paired_interval, seeded_noise, sha256, snapshot, verify_artifacts, verify_snapshot, write_csv, write_json

METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def deep_forward(model, source, snrs, noise):
    encoded = model.encode(source, snrs)
    normalized, _power = model.normalize_channel_input(encoded)
    flat = normalized.flatten(1)
    indices = model.active_real_indices.to(source.device)
    active = flat.index_select(1, indices)
    if active.shape[1] != 6120:
        raise RuntimeError('perceptual DeepJSCC is not a 3060-use model')
    received_active = active + noise.reshape_as(active) * torch.pow(10.0, snrs / 10).rsqrt()[:, None]
    received = torch.zeros_like(flat).index_copy(1, indices, received_active).reshape_as(normalized)
    return model.decode(received, snrs).clamp(0, 1), active.reshape(len(source), 3060, 2), received_active.reshape(len(source), 3060, 2)


def statistics(rows, config):
    names = ('prefix_parallel', 'prefix_next_scale', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc')
    lookup = {(row['image_index'], row['snr_db'], row['seed'], row['arm']): row for row in rows}
    expected = {(index, snr, seed, name) for index in range(100) for snr in config['evaluation']['snrs_db'] for seed in config['evaluation']['noise_seeds'] for name in names}
    if set(lookup) != expected or len(lookup) != len(rows):
        raise RuntimeError('incomplete learned-prefix evaluation grid')
    summary = []
    for snr in config['evaluation']['snrs_db']:
        for name in names:
            selected = [row for row in rows if row['snr_db'] == snr and row['arm'] == name]
            result = {'snr_db': snr, 'arm': name, 'transmissions': len(selected), 'source_images': 100}
            result.update({metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS})
            for field in ('token_accuracy', 'header_usable', 'header_false_acceptance', 'inference_seconds', 'data_power'):
                values = [row[field] for row in selected if row[field] != '']
                result[field] = float(np.mean(values)) if values else None
            summary.append(result)
    contrasts = [('prefix_next_scale', 'prefix_parallel')]
    contrasts.extend((method, control) for method in ('prefix_parallel', 'prefix_next_scale') for control in ('digital_m8', 'digital_adaptive', 'perceptual_deepjscc'))
    paired = []
    for interval in [config['evaluation']['primary_snr_db'], *[[snr] for snr in config['evaluation']['snrs_db']]]:
        for method, control in contrasts:
            for metric in METRICS:
                values = [np.mean([lookup[index, snr, seed, method][metric] - lookup[index, snr, seed, control][metric]
                                   for snr in interval for seed in config['evaluation']['noise_seeds']]) for index in range(100)]
                result = paired_interval(values, config['evaluation']['bootstrap_seed'], config['evaluation']['bootstrap_resamples'])
                paired.append({'snr_db': '+'.join(str(value) for value in interval), 'method': method, 'control': control,
                               'metric': metric, 'delta': result['gain'], 'ci_low': result['ci_low'], 'ci_high': result['ci_high']})
    return summary, paired


@torch.no_grad()
def main():
    config = yaml.safe_load((ROOT / 'configs/learned_prefix_jscc.yaml').read_text())
    training = ROOT / config['outputs']['training']
    receipt = verify_artifacts(training, 'completion.json')
    verify_snapshot(receipt['source_hashes'])
    if receipt['status'] != 'BOTH_PREFIX_MODELS_TRAINED_AND_CALIBRATED' or receipt['optimizer_updates_per_variant'] != 10000:
        raise RuntimeError('both fixed-budget training runs must finish before development evaluation')
    output = create_output(ROOT / config['outputs']['evaluation'])
    sources = snapshot(output, [Path(__file__), ROOT / 'configs/learned_prefix_jscc.yaml'])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda:0')
    paths = yaml.safe_load((ROOT / config['model_config']).read_text())['paths']
    quality = yaml.safe_load((ROOT / config['quality_config']).read_text())['quality']
    vae, var = load_models(paths, device)
    perceptual, dino, _weights = load_quality_models(quality, device)
    codecs = {}
    for variant in config['variants']:
        selected = receipt['selected'][variant]
        checkpoint = training / selected['checkpoint']
        if sha256(checkpoint) != selected['checkpoint_sha256']:
            raise RuntimeError('selected communication checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        codec = build_codec(config, vae, variant, device)
        codec.load_state_dict(state['model'], strict=True)
        codecs[variant] = codec.eval().requires_grad_(False)
    sys.path.insert(0, str(ROOT / 'src'))
    from cadsd_jscc.exact_budget_strong_jscc import build_exact_budget_model
    for key in ('checkpoint', 'initialization_checkpoint'):
        expected = config['deepjscc']['checkpoint_sha256' if key == 'checkpoint' else 'initialization_sha256']
        if sha256(config['deepjscc'][key]) != expected:
            raise RuntimeError('perceptual DeepJSCC checkpoint changed')
    initial = torch.load(config['deepjscc']['initialization_checkpoint'], map_location='cpu', weights_only=False)
    deep = build_exact_budget_model(initial)
    deep_state = torch.load(config['deepjscc']['checkpoint'], map_location='cpu', weights_only=False)
    deep.load_state_dict(deep_state['model'], strict=True)
    deep = deep.to(device).eval().requires_grad_(False)
    models = {'vae': vae, 'var': var, 'lpips': perceptual, 'dino': dino, 'deep': deep, **codecs}
    before = {name: state_sha256(model) for name, model in models.items()}
    if any(before[name] != receipt['frozen_before'][name] for name in ('vae', 'var', 'lpips')):
        raise RuntimeError('evaluation changed the frozen training backbones')
    old = ROOT / 'outputs/VAR-PROGRESSIVE-CHANNEL-001'
    transition = ROOT / 'outputs/VAR-WHOLE-FRAME-PRIOR-001'
    old_receipt = verify_artifacts(old, 'completion.json')
    transition_receipt = verify_artifacts(transition, 'completion.json')
    old_rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in read_csv(old / 'per_frame.csv')}
    transition_rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in read_csv(transition / 'per_frame.csv')}
    targets = json.loads((old / 'populations.json').read_text())['target']
    with np.load(ROOT / 'outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/source_tokens.npz', allow_pickle=False) as cache:
        tokens = dict(zip(cache['image_ids'].tolist(), cache['tokens'].copy()))
    header = HeaderProtocol()
    rows, ablations, maximum_power_error = [], [], 0.0
    for index, target in enumerate(targets):
        directory = output / 'images' / f'{index:03d}'
        directory.mkdir(parents=True)
        with np.load(old / 'images' / f'{index:03d}' / 'reconstructions.npz', allow_pickle=False) as cache:
            original_images, original = cache['images'].copy(), cache['source'].copy()
        with np.load(transition / 'images' / f'{index:03d}' / 'reconstructions.npz', allow_pickle=False) as cache:
            transition_images = cache['images'].copy()
        transition_frames = {(entry['snr_db'], entry['seed']): entry['receivers'] for entry in json.loads((transition / 'images' / f'{index:03d}' / 'receivers.json').read_text())}
        source = torch.tensor(original[None], device=device)
        source_tokens = torch.tensor(tokens[target['image_id']][:255][None].astype(np.int64), device=device)
        rendered, waveforms, predictions, local_rows, image_refs = [], {}, {}, [], {}
        for snr_index, snr in enumerate(config['evaluation']['snrs_db']):
            for seed_index, seed in enumerate(config['evaluation']['noise_seeds']):
                frame = f'frame_{snr_index}_{seed_index}'
                noise = seeded_noise(target['image_id'], seed, (3060, 2))
                decoded, usable, false = header.decode([target['class_index']], [snr], noise[None, :68])
                label = torch.tensor(decoded, dtype=torch.long, device=device)
                snrs = torch.full((1,), snr, device=device)
                normal = torch.tensor(noise[None], device=device, dtype=torch.float32)
                common = {'image_index': index, 'image_id': target['image_id'], 'snr_db': snr, 'seed': seed, 'total_complex_uses': 3060,
                          'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest()}
                for variant, codec in codecs.items():
                    tick = time.perf_counter()
                    transmitted = codec.transmit(source_tokens)
                    received = transmitted + normal[:, 68:] * torch.pow(10.0, snrs / 10).rsqrt()[:, None, None]
                    if usable[0]:
                        result = codec.receive(received, label, snrs, vae, var)
                        image, indices = result['image'], result['indices']
                    else:
                        image, indices = torch.full((1, 3, 256, 256), 0.5, device=device), None
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - tick
                    arm = 'prefix_' + variant
                    reference = 'new:' + str(len(rendered))
                    rendered.append(image[0].cpu().numpy())
                    data_power = float(transmitted.square().sum(-1).mean())
                    maximum_power_error = max(maximum_power_error, abs(data_power - 2))
                    waveforms[frame + '_' + variant + '_tx'] = transmitted[0].cpu().numpy()
                    waveforms[frame + '_' + variant + '_rx'] = received[0].cpu().numpy()
                    predictions[frame + '_' + variant] = np.empty(0, dtype=np.uint16) if indices is None else indices[0].cpu().numpy().astype(np.uint16)
                    local_rows.append({**common, 'arm': arm, 'header_uses': 68, 'data_uses': 2992, 'header_usable': int(usable[0]),
                                       'header_false_acceptance': int(false[0]), 'received_class': int(decoded[0]) if usable[0] else '',
                                       'token_accuracy': float((indices == source_tokens).float().mean()) if indices is not None else 0.0,
                                       'data_power': data_power, 'inference_seconds': elapsed, 'image_ref': reference})
                tick = time.perf_counter()
                image, transmitted, received = deep_forward(deep, source, snrs, normal)
                torch.cuda.synchronize(device)
                reference = 'new:' + str(len(rendered))
                rendered.append(image[0].cpu().numpy())
                power = float(transmitted.square().sum(-1).mean())
                maximum_power_error = max(maximum_power_error, abs(power - 2))
                waveforms[frame + '_deep_tx'], waveforms[frame + '_deep_rx'] = transmitted[0].cpu().numpy(), received[0].cpu().numpy()
                local_rows.append({**common, 'arm': 'perceptual_deepjscc', 'header_uses': 0, 'data_uses': 3060, 'header_usable': '',
                                   'header_false_acceptance': '', 'received_class': '', 'token_accuracy': '', 'data_power': power,
                                   'inference_seconds': time.perf_counter() - tick, 'image_ref': reference})
                for arm, previous_arm in (('digital_m8', 'whole_m8'), ('digital_adaptive', 'whole_adaptive')):
                    if snr in (5.0, 6.0):
                        record = transition_rows[index, snr, seed, previous_arm]
                        kind, location = record['image_ref'].split(':')
                        reference = ('old:' if kind == 'input' else 'transition:') + location
                        image_refs[reference] = (original_images if kind == 'input' else transition_images)[int(location)]
                        valid_header = int(transition_frames[snr, seed][previous_arm]['header']['accepted'])
                        false_header = int(record['header_false_acceptance'])
                    else:
                        record = old_rows[index, snr, seed, previous_arm]
                        reference = 'old:' + record['reconstruction_index']
                        image_refs[reference] = original_images[int(record['reconstruction_index'])]
                        valid_header = int(record['header_accepted'])
                        false_header = int(record['header_false_acceptance'])
                    local_rows.append({**common, 'arm': arm, 'header_uses': 68, 'data_uses': 2992, 'header_usable': valid_header,
                                       'header_false_acceptance': false_header, 'received_class': '', 'token_accuracy': '',
                                       'data_power': 2.0, 'inference_seconds': '', 'image_ref': reference})
        references = [*['new:' + str(position) for position in range(len(rendered))], *sorted(image_refs)]
        images = rendered + [image_refs[reference] for reference in sorted(image_refs)]
        scores, _source_feature, _features = quality_metrics(original, images, perceptual, dino, device)
        metric_lookup = dict(zip(references, scores))
        for row in local_rows:
            row.update(metric_lookup[row['image_ref']])
        rows.extend(local_rows)
        np.savez_compressed(directory / 'reconstructions.npz', images=np.stack(rendered))
        np.savez(directory / 'waveforms.npz', **waveforms)
        np.savez(directory / 'prefix_indices.npz', **predictions)
        write_csv(output / 'per_frame.csv', rows)
        print(f'learned-prefix evaluation {index + 1}/100 rows={len(rows)}', flush=True)
        other = targets[(index + 1) % 100]
        other_tokens = torch.tensor(tokens[other['image_id']][:255][None].astype(np.int64), device=device)
        noise = seeded_noise(target['image_id'], 2001, (3060, 2))
        decoded, usable, _false = header.decode([target['class_index']], [7.0], noise[None, :68])
        for variant, codec in codecs.items():
            transmitted = codec.transmit(other_tokens)
            received = transmitted + torch.tensor(noise[None, 68:], device=device, dtype=torch.float32) / math_sqrt_gamma(7.0)
            result = codec.receive(received, torch.tensor(decoded, dtype=torch.long, device=device), torch.tensor([7.0], device=device), vae, var) if usable[0] else None
            image = result['image'][0].cpu().numpy() if result is not None else np.full((3, 256, 256), 0.5, dtype=np.float32)
            values, _source_feature, _features = quality_metrics(original, [image], perceptual, dino, device)
            np.save(directory / (variant + '_shuffled_body.npy'), image)
            ablations.append({'image_index': index, 'image_id': target['image_id'], 'body_from_image_id': other['image_id'],
                              'arm': 'prefix_' + variant, 'snr_db': 7.0, 'seed': 2001, **values[0]})
        if not empty_cache(var):
            raise RuntimeError('evaluation leaked a previous VAR state')
    summary, paired = statistics(rows, config)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'paired_quality.csv', paired)
    write_csv(output / 'shuffled_body.csv', ablations)
    if maximum_power_error > 1e-5 or before != {name: state_sha256(model) for name, model in models.items()}:
        raise RuntimeError('evaluation changed power or model weights')
    verify_snapshot(sources)
    write_json(output / 'completion.json', {'status': 'TRAINED_PREFIX_JSCC_DEVELOPMENT_EVALUATION_COMPLETE', 'rows': len(rows), 'source_images': 100,
                                           'training_completion_sha256': sha256(training / 'completion.json'), 'source_hashes': sources,
                                           'frozen_before': before, 'frozen_after': before, 'maximum_data_power_error': maximum_power_error,
                                           'old_digital_receipt_sha256': sha256(old / 'completion.json'), 'transition_receipt_sha256': sha256(transition / 'completion.json'),
                                           'no_new_test_data': True, 'no_teacher_at_evaluation': True, 'output_hashes': artifact_hashes(output)})


def math_sqrt_gamma(snr):
    return 10 ** (snr / 20)


if __name__ == '__main__':
    main()
