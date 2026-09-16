#!/usr/bin/env python3
"""Audit learned-prefix physical budgets, received-only reconstruction and image statistics."""

import csv
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

from audit_progressive_channel import score_images
from evaluate_learned_prefix import deep_forward
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.prefix_learning_support import build_codec
from var_comm.prefix_training_data import HeaderProtocol, LEGACY
from var_comm.progressive import complete_image, split_prefix
from var_comm.quality import load_quality_models
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, snapshot, verify_artifacts, verify_snapshot, write_json

METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


@torch.no_grad()
def main():
    config = yaml.safe_load((ROOT / 'configs/learned_prefix_jscc.yaml').read_text())
    run = ROOT / config['outputs']['evaluation']
    receipt = verify_artifacts(run, 'completion.json')
    verify_snapshot(receipt['source_hashes'])
    training = ROOT / config['outputs']['training']
    trained = verify_artifacts(training, 'completion.json', receipt['training_completion_sha256'])
    verify_snapshot(trained['source_hashes'])
    output = create_output(ROOT / 'outputs/VAR-PREFIX-JSCC-EVAL-AUDIT-001')
    sources = snapshot(output, [Path(__file__), ROOT / 'scripts/audit_progressive_channel.py'])
    started = time.perf_counter()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda:0')
    model_paths = yaml.safe_load((ROOT / config['model_config']).read_text())['paths']
    quality = yaml.safe_load((ROOT / config['quality_config']).read_text())['quality']
    vae, var = load_models(model_paths, device)
    perceptual, dino, _weights = load_quality_models(quality, device)
    codecs = {}
    for name in config['variants']:
        selected = trained['selected'][name]
        model = build_codec(config, vae, name, device)
        checkpoint = torch.load(training / selected['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        codecs[name] = model.eval().requires_grad_(False)
        assert state_sha256(model) == receipt['frozen_before'][name]
        assert torch.equal(model.codebook, vae.quantize.embedding.weight)
    sys.path.insert(0, str(LEGACY / 'src'))
    from cadsd_jscc.exact_budget_strong_jscc import build_exact_budget_model
    deep = build_exact_budget_model(torch.load(config['deepjscc']['initialization_checkpoint'], map_location='cpu', weights_only=False))
    deep.load_state_dict(torch.load(config['deepjscc']['checkpoint'], map_location='cpu', weights_only=False)['model'])
    deep = deep.to(device).eval().requires_grad_(False)
    assert state_sha256(deep) == receipt['frozen_before']['deep']
    old = ROOT / 'outputs/VAR-PROGRESSIVE-CHANNEL-001'
    transition = ROOT / 'outputs/VAR-WHOLE-FRAME-PRIOR-001'
    targets = json.loads((old / 'populations.json').read_text())['target']
    with np.load(ROOT / 'outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/source_tokens.npz', allow_pickle=False) as cache:
        tokens = dict(zip(cache['image_ids'].tolist(), cache['tokens'].copy()))
    rows = read_csv(run / 'per_frame.csv')
    ablation_rows = {(int(row['image_index']), row['arm']): row for row in read_csv(run / 'shuffled_body.csv')}
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    names = ('prefix_parallel', 'prefix_next_scale', 'perceptual_deepjscc', 'digital_m8', 'digital_adaptive')
    expected = {(index, snr, seed, name) for index in range(100) for snr in config['evaluation']['snrs_db'] for seed in config['evaluation']['noise_seeds'] for name in names}
    assert len(rows) == len(lookup) == 10500 and set(lookup) == expected
    header = HeaderProtocol()
    scored, maximum_metrics = {}, np.zeros(3)
    max_render_error = max_replay_error = max_noise_error = 0.0
    rendered_count = neural_replays = metrics_count = deep_replays = ablation_replays = 0
    for index, target in enumerate(targets):
        directory = run / 'images' / f'{index:03d}'
        with np.load(directory / 'reconstructions.npz', allow_pickle=False) as cache:
            generated = cache['images'].copy()
        with np.load(directory / 'waveforms.npz', allow_pickle=False) as cache:
            waves = {name: cache[name] for name in cache.files}
        with np.load(directory / 'prefix_indices.npz', allow_pickle=False) as cache:
            predictions = {name: cache[name] for name in cache.files}
        with np.load(old / 'images' / f'{index:03d}' / 'reconstructions.npz', allow_pickle=False) as cache:
            old_images, source = cache['images'].copy(), cache['source'].copy()
        with np.load(transition / 'images' / f'{index:03d}' / 'reconstructions.npz', allow_pickle=False) as cache:
            transitional = cache['images'].copy()
        local_rows = [row for row in rows if int(row['image_index']) == index]
        references = sorted({row['image_ref'] for row in local_rows})
        source_arrays = {'new': generated, 'old': old_images, 'transition': transitional}
        images = np.stack([source_arrays[reference.split(':')[0]][int(reference.split(':')[1])] for reference in references])
        assert np.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
        values = score_images(source, images, perceptual, dino, device)
        metrics_count += len(images)
        for position, reference in enumerate(references):
            scored[index, reference] = {metric: float(scores[position]) for metric, scores in zip(METRICS, values)}
        for row in local_rows:
            maximum_metrics = np.maximum(maximum_metrics, [abs(float(row[metric]) - scored[index, row['image_ref']][metric]) for metric in METRICS])
            assert int(row['header_uses']) + int(row['data_uses']) == int(row['total_complex_uses']) == 3060
        for snr_index, snr in enumerate(config['evaluation']['snrs_db']):
            for seed_index, seed in enumerate(config['evaluation']['noise_seeds']):
                frame = f'frame_{snr_index}_{seed_index}'
                noise = seeded_noise(target['image_id'], seed, (3060, 2))
                decoded, usable, false = header.decode([target['class_index']], [snr], noise[None, :68])
                for name in names:
                    row = lookup[index, snr, seed, name]
                    assert row['noise_sha256'] == hashlib.sha256(noise.tobytes()).hexdigest()
                for variant, codec in codecs.items():
                    row = lookup[index, snr, seed, 'prefix_' + variant]
                    transmitted, received = waves[frame + '_' + variant + '_tx'], waves[frame + '_' + variant + '_rx']
                    assert transmitted.shape == received.shape == (2992, 2)
                    assert abs(np.mean(np.sum(transmitted.astype(np.float64) ** 2, axis=-1)) - 2) < 1e-5
                    error = float(np.max(np.abs(received - transmitted - noise[68:] / (10 ** (snr / 20)))))
                    max_noise_error = max(max_noise_error, error)
                    assert int(row['header_usable']) == int(usable[0]) and int(row['header_false_acceptance']) == int(false[0])
                    predicted = predictions[frame + '_' + variant]
                    if usable[0]:
                        assert int(row['received_class']) == int(decoded[0]) and predicted.shape == (255,)
                        assert abs(float(row['token_accuracy']) - np.mean(predicted == tokens[target['image_id']][:255])) < 1e-6
                        reference = complete_image(vae, var, split_prefix(predicted, 8), int(decoded[0]), device)
                        rendered_count += 1
                    else:
                        assert not len(predicted) and row['received_class'] == '' and float(row['token_accuracy']) == 0
                        reference = np.full((3, 256, 256), 0.5, dtype=np.float32)
                    image = generated[int(row['image_ref'].split(':')[1])]
                    max_render_error = max(max_render_error, float(np.max(np.abs(reference - image))))
                    if index % 5 == 0 and snr in (1.0, 7.0) and seed == 2001:
                        source_tokens = torch.tensor(tokens[target['image_id']][:255][None].astype(np.int64), device=device)
                        encoded = codec.transmit(source_tokens)[0].cpu().numpy()
                        assert np.max(np.abs(encoded - transmitted)) < 1e-6
                        if usable[0]:
                            result = codec.receive(torch.tensor(received[None], device=device), torch.tensor(decoded, dtype=torch.long, device=device), torch.tensor([snr], device=device), vae, var)
                            assert np.array_equal(result['indices'][0].cpu().numpy(), predicted)
                            max_replay_error = max(max_replay_error, float(np.max(np.abs(result['image'][0].cpu().numpy() - image))))
                        neural_replays += 1
                transmitted, received = waves[frame + '_deep_tx'], waves[frame + '_deep_rx']
                assert transmitted.shape == received.shape == (3060, 2)
                assert abs(np.mean(np.sum(transmitted.astype(np.float64) ** 2, axis=-1)) - 2) < 1e-5
                max_noise_error = max(max_noise_error, float(np.max(np.abs(received - transmitted - noise / (10 ** (snr / 20))))))
                if index % 5 == 0 and snr in (1.0, 7.0) and seed == 2001:
                    image, encoded, received_again = deep_forward(deep, torch.tensor(source[None], device=device), torch.tensor([snr], device=device),
                                                                   torch.tensor(noise[None], dtype=torch.float32, device=device))
                    assert np.max(np.abs(encoded[0].cpu().numpy() - transmitted)) < 1e-6
                    assert np.max(np.abs(received_again[0].cpu().numpy() - received)) < 1e-6
                    reference = generated[int(lookup[index, snr, seed, 'perceptual_deepjscc']['image_ref'].split(':')[1])]
                    max_replay_error = max(max_replay_error, float(np.max(np.abs(image[0].cpu().numpy() - reference))))
                    deep_replays += 1
        other = targets[(index + 1) % 100]
        other_tokens = torch.tensor(tokens[other['image_id']][:255][None].astype(np.int64), device=device)
        noise = seeded_noise(target['image_id'], 2001, (3060, 2))
        decoded, usable, _false = header.decode([target['class_index']], [7.0], noise[None, :68])
        for name, codec in codecs.items():
            shifted = codec.transmit(other_tokens) + torch.tensor(noise[None, 68:], dtype=torch.float32, device=device) / (10 ** 0.35)
            if usable[0]:
                result = codec.receive(shifted, torch.tensor(decoded, dtype=torch.long, device=device), torch.tensor([7.0], device=device), vae, var)
                image = result['image'][0].cpu().numpy()
            else:
                image = np.full((3, 256, 256), 0.5, dtype=np.float32)
            expected_image = np.load(directory / (name + '_shuffled_body.npy'))
            assert np.max(np.abs(image - expected_image)) < 1e-6
            values = score_images(source, image[None], perceptual, dino, device)
            row = ablation_rows[index, 'prefix_' + name]
            assert row['body_from_image_id'] == other['image_id']
            assert all(abs(float(row[metric]) - float(value[0])) < 2e-5 for metric, value in zip(METRICS, values))
            ablation_replays += 1
        print(f'learned-prefix audit {index + 1}/100 metrics={metrics_count} hard-prefix renders={rendered_count}', flush=True)
    assert maximum_metrics.max() < 2e-5 and max_render_error < 1e-4 and max_replay_error < 1e-6 and max_noise_error < 5e-6
    for item in read_csv(run / 'summary.csv'):
        selected = [row for row in rows if row['arm'] == item['arm'] and row['snr_db'] == item['snr_db']]
        assert len(selected) == 300
        for metric in METRICS:
            assert abs(float(item[metric]) - np.mean([scored[int(row['image_index']), row['image_ref']][metric] for row in selected])) < 2e-5
    draws = np.random.default_rng(config['evaluation']['bootstrap_seed']).integers(100, size=(10000, 100))
    max_interval_error = 0.0
    for item in read_csv(run / 'paired_quality.csv'):
        interval = [float(value) for value in item['snr_db'].split('+')]
        differences = []
        for index in range(100):
            values = []
            for snr in interval:
                for seed in config['evaluation']['noise_seeds']:
                    first = lookup[index, snr, seed, item['method']]
                    second = lookup[index, snr, seed, item['control']]
                    values.append(scored[index, first['image_ref']][item['metric']] - scored[index, second['image_ref']][item['metric']])
            differences.append(np.mean(values))
        differences = np.asarray(differences)
        low, high = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
        max_interval_error = max(max_interval_error, abs(differences.mean() - float(item['delta'])), abs(low - float(item['ci_low'])), abs(high - float(item['ci_high'])))
    assert max_interval_error < 2e-5
    for name, model in {'vae': vae, 'var': var, 'lpips': perceptual, 'dino': dino, 'deep': deep, **codecs}.items():
        assert state_sha256(model) == receipt['frozen_before'][name]
    verify_snapshot(sources)
    result = {'status': 'LEARNED_PREFIX_EVALUATION_AUDIT_PASS', 'rows_checked': len(rows), 'images_neurally_rescored': metrics_count,
              'hard_prefix_images_rerendered': rendered_count, 'full_learned_receiver_replays': neural_replays,
              'perceptual_DeepJSCC_replays': deep_replays, 'fixed_shuffled_body_replays': ablation_replays,
              'max_metric_errors_PSNR_LPIPS_DINO': maximum_metrics.tolist(), 'max_hard_prefix_render_error': max_render_error,
              'max_full_receiver_replay_error': max_replay_error, 'max_physical_noise_error': max_noise_error,
              'max_paired_interval_error': max_interval_error, 'evaluation_receipt_sha256': sha256(run / 'completion.json'),
              'elapsed_seconds': time.perf_counter() - started, 'source_hashes': sources, 'output_hashes': artifact_hashes(output)}
    write_json(output / 'audit.json', result)
    print({name: value for name, value in result.items() if name not in ('source_hashes', 'output_hashes')}, flush=True)


if __name__ == '__main__':
    main()
