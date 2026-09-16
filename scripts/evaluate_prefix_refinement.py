#!/usr/bin/env python3
"""Evaluate frozen refinement selections using the unchanged hard next-scale receiver."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
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
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.prefix_learning_support import build_codec
from var_comm.prefix_refinement import receive_prefix
from var_comm.prefix_training_data import HeaderProtocol
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.study import artifact_hashes, create_output, paired_interval, seeded_noise, sha256, snapshot, verify_artifacts, verify_snapshot, write_csv, write_json

METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def statistics(rows, config):
    names = config['branches'] + config['evaluation']['controls']
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    snrs, seeds = config['evaluation']['snrs_db'], config['evaluation']['noise_seeds']
    expected = {(index, snr, seed, name) for index in range(100) for snr in snrs for seed in seeds for name in names}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise RuntimeError('incomplete or duplicated refinement evaluation grid')
    summaries, paired = [], []
    for snr in snrs:
        for name in names:
            selected = [lookup[index, snr, seed, name] for index in range(100) for seed in seeds]
            summaries.append({'snr_db': snr, 'arm': name, 'transmissions': len(selected), 'source_images': 100,
                              **{metric: float(np.mean([float(row[metric]) for row in selected])) for metric in METRICS},
                              'header_usable': float(np.mean([float(row['header_usable']) for row in selected])),
                              'header_false_acceptance': float(np.mean([float(row['header_false_acceptance']) for row in selected]))})
    comparisons = [('two_stage', 'continuation')] + [(branch, control) for branch in config['branches'] for control in config['evaluation']['controls']]
    for group in [config['evaluation']['primary_snr_db'], *[[snr] for snr in snrs]]:
        for method, control in comparisons:
            for metric in METRICS:
                differences = [np.mean([float(lookup[index, snr, seed, method][metric]) - float(lookup[index, snr, seed, control][metric]) for snr in group for seed in seeds]) for index in range(100)]
                interval = paired_interval(differences, config['evaluation']['bootstrap_seed'], config['evaluation']['bootstrap_resamples'])
                paired.append({'snr_db': '+'.join(str(snr) for snr in group), 'method': method, 'control': control, 'metric': metric,
                               'delta': interval['gain'], 'ci_low': interval['ci_low'], 'ci_high': interval['ci_high']})
    return summaries, paired


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    config = yaml.safe_load((ROOT / 'configs/prefix_refinement.yaml').read_text())
    base = yaml.safe_load((ROOT / config['base_config']).read_text())
    training, previous = ROOT / config['outputs']['training'], ROOT / config['evaluation']['old_evaluation']
    receipt = verify_artifacts(training, 'completion.json')
    if receipt['status'] != 'MATCHED_PREFIX_REFINEMENT_TRAINING_COMPLETE' or receipt['additional_updates_per_branch'] != 10000:
        raise RuntimeError('full training and calibration must finish before development evaluation')
    verify_snapshot(receipt['source_hashes'])
    verify_artifacts(previous, 'completion.json', config['evaluation']['old_evaluation_receipt_sha256'])
    requested = ROOT / config['outputs']['evaluation']
    if (requested / 'completion.json').exists():
        raise FileExistsError('completed refinement evaluation is frozen')
    if arguments.resume:
        output = requested
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_snapshot(metadata['source_hashes'])
    else:
        output = create_output(requested)
        metadata = {'source_hashes': snapshot(output, [Path(__file__), ROOT / 'configs/prefix_refinement.yaml']),
                    'training_receipt_sha256': sha256(training / 'completion.json'), 'local_started': datetime.now().astimezone().isoformat()}
        write_json(output / 'metadata.json', metadata)
    if metadata['training_receipt_sha256'] != sha256(training / 'completion.json'):
        raise RuntimeError('resume metadata belongs to another training run')
    while True:
        available = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '--id=0'], text=True).strip())
        if available >= config['execution']['minimum_free_GPU_MiB']:
            break
        write_json(output / 'status.json', {'status': 'WAITING_FOR_GPU_FOR_EVALUATION', 'free_GPU_MiB': available})
        print(f'evaluation waiting for GPU: {available} MiB free; no model loaded', flush=True)
        time.sleep(config['execution']['GPU_wait_poll_seconds'])
    write_json(output / 'status.json', {'status': 'EVALUATING', 'completed_source_images': 0})
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda:0')
    paths = yaml.safe_load((ROOT / base['model_config']).read_text())['paths']
    quality = yaml.safe_load((ROOT / base['quality_config']).read_text())['quality']
    vae, var = load_models(paths, device)
    perceptual, dino, weights = load_quality_models(quality, device)
    codecs = {}
    for branch in config['branches']:
        selected = receipt['selected'][branch]
        path = training / selected['checkpoint']
        if sha256(path) != selected['checkpoint_sha256']:
            raise RuntimeError('selected model changed after calibration')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        codec = build_codec(base, vae, 'next_scale', device)
        codec.load_state_dict(checkpoint['model'], strict=True)
        codecs[branch] = codec.eval().requires_grad_(False)
    models = {'vae': vae, 'var': var, 'lpips': perceptual, 'dino': dino, **codecs}
    before = {name: state_sha256(model) for name, model in models.items()}
    if any(before[name] != receipt['frozen_after'][name] for name in ('vae', 'var', 'lpips')):
        raise RuntimeError('evaluation changed frozen backbones')
    old_rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in read_csv(previous / 'per_frame.csv')}
    origin = ROOT / 'outputs/VAR-PROGRESSIVE-CHANNEL-001'
    origin_receipt = json.loads((origin / 'completion.json').read_text())
    if sha256(origin / 'populations.json') != origin_receipt['output_hashes']['populations.json']:
        raise RuntimeError('development population manifest changed')
    targets = json.loads((origin / 'populations.json').read_text())['target']
    token_root = ROOT / 'outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001'
    token_receipt = json.loads((token_root / 'completion.json').read_text())
    if sha256(token_root / 'source_tokens.npz') != token_receipt['output_hashes']['source_tokens.npz']:
        raise RuntimeError('development source tokens changed')
    with np.load(token_root / 'source_tokens.npz', allow_pickle=False) as cache:
        tokens = dict(zip(cache['image_ids'].tolist(), cache['tokens'].copy()))
    header = HeaderProtocol()
    rows = []
    maximum_power_error, replay_count = 0.0, 0
    for index, target in enumerate(targets):
        directory = output / 'images' / f'{index:03d}'
        if (directory / 'receipt.json').exists():
            local_receipt = verify_artifacts(directory, 'receipt.json')
            rows.extend(read_csv(directory / 'per_frame.csv'))
            maximum_power_error = max(maximum_power_error, local_receipt['maximum_power_error'])
            replay_count += local_receipt['new_wrapper_replays']
            continue
        directory.mkdir(parents=True, exist_ok=True)
        original_path = origin / 'images' / f'{index:03d}' / 'reconstructions.npz'
        if sha256(original_path) != origin_receipt['output_hashes'][str(original_path.relative_to(origin))]:
            raise RuntimeError('development source image changed')
        with np.load(original_path, allow_pickle=False) as cache:
            original = cache['source'].copy()
        source_tokens = torch.tensor(tokens[target['image_id']][:255][None].astype(np.int64), device=device)
        rendered, references, predictions, waveforms, local_rows = [], {}, {}, {}, []
        local_power_error, local_replays = 0.0, 0
        for snr_index, snr in enumerate(config['evaluation']['snrs_db']):
            for seed_index, seed in enumerate(config['evaluation']['noise_seeds']):
                frame = f'frame_{snr_index}_{seed_index}'
                noise = seeded_noise(target['image_id'], seed, (3060, 2))
                noise_hash = hashlib.sha256(noise.tobytes()).hexdigest()
                baseline = old_rows[index, snr, seed, 'prefix_next_scale']
                if noise_hash != baseline['noise_sha256']:
                    raise RuntimeError('development source/noise pairing changed')
                decoded, usable, false = header.decode([target['class_index']], [snr], noise[None, :68])
                if int(usable[0]) != int(baseline['header_usable']) or int(false[0]) != int(baseline['header_false_acceptance']):
                    raise RuntimeError('fixed-m8 header differs from the original chain')
                condition = torch.tensor(decoded, device=device)
                snrs = torch.full((1,), snr, device=device)
                normal = torch.tensor(noise[None, 68:], dtype=torch.float32, device=device)
                common = {'image_index': index, 'image_id': target['image_id'], 'snr_db': snr, 'seed': seed,
                          'total_complex_uses': 3060, 'noise_sha256': noise_hash, 'header_uses': 68, 'data_uses': 2992}
                for branch, codec in codecs.items():
                    torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    transmitted = codec.transmit(source_tokens)
                    received = transmitted + normal * torch.pow(10.0, snrs / 10).rsqrt()[:, None, None]
                    result = codec.receive(received, condition, snrs, vae, var) if usable[0] else None
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - tick
                    image = result['image'][0].cpu().numpy() if result is not None else np.full((3, 256, 256), 0.5, dtype=np.float32)
                    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    if image_hash not in references:
                        references[image_hash] = len(rendered)
                        rendered.append(image)
                    image_ref = 'new:' + str(references[image_hash])
                    power = float(transmitted.square().sum(-1).mean())
                    local_power_error = max(local_power_error, abs(power - 2))
                    receiver_ref = f'{frame}_{branch}'
                    waveforms[receiver_ref + '_tx'] = transmitted[0].cpu().numpy()
                    waveforms[receiver_ref + '_rx'] = received[0].cpu().numpy()
                    predictions[receiver_ref] = result['indices'][0].cpu().numpy().astype(np.uint16) if result is not None else np.full(255, 65535, dtype=np.uint16)
                    if index < 2 and seed == 2001 and snr in (1., 7., 19.) and result is not None:
                        replay = receive_prefix(codec, received, condition, snrs, vae, var)
                        if not torch.equal(replay['indices'], result['indices']) or not torch.equal(replay['image'], result['image']):
                            raise RuntimeError('trained state-tracking receiver changed the hard deployment output')
                        local_replays += 1
                    local_rows.append({**common, 'arm': branch, 'header_usable': int(usable[0]), 'header_false_acceptance': int(false[0]),
                                       'received_class': int(decoded[0]) if usable[0] else -1,
                                       'token_accuracy': float((result['indices'] == source_tokens).float().mean()) if result is not None else 0.0,
                                       'data_power': power, 'inference_seconds': elapsed, 'image_ref': image_ref,
                                       'psnr_db': '', 'lpips_alex': '', 'dino_cosine': '', 'image_sha256': image_hash,
                                       'receiver_ref': receiver_ref, 'checkpoint_additional_step': receipt['selected'][branch]['additional_step']})
                for arm, old_arm in [('starting_next_scale', 'prefix_next_scale'), ('digital_m8', 'digital_m8'), ('digital_adaptive', 'digital_adaptive')]:
                    saved = old_rows[index, snr, seed, old_arm]
                    if saved['noise_sha256'] != noise_hash or int(saved['total_complex_uses']) != 3060:
                        raise RuntimeError('frozen control pairing or budget changed')
                    local_rows.append({**common, 'arm': arm, **{key: saved[key] for key in ('header_usable', 'header_false_acceptance', 'received_class', 'token_accuracy', 'data_power', 'inference_seconds')},
                                       'image_ref': 'previous:' + saved['image_ref'], **{metric: float(saved[metric]) for metric in METRICS},
                                       'image_sha256': '', 'receiver_ref': '', 'checkpoint_additional_step': 0 if arm == 'starting_next_scale' else ''})
        scores, source_feature, features = quality_metrics(original, rendered, perceptual, dino, device)
        for row in local_rows:
            if row['arm'] in codecs:
                row.update(scores[int(row['image_ref'].split(':')[1])])
        np.savez_compressed(directory / 'reconstructions.npz', images=np.stack(rendered))
        np.savez(directory / 'waveforms.npz', **waveforms)
        np.savez(directory / 'prefix_indices.npz', **predictions)
        write_csv(directory / 'per_frame.csv', local_rows)
        write_json(directory / 'receipt.json', {'maximum_power_error': local_power_error, 'new_wrapper_replays': local_replays,
                   'output_hashes': artifact_hashes(directory)})
        rows.extend(local_rows)
        maximum_power_error = max(maximum_power_error, local_power_error)
        replay_count += local_replays
        if not empty_cache(var):
            raise RuntimeError('evaluation contaminated the next receiver state')
        write_csv(output / 'per_frame.csv', rows)
        write_json(output / 'status.json', {'status': 'EVALUATING', 'completed_source_images': index + 1})
        print(f'refinement development {index + 1}/100 rows={len(rows)}', flush=True)
    summary, paired = statistics(rows, config)
    write_csv(output / 'per_frame.csv', rows)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'paired_quality.csv', paired)
    after = {name: state_sha256(model) for name, model in models.items()}
    if maximum_power_error > 1e-5 or before != after:
        raise RuntimeError('evaluation changed power or frozen weights')
    verify_snapshot(metadata['source_hashes'])
    write_json(output / 'status.json', {'status': 'EVALUATION_COMPLETE', 'completed_source_images': 100})
    write_json(output / 'completion.json', {'status': 'PREFIX_REFINEMENT_DEVELOPMENT_COMPLETE', 'rows': len(rows), 'source_images': 100,
               'source_hashes': metadata['source_hashes'], 'training_receipt_sha256': sha256(training / 'completion.json'),
               'old_evaluation_receipt_sha256': sha256(previous / 'completion.json'), 'maximum_power_error': maximum_power_error,
               'new_wrapper_replays': replay_count, 'frozen_before': before, 'frozen_after': after, 'no_new_test_data': True,
               'output_hashes': artifact_hashes(output)})
    print('PREFIX_REFINEMENT_DEVELOPMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
