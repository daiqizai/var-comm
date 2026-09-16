#!/usr/bin/env python3
"""Post-evaluation baseline sanity: nearest trained NN condition, unchanged actual AWGN."""

import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

from var_comm.next_scale_prior import state_sha256
from var_comm.prefix_training_data import LEGACY
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.study import artifact_hashes, create_output, paired_interval, seeded_noise, sha256, snapshot, verify_artifacts, write_csv, write_json


@torch.no_grad()
def main():
    config = yaml.safe_load((ROOT / 'configs/learned_prefix_jscc.yaml').read_text())
    main_run = ROOT / config['outputs']['evaluation']
    receipt = verify_artifacts(main_run, 'completion.json')
    output = create_output(ROOT / 'outputs/VAR-PREFIX-DEEP-COND-SANITY-001')
    sources = snapshot(output, [Path(__file__)])
    write_json(output / 'protocol.json', {'role': 'post-evaluation strong-control sanity, not a new trained arm',
                                         'physical_snr_db': [5.0, 6.0], 'NN_condition_db': {'5.0': 4.0, '6.0': 7.0},
                                         'rule': 'nearest SNR from frozen training support, no per-image or metric selection',
                                         'physical_noise': 'unchanged real variance 1/gamma_actual', 'total_data_complex_uses': 3060,
                                         'trained_weights_changed': False, 'primary_1_4_7_results_changed': False})
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    sys.path.insert(0, str(LEGACY / 'src'))
    from cadsd_jscc.exact_budget_strong_jscc import build_exact_budget_model
    initial = torch.load(config['deepjscc']['initialization_checkpoint'], map_location='cpu', weights_only=False)
    model = build_exact_budget_model(initial)
    model.load_state_dict(torch.load(config['deepjscc']['checkpoint'], map_location='cpu', weights_only=False)['model'])
    model = model.to(device).eval().requires_grad_(False)
    before = state_sha256(model)
    assert before == receipt['frozen_before']['deep']
    paths = yaml.safe_load((ROOT / config['quality_config']).read_text())['quality']
    perceptual, dino, _weights = load_quality_models(paths, device)
    parent = ROOT / 'outputs/VAR-PROGRESSIVE-CHANNEL-001'
    targets = json.loads((parent / 'populations.json').read_text())['target']
    rows, max_noise_error = [], 0.0
    for index, target in enumerate(targets):
        with np.load(parent / 'images' / f'{index:03d}' / 'reconstructions.npz', allow_pickle=False) as cache:
            original = cache['source'].copy()
        source = torch.tensor(original[None], device=device)
        images, local = [], []
        for actual_snr, nominal_snr in ((5.0, 4.0), (6.0, 7.0)):
            condition = torch.tensor([nominal_snr], device=device)
            encoded = model.encode(source, condition)
            transmitted, _power = model.normalize_channel_input(encoded)
            flat = transmitted.flatten(1)
            active = flat.index_select(1, model.active_real_indices)
            assert active.shape == (1, 6120) and abs(float(active.square().mean()) - 1) < 1e-5
            for seed in config['evaluation']['noise_seeds']:
                noise = seeded_noise(target['image_id'], seed, (3060, 2))
                standard = torch.tensor(noise.reshape(1, 6120), device=device, dtype=torch.float32)
                received_active = active + standard / (10 ** (actual_snr / 20))
                max_noise_error = max(max_noise_error, float((received_active - active - standard / (10 ** (actual_snr / 20))).abs().max()))
                received = torch.zeros_like(flat).index_copy(1, model.active_real_indices, received_active).reshape_as(encoded)
                images.append(model.decode(received, condition).clamp(0, 1)[0].cpu().numpy())
                local.append({'image_index': index, 'image_id': target['image_id'], 'snr_db': actual_snr, 'condition_snr_db': nominal_snr,
                              'seed': seed, 'total_complex_uses': 3060, 'data_power': float(active.square().mean() * 2)})
        scores, _source_feature, _features = quality_metrics(original, images, perceptual, dino, device)
        for row, values in zip(local, scores):row.update(values)
        rows.extend(local)
        np.savez_compressed(output / f'images_{index:03d}.npz', images=np.stack(images))
    write_csv(output / 'per_frame.csv', rows)
    summary = []
    for snr in (5.0, 6.0):
        selected = [row for row in rows if row['snr_db'] == snr]
        summary.append({'snr_db': snr, 'condition_snr_db': 4.0 if snr == 5 else 7.0, 'transmissions': len(selected),
                        **{metric: float(np.mean([row[metric] for row in selected])) for metric in ('psnr_db', 'lpips_alex', 'dino_cosine')}})
    write_csv(output / 'summary.csv', summary)
    with (main_run / 'per_frame.csv').open() as handle:
        primary = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in csv.DictReader(handle)}
    contrasts = []
    for snr in (5.0, 6.0):
        for metric in ('psnr_db', 'lpips_alex', 'dino_cosine'):
            differences = []
            for index in range(100):
                entries = [row for row in rows if row['image_index'] == index and row['snr_db'] == snr]
                differences.append(np.mean([float(primary[index, snr, row['seed'], 'prefix_next_scale'][metric]) - row[metric] for row in entries]))
            interval = paired_interval(differences, config['evaluation']['bootstrap_seed'], config['evaluation']['bootstrap_resamples'])
            contrasts.append({'snr_db': snr, 'metric': metric, 'next_minus_support_aware_DeepJSCC': interval['gain'], 'ci_low': interval['ci_low'], 'ci_high': interval['ci_high']})
    write_csv(output / 'paired_vs_next.csv', contrasts)
    assert before == state_sha256(model) and max_noise_error < 1e-6
    write_json(output / 'completion.json', {'status': 'DEEP_SNR_SUPPORT_SANITY_COMPLETE', 'rows': len(rows), 'frozen_deep_state': before,
                                            'main_evaluation_receipt_sha256': sha256(main_run / 'completion.json'), 'max_noise_error': max_noise_error,
                                            'source_hashes': sources, 'output_hashes': artifact_hashes(output)})
    print(summary, flush=True)
    print(contrasts, flush=True)


if __name__ == '__main__':
    main()
