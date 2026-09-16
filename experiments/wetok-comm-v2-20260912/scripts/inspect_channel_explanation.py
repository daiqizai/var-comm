"""Training-only no-update probe of receiver hypotheses against the paid waveform."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np
import torch

from evaluate_interfaces import require_uncontended_gpu
from train_milestone import write_csv
from wetok_comm.common import PROJECT, configure_torch, now, output_path, sha256, snapshot, verify_sources, write_json
from wetok_comm.geometry_evaluation import load_geometry_evaluation, model_registry
from wetok_comm.geometry_study import geometry_output
from wetok_comm.native import indices_to_features
from wetok_comm.training import module_sha256, paired_batches


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: 16 training sources, frozen communication E/R only, no image Decoder, no updates')
        return
    require_uncontended_gpu()
    configure_torch()
    evaluation, config, base, qualification = load_geometry_evaluation()
    training = geometry_output(config, 'training')
    milestone_path = training / 'milestones/total_0007000.json'
    milestone = json.loads(milestone_path.read_text())
    verify_sources(milestone['source_hashes'])
    output = PROJECT / 'outputs/WETOK-CHANNEL-EXPLANATION-PROBE-20260913'
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__), EXPERIMENT / 'docs/channel_explanation_probe.md',
        EXPERIMENT / 'configs/geometry_study.yaml', *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])
    cache = output_path(base, 'cache')
    receipt = json.loads((cache / 'completion.json').read_text())
    if sha256(cache / 'train_codes.npy') != receipt['output_hashes']['train_codes.npy']:
        raise RuntimeError('training source cache changed')
    codes = np.load(cache / 'train_codes.npy', mmap_mode='r', allow_pickle=False)
    batches = list(paired_batches(base, 20000, 7000, 7004))
    grouped = np.concatenate([np.array(codes[batch['indices'].numpy(), batch['flip'].long().numpy()], copy=True) for batch in batches])
    source = indices_to_features(torch.from_numpy(grouped)).to('cuda:0')
    noise = torch.cat([batch['noise'] for batch in batches]).to('cuda:0')
    source_indices = torch.cat([batch['indices'] for batch in batches]).tolist()
    networks, choices = model_registry(config, base, qualification, milestone, training, 'cuda:0')
    networks = {name: model for name, model in networks.items() if name.startswith('geometry204x30__')}
    hashes = {name: module_sha256(model) for name, model in networks.items()}
    rows = []
    for name, network in networks.items():
        for snr, noiseless in ((1., False), (7., False), (19., False), (19., True)):
            condition = 'noiseless_nominal19' if noiseless else f'AWGN_{snr:g}dB'
            for index in range(len(source)):
                truth = source[index:index + 1]
                snrs = torch.tensor([snr], device='cuda:0')
                captured = {}
                def capture_base(module, inputs, values):
                    captured['base'] = values.detach()
                def capture_residual(module, inputs, values):
                    captured['residual'] = values.detach()
                hooks = [network.encoder.channel_lift.register_forward_hook(capture_base),
                         network.encoder.output.register_forward_hook(capture_residual)]
                signal = network.transmit(truth, snrs)
                for hook in hooks:
                    hook.remove()
                base_signal = torch.einsum('mn,bnc->bmc', network.encoder.spatial_mixing, captured['base'])
                residual_signal = torch.einsum('mn,bnc->bmc', network.encoder.spatial_mixing, captured['residual'])
                residual_fraction = float(residual_signal.square().sum() / (base_signal + residual_signal).square().sum().clamp_min(1e-20))
                received = signal if noiseless else signal + noise[index:index + 1] * torch.pow(10., snrs / 10).rsqrt()[:, None, None]
                result = network.receive(received, snrs)
                soft = result['receiver_features']
                hard = result['native_fq']
                soft_signal = network.transmit(soft, snrs)
                hard_signal = network.transmit(hard, snrs)
                replay = network.transmit(truth, snrs)
                replay_error = float((replay - signal).abs().max())
                if replay_error > 1e-6:
                    raise RuntimeError('shared Encoder is not a deterministic replay of the actual transmission')
                power_error = float((signal.square().sum(-1).mean(-1) - 2).abs().max())
                if power_error > 1e-5:
                    raise RuntimeError('source transmission changed the paid energy')
                observed_noise = float((received - signal).square().mean())
                soft_residual = float((received - soft_signal).square().mean())
                hard_residual = float((received - hard_signal).square().mean())
                rows.append({'arm': name, 'condition': condition, 'training_index': source_indices[index],
                    'source_replay_max_error': replay_error, 'power_max_error': power_error,
                    'true_noise_mean_square': observed_noise,
                    'soft_channel_residual_mean_square': soft_residual,
                    'hard_channel_residual_mean_square': hard_residual,
                    'soft_signal_prediction_mean_square_error': float((signal - soft_signal).square().mean()),
                    'hard_signal_prediction_mean_square_error': float((signal - hard_signal).square().mean()),
                    'soft_residual_over_noise': soft_residual / observed_noise if observed_noise > 0 else '',
                    'hard_residual_over_noise': hard_residual / observed_noise if observed_noise > 0 else '',
                    'soft_source_feature_mean_square_error': float((soft - truth).square().mean()),
                    'hard_source_bit_error_rate': float(hard.ne(truth).float().mean()),
                    'TX_neural_residual_energy_over_actual_raw_signal_energy': residual_fraction})
            print(name, condition, 'complete', flush=True)
    if hashes != {name: module_sha256(model) for name, model in networks.items()}:
        raise RuntimeError('no-update channel probe changed communication weights')
    write_csv(output / 'per_source.csv', rows)
    summaries = []
    for name in networks:
        for condition in sorted({row['condition'] for row in rows}):
            subset = [row for row in rows if row['arm'] == name and row['condition'] == condition]
            summary = {'arm': name, 'condition': condition, 'training_sources': len(subset)}
            for field in subset[0]:
                if field in ('arm', 'condition', 'training_index'):
                    continue
                values = [row[field] for row in subset if row[field] != '']
                summary[field] = float(np.mean(values)) if values else ''
            summaries.append(summary)
    write_csv(output / 'summary.csv', summaries)
    verify_sources(source_hashes)
    write_json(output / 'completion.json', {'status': 'CHANNEL_EXPLANATION_DIAGNOSTIC_COMPLETE', 'completed_local': now(),
        'rows': len(rows), 'training_sources': 16, 'source_hashes': source_hashes,
        'milestone_sha256': sha256(milestone_path), 'selected_models': choices,
        'frozen_communication_hashes': hashes, 'optimizer_updates': 0, 'visual_decoder_used': False,
        'source_batch_fingerprints': [batch['fingerprint'] for batch in batches],
        'scope': 'Frozen training-source E/R probe only. Soft re-encoding is a continuous extension, not a native-code likelihood or deployed recovery.'})


if __name__ == '__main__':
    main()
