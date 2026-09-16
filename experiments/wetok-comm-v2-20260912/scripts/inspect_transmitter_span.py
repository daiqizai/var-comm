"""CPU training-source diagnostics, not a bound on the complete nonlinear channel code."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np
import torch

from wetok_comm.common import PROJECT, configure_torch, now, output_path, sha256, verify_sources, write_json
from wetok_comm.interface_study import interface_definitions, interface_output, load_interface_study, make_interface_network
from wetok_comm.native import indices_to_features
from wetok_comm.training import paired_batches


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: 16 fixed training sources, CPU transmitter forward, no visual Decoder or updates')
        return
    configure_torch()
    torch.set_num_threads(2)
    study, base, parent = load_interface_study()
    training = interface_output(study, 'training')
    verify_sources(json.loads((training / 'metadata.json').read_text())['source_hashes'])
    with (training / 'resume.pt').open('rb') as handle:
        current = torch.load(handle, map_location='cpu', weights_only=True)
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if current['completed_additional_updates'] != 1000:
        raise RuntimeError('this registered diagnostic requires the actual 1000-update interface boundary')
    cache = output_path(base, 'cache')
    cache_receipt = json.loads((cache / 'completion.json').read_text())
    if sha256(cache / 'train_codes.npy') != cache_receipt['output_hashes']['train_codes.npy']:
        raise RuntimeError('native training source cache changed')
    codes = np.load(cache / 'train_codes.npy', mmap_mode='r', allow_pickle=False)
    batches = list(paired_batches(base, 20000, 2000, 2004))
    grouped = np.concatenate([np.array(codes[batch['indices'].numpy(), batch['flip'].long().numpy()], copy=True) for batch in batches])
    sources = indices_to_features(torch.from_numpy(grouped))
    states = []
    for variant in study['variants']:
        model = torch.load(parent / variant / 'checkpoints/step_0002000.pt', map_location='cpu', weights_only=True)['model']
        states.append((f'parent2000__{variant}', {'variant': variant, 'interface': 'hard_identity'}, model))
    states.extend((name, definition, current['models'][name]) for name, definition in interface_definitions(study).items())
    records = []
    for name, definition, state in states:
        network = make_interface_network(base, definition, state, 'cpu').eval().requires_grad_(False)
        lift = network.encoder.channel_lift.weight
        channel_rank = int(torch.linalg.matrix_rank(lift))
        spatial_rank = int(torch.linalg.matrix_rank(network.encoder.spatial_mixing))
        basis, singular, unused = torch.linalg.svd(lift, full_matrices=True)
        complement = basis[:, channel_rank:]
        symbols = torch.cat([network.transmit(sources[start:start + 4], torch.full((4,), 19.))
                             for start in range(0, len(sources), 4)])
        values = symbols.reshape(len(sources), base['model']['channel_positions'], base['model']['channel_features'])
        energy = values.square().sum((1, 2))
        complement_energy = (values @ complement).square().sum((1, 2)) / energy
        flattened = values.flatten(0, 1).double()
        centered = flattened - flattened.mean(0, keepdim=True)
        covariance = centered.t() @ centered / len(flattened)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        power_error = float((symbols.square().sum(-1).mean(-1) - 2).abs().max())
        if power_error > 1e-5:
            raise RuntimeError('diagnostic changed the actual normalized power')
        records.append({'arm': name, 'linear_skip_spatial_rank': spatial_rank, 'linear_skip_channel_rank': channel_rank,
            'linear_skip_operator_rank': spatial_rank * channel_rank, 'paid_real_coordinates': 6120,
            'channel_lift_complement_dimensions': complement.shape[1],
            'actual_signal_energy_fraction_outside_lift_mean': float(complement_energy.mean()),
            'actual_signal_energy_fraction_outside_lift_max': float(complement_energy.max()),
            'pooled_40_channel_covariance_participation_ratio': float(eigenvalues.sum().square() / eigenvalues.square().sum()),
            'smallest_8_channel_covariance_eigenvalue_fraction': float(eigenvalues[:8].sum() / eigenvalues.sum()),
            'power_max_error': power_error})
        print(name, records[-1], flush=True)
        del network
    output = PROJECT / 'outputs/WETOK-TX-SPAN-DIAGNOSTIC-20260912'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', {'status': 'TRAINING_SOURCE_CPU_TX_DIAGNOSTIC', 'completed_local': now(),
        'source_images': len(sources), 'source_role': 'training_only', 'nominal_snr_db': 19,
        'optimizer_updates': 0, 'GPU_used': False, 'visual_decoder_loaded': False,
        'current_checkpoint_sha256': digest.hexdigest(), 'source_batch_fingerprints': [batch['fingerprint'] for batch in batches],
        'records': records, 'limitations': 'Linear skip rank is not full nonlinear code rank or a capacity/quality bound. Small-sample covariance is not transmitted information or semantic scale evidence.'})


if __name__ == '__main__':
    main()
