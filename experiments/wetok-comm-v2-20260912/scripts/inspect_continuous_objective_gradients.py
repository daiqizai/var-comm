"""Small CPU training-only gradient probe, not an efficacy gate or optimizer update."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import torch

from wetok_comm.common import PROJECT, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.interface_study import interface_losses, interface_output, load_interface_study, make_interface_network
from wetok_comm.model import communicate
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, paired_batches, read_population


def cosine(first, second):
    denominator = float(first.norm() * second.norm())
    return float(torch.dot(first, second)) / denominator if denominator > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: four fixed training examples x three continuous models, CPU gradients, zero parameter updates')
        return
    configure_torch()
    torch.set_num_threads(2)
    study, base, parent = load_interface_study()
    training = interface_output(study, 'training')
    verify_sources(json.loads((training / 'metadata.json').read_text())['source_hashes'])
    with (training / 'resume.pt').open('rb') as handle:
        current = torch.load(handle, map_location='cpu', weights_only=True)
        handle.seek(0)
        checkpoint_digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checkpoint_digest.update(chunk)
    completed = current['completed_additional_updates']
    if completed < 1500:
        raise RuntimeError('the observed deterioration checkpoint is not yet saved')
    output = PROJECT / 'outputs' / f'WETOK-CONTINUOUS-GRADIENT-PROBE-20260912-{completed:07d}'
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/interface_study.yaml',
                                     EXPERIMENT / 'src/wetok_comm/interface_study.py'])
    batch = next(paired_batches(base, 20000, 5000, 5001))
    population = read_population(base, 'train')
    inputs = batch_inputs(population, batch, torch.device('cpu'))
    del population
    decoder, perceptual = FrozenWeTok('cpu', 'decoder'), load_lpips('cpu')
    frozen = {'native': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
    rows = []
    for variant in study['variants']:
        name = 'continuous_mean__' + variant
        network = make_interface_network(base, {'variant': variant, 'interface': 'continuous_mean'}, current['models'][name], 'cpu').train()
        original_hash = module_sha256(network)
        named_parameters = tuple(network.named_parameters())
        parameters = tuple(value for key, value in named_parameters)
        encoder_length = sum(value.numel() for key, value in named_parameters if key.startswith('encoder.'))
        if not all(key.startswith('encoder.') for key, value in named_parameters[:len(tuple(network.encoder.parameters()))]):
            raise RuntimeError('parameter partition no longer matches the encoder/receiver audit')
        for index in range(4):
            part = {key: value[index:index + 1] for key, value in inputs.items()}
            result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
            objective, components, unused = interface_losses(result, part['source_fq'], part['images'], decoder, perceptual, study['weights'])
            gradients = {}
            for component, values in components.items():
                computed = torch.autograd.grad(values.mean(), parameters, retain_graph=True, allow_unused=True)
                gradients[component] = torch.cat([(gradient.detach().reshape(-1).double() if gradient is not None
                    else torch.zeros(parameter.numel(), dtype=torch.float64)) for parameter, gradient in zip(parameters, computed)])
            for scope, selected in (('all', slice(None)), ('encoder', slice(0, encoder_length)), ('receiver', slice(encoder_length, None))):
                selected_gradients = {key: value[selected] for key, value in gradients.items()}
                weighted = {key: study['weights'][key] * value for key, value in selected_gradients.items()}
                total = sum(weighted.values())
                rows.append({'arm': name, 'training_index': int(batch['indices'][index]), 'snr_db': float(part['snrs'][0]),
                    'parameter_scope': scope, 'mse_lpips_gradient_cosine': cosine(selected_gradients['mse'], selected_gradients['lpips']),
                    'total_gradient_lpips_cosine': cosine(total, selected_gradients['lpips']),
                    'image_vs_auxiliary_cosine': cosine(weighted['mse'] + weighted['lpips'], weighted['bits'] + weighted['state']),
                    'weighted_gradient_norms': {key: float(value.norm()) for key, value in weighted.items()},
                    'loss_values': {key: float(value.detach().mean()) for key, value in components.items()}})
            print(name, index, {key: rows[-3][key] for key in ('mse_lpips_gradient_cosine', 'total_gradient_lpips_cosine', 'weighted_gradient_norms')}, flush=True)
            del result, objective, components, unused, gradients, computed, selected_gradients, weighted, total
        if module_sha256(network) != original_hash or any(parameter.grad is not None for parameter in network.parameters()):
            raise RuntimeError('gradient-only probe changed communication parameters or accumulated updates')
        del network
    if frozen != {'native': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
        raise RuntimeError('frozen visual model changed')
    verify_sources(source_hashes)
    write_json(output / 'summary.json', {'status': 'CONTINUOUS_OBJECTIVE_TRAINING_GRADIENT_DIAGNOSTIC', 'completed_local': now(),
        'checkpoint_additional_updates': completed, 'checkpoint_sha256_at_read': checkpoint_digest.hexdigest(),
        'training_images': 4, 'source_batch_fingerprint': batch['fingerprint'], 'source_hashes': source_hashes,
        'optimizer_updates': 0, 'GPU_used': False, 'rows': rows,
        'limitations': 'Four training samples only. Gradient cosines are not the Adam update direction, do not estimate independent image quality, and do not justify changing the live protocol.'})


if __name__ == '__main__':
    main()
