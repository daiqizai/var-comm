"""Real frozen-Decoder, no-update engineering checks for the interface study."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import torch

from wetok_comm.common import configure_torch, now, snapshot, write_json
from wetok_comm.interface_study import interface_definitions, interface_losses, interface_output, load_interface_study, make_interface_network
from wetok_comm.model import communicate
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    study, base, parent = load_interface_study()
    if not arguments.execute:
        print('PLAN ONLY: nine interface arms, real frozen decoder gradients, zero optimizer updates')
        return
    active = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU compute process exists; do not interfere with its workload')
    configure_torch()
    output = interface_output(study, 'profile')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'scripts/train_interfaces.py',
        EXPERIMENT / 'scripts/train_milestone.py', EXPERIMENT / 'configs/interface_study.yaml',
        EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/interface_protocol.md',
        *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])
    device = torch.device('cuda:0')
    decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
    frozen = {'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
    batch = next(paired_batches(base, 20000, study['parent_step'], study['parent_step'] + 1))
    inputs = batch_inputs(read_population(base, 'train'), batch, device)
    part = {key: value[:1] for key, value in inputs.items()}
    records, hard_images, parent_hashes = [], {}, {}
    for name, definition in interface_definitions(study).items():
        stored = torch.load(parent / definition['variant'] / 'checkpoints/step_0002000.pt', map_location='cpu', weights_only=True)
        network = make_interface_network(base, definition, stored['model'], device)
        initial = module_sha256(network)
        if definition['variant'] in parent_hashes and parent_hashes[definition['variant']] != initial:
            raise RuntimeError('interface arms do not share the same parent parameters')
        parent_hashes[definition['variant']] = initial
        network.train()
        result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
        objective, components, diagnostics = interface_losses(result, part['source_fq'], part['images'], decoder, perceptual, study['weights'])
        parameters = tuple(network.parameters())
        encoder_names = {id(value) for value in network.encoder.parameters()}
        norms = {}
        for component, values in components.items():
            gradients = torch.autograd.grad(values.mean(), parameters, retain_graph=True, allow_unused=True)
            squares = {'encoder': 0., 'receiver': 0.}
            for parameter, gradient in zip(parameters, gradients):
                if gradient is not None:
                    key = 'encoder' if id(parameter) in encoder_names else 'receiver'
                    squares[key] += float(gradient.double().square().sum())
            norms[component] = {key: value ** .5 for key, value in squares.items()}
            if component in ('mse', 'lpips') and not all(0 < value < float('inf') for value in norms[component].values()):
                raise RuntimeError('image gradients do not reach both communication halves')
        training_image = result['image'].detach().clone()
        if definition['interface'] == 'hard_identity':
            hard_images[definition['variant']] = training_image.cpu()
        elif definition['interface'] == 'hard_bounded':
            if not torch.equal(hard_images[definition['variant']], training_image.cpu()):
                raise RuntimeError('changing only ST changed the hard image forward')
        network.zero_grad(set_to_none=True)
        objective.backward()
        if any(value.grad is not None for value in decoder.codec.parameters()) or any(value.grad is not None for value in perceptual.parameters()):
            raise RuntimeError('visual weights received gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True))
        with torch.no_grad():
            network.eval()
            replay = communicate(network, part['source_fq'], part['snrs'], part['noise'])
            evaluation_image = decoder.decode(replay['receiver_features'])
        if not torch.equal(training_image, evaluation_image):
            raise RuntimeError('training/evaluation interface forwards differ')
        del result, objective, components, diagnostics, replay, gradients
        network.train().zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
        objective, unused, unused = interface_losses(result, part['source_fq'], part['images'], decoder, perceptual, study['weights'])
        objective.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        power_error = float((result['transmitted'].square().sum(-1).mean(-1) - 2).abs().max())
        if power_error > 1e-5 or module_sha256(network) != initial:
            raise RuntimeError('no-update profile changed parameters or physical power')
        records.append({'arm': name, **definition, 'trainable_parameters': sum(value.numel() for value in parameters),
                        'component_gradient_norms': norms, 'joint_gradient_norm_before_clip': norm,
                        'training_evaluation_pixels_equal': True, 'power_max_error': power_error,
                        'one_microbatch_forward_backward_seconds': seconds,
                        'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'feature_min': float(result['receiver_features'].detach().min()),
                        'feature_max': float(result['receiver_features'].detach().max())})
        print(name, 'PASS', 'microbatch_seconds', round(seconds, 4), flush=True)
        del network, result, objective, parameters, unused
    if frozen != {'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
        raise RuntimeError('frozen visual weights changed in profile')
    seconds_per_paired_update = 4 * sum(row['one_microbatch_forward_backward_seconds'] for row in records)
    write_json(output / 'profile.json', {'status': 'INTERFACE_GRADIENT_PROFILE_PASS', 'completed_local': now(),
        'optimizer_updates': 0, 'records': records, 'batch_fingerprint': batch['fingerprint'],
        'source_hashes': sources, 'frozen': frozen, 'initial_hashes_by_variant': parent_hashes,
        'estimated_1000_update_training_GPU_hours_excluding_calibration': seconds_per_paired_update * 1000 / 3600,
        'estimated_5000_update_training_GPU_hours_excluding_calibration': seconds_per_paired_update * 5000 / 3600,
        'estimate_limitation': 'single warmed microbatch per arm, excludes optimizer, I/O and calibration; not a completion promise'})
    print('INTERFACE_GRADIENT_PROFILE_PASS', flush=True)


if __name__ == '__main__':
    main()
