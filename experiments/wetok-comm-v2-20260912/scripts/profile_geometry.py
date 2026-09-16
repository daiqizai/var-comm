"""No-update real frozen-Decoder checks before the new geometry training."""

import argparse
import json
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import torch

from evaluate_interfaces import require_uncontended_gpu
from wetok_comm.common import PROJECT, configure_torch, now, snapshot, verify_sources, write_json
from wetok_comm.geometry_study import geometry_network, geometry_output, load_geometry_study
from wetok_comm.interface_study import interface_losses
from wetok_comm.model import communicate
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: real new-geometry image gradients and exact budget, zero optimizer updates')
        return
    require_uncontended_gpu()
    configure_torch()
    config, base, qualification = load_geometry_study()
    output = geometry_output(config, 'profile')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'scripts/train_geometry.py',
        EXPERIMENT / 'scripts/evaluate_interfaces.py', EXPERIMENT / 'scripts/train_milestone.py',
        EXPERIMENT / 'configs/geometry_study.yaml', EXPERIMENT / 'docs/geometry_training_protocol.md',
        *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])
    device = torch.device('cuda:0')
    decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
    frozen = {'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
    control = json.loads((PROJECT / 'outputs' / config['control_training'] / config['control_milestone']).read_text())
    if frozen != control['frozen']:
        raise RuntimeError('geometry would use a different visual model from its controls')
    batch = next(paired_batches(base, 20000, 0, 1))
    inputs = batch_inputs(read_population(base, 'train'), batch, device)
    part = {key: value[:1] for key, value in inputs.items()}
    records = []
    for variant in config['variants']:
        network, match = geometry_network(config, base, qualification, variant, device)
        before = module_sha256(network)
        network.train()
        result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
        objective, components, unused = interface_losses(result, part['source_fq'], part['images'], decoder, perceptual, config['training']['image_weights'])
        parameters = tuple(network.parameters())
        image_objective = components['mse'].mean() + .01 * components['lpips'].mean()
        gradients = torch.autograd.grad(image_objective, parameters, retain_graph=True, allow_unused=True)
        encoder_ids = {id(parameter) for parameter in network.encoder.parameters()}
        norms = {'encoder': 0., 'receiver': 0.}
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                key = 'encoder' if id(parameter) in encoder_ids else 'receiver'
                norms[key] += float(gradient.double().square().sum())
        if not all(0 < value < float('inf') for value in norms.values()):
            raise RuntimeError('real image gradient did not reach both geometry E/D halves')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
        train_image = result['image'].detach().clone()
        with torch.no_grad():
            network.eval()
            replay = communicate(network, part['source_fq'], part['snrs'], part['noise'])
            evaluation_image = decoder.decode(replay['receiver_features'])
        if not torch.equal(train_image, evaluation_image):
            raise RuntimeError('geometry training/evaluation interface changed')
        network.train().zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        fresh = communicate(network, part['source_fq'], part['snrs'], part['noise'])
        loss, unused, unused = interface_losses(fresh, part['source_fq'], part['images'], decoder, perceptual, config['training']['image_weights'])
        loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        power_error = float((fresh['transmitted'].detach().square().sum(-1).mean(-1) - 2).abs().max())
        if power_error > 1e-5 or module_sha256(network) != before:
            raise RuntimeError('no-update profile changed parameters or paid energy')
        records.append({'variant': variant, 'candidate_parameters': match['candidate_parameters'],
            'control_parameters': match['control_parameters'], 'image_gradient_norms': {key: value ** .5 for key, value in norms.items()},
            'one_microbatch_image_forward_backward_seconds': seconds, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
            'train_eval_pixels_equal': True, 'power_max_error': power_error})
        print(variant, 'GEOMETRY_PROFILE_PASS', round(seconds, 4), flush=True)
        del network, result, objective, components, fresh, loss, replay, gradients, parameters, unused
    if frozen != {'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
        raise RuntimeError('geometry profile changed frozen visual parameters')
    if any(parameter.grad is not None for parameter in decoder.codec.parameters()) or any(parameter.grad is not None for parameter in perceptual.parameters()):
        raise RuntimeError('frozen visual parameters accumulated gradients')
    verify_sources(sources)
    write_json(output / 'profile.json', {'status': 'GEOMETRY_GRADIENT_PROFILE_PASS', 'completed_local': now(),
        'optimizer_updates': 0, 'source_hashes': sources, 'frozen': frozen, 'records': records,
        'source_batch_fingerprint': batch['fingerprint'], 'CPU_threads': torch.get_num_threads(),
        'estimated_1000_image_update_hours_without_calibration': sum(row['one_microbatch_image_forward_backward_seconds'] for row in records) * 4000 / 3600,
        'control_training_reused_not_retrained': True})


if __name__ == '__main__':
    main()
