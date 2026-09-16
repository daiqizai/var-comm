"""Profile real receiver gradients without shadowing Python's standard profiling module."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src'), str(REFERENCE / 'scripts')]

import torch

from evaluate_interfaces import require_uncontended_gpu
from innovation_comm.common import load_parent, new_system, output_path, settings
from innovation_comm.runtime import actual_observation, receiver_loss
from wetok_comm.common import configure_torch, now, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: frozen E parameter checks and input gradients, identical waveform controls, no optimizer update')
        return
    require_uncontended_gpu()
    configure_torch()
    config, base, parent_milestone = settings()
    output = output_path(config, 'profile')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'scripts/train.py', EXPERIMENT / 'configs/study.yaml',
        EXPERIMENT / 'docs/protocol.md', *sorted((EXPERIMENT / 'src/innovation_comm').glob('*.py')),
        REFERENCE / 'src/wetok_comm/interface_study.py', REFERENCE / 'src/wetok_comm/training.py',
        REFERENCE / 'src/wetok_comm/native.py', REFERENCE / 'src/wetok_comm/model.py'])
    device = torch.device('cuda:0')
    parent = load_parent(config, base, device)
    decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
    frozen = {'encoder': module_sha256(parent.encoder), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
    if {'codec': frozen['codec'], 'lpips': frozen['lpips']} != parent_milestone['frozen']:
        raise RuntimeError('visual parent differs from the completed geometry study')
    batch = next(paired_batches(base, 20000, 7000, 7001))
    inputs = batch_inputs(read_population(base, 'train'), batch, device)
    part = {key: value[:1] for key, value in inputs.items()}
    signal, observed = actual_observation(parent, part)
    with torch.no_grad():
        parent_image = decoder.decode(parent.receive(observed, part['snrs'])['receiver_features'])
    rows, initial_images = [], {}
    for name in config['variants']:
        system = new_system(parent, name, config, device).train()
        initial_hash = module_sha256(system)
        if module_sha256(system.encoder) != frozen['encoder'] or system.encoder.training:
            raise RuntimeError('common encoder values or eval mode were not preserved')
        with torch.no_grad():
            if not torch.equal(signal, system.transmit(part['source_fq'], part['snrs'])):
                raise RuntimeError('receiver arms would receive different source waveforms')
        loss, components, unused, result = receiver_loss(system, observed, part, decoder, perceptual, config['training']['weights'])
        image_loss = components['mse'].mean() + .01 * components['lpips'].mean()
        parameters = tuple(value for value in system.parameters() if value.requires_grad)
        gradients = torch.autograd.grad(image_loss, parameters, retain_graph=True, allow_unused=True)
        norm = sum(float(value.double().square().sum()) for value in gradients if value is not None) ** .5
        if not 0 < norm < float('inf'):
            raise RuntimeError('final image gradients do not reach the receiver')
        image = result['image'].detach().clone()
        initial_images[name] = image.cpu()
        with torch.no_grad():
            system.eval()
            replay = decoder.decode(system.receive(observed, part['snrs'])['receiver_features'])
        if not torch.equal(image, replay):
            raise RuntimeError('receiver train/eval forward differs')
        if name in ('single_pass', 'multiscale_no_history') and not torch.equal(image, parent_image):
            raise RuntimeError('baseline initialization no longer matches the fixed parent image')
        if name in ('multiscale_prediction_features', 'multiscale_innovation') and not torch.equal(image.cpu(), initial_images['multiscale_state_history']):
            raise RuntimeError('zero extra projection changed the state-only initialization')
        del loss, components, unused, result, gradients, image_loss
        input_gradient_norms = []
        if system.has_feature:
            zero_projection = copy.deepcopy(system.fusion_projection.state_dict())
            with torch.no_grad():
                system.fusion_projection.weight.normal_(0, .01)
            system.train().zero_grad(set_to_none=True)
            loss, components, unused, result = receiver_loss(system, observed, part, decoder, perceptual, config['training']['weights'])
            for value in result['source_hypotheses']:
                if not value.requires_grad:
                    raise RuntimeError('frozen E was wrapped in no_grad and lost the hypothesis input graph')
                value.retain_grad()
            for value in result['predicted_symbols']:
                if not value.requires_grad:
                    raise RuntimeError('predicted waveform lost the frozen-Encoder input gradient path')
                value.retain_grad()
            (components['mse'].mean() + .01 * components['lpips'].mean()).backward()
            input_gradient_norms = [float(value.grad.norm()) if value.grad is not None else 0. for value in result['source_hypotheses']]
            if not all(value > 0 for value in input_gradient_norms):
                raise RuntimeError('image gradient does not cross the frozen source-hypothesis encoder')
            if not all(value.grad is not None and float(value.grad.norm()) > 0 for value in result['predicted_symbols']):
                raise RuntimeError('image objective does not use the re-encoded feature path')
            system.fusion_projection.load_state_dict(zero_projection)
            del loss, components, unused, result
        if any(value.grad is not None for value in system.encoder.parameters()):
            raise RuntimeError('frozen transmitter parameters received gradients')
        system.train().zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        loss, unused, unused, result = receiver_loss(system, observed, part, decoder, perceptual, config['training']['weights'])
        loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        if module_sha256(system) != initial_hash:
            raise RuntimeError('temporary input-gradient probe changed the training initialization')
        rows.append({'variant': name, 'trainable_parameters': sum(value.numel() for value in system.parameters() if value.requires_grad),
            'frozen_encoder_parameters': sum(value.numel() for value in system.encoder.parameters()),
            'image_gradient_norm': norm, 'hypothesis_gradient_norms_with_temporary_nonzero_projection': input_gradient_norms,
            'microbatch_RX_forward_backward_seconds': seconds, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
            'train_eval_pixels_equal': True, 'shared_waveform_equal': True})
        print(name, 'PROFILE_PASS', round(seconds, 4), flush=True)
        del system, loss, unused, result, parameters
    if frozen != {'encoder': module_sha256(parent.encoder), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
        raise RuntimeError('frozen shared models changed during receiver profile')
    if any(value.grad is not None for value in decoder.codec.parameters()) or any(value.grad is not None for value in perceptual.parameters()):
        raise RuntimeError('frozen image networks accumulated parameter gradients')
    verify_sources(sources)
    write_json(output / 'profile.json', {'status': 'INNOVATION_GRADIENT_PROFILE_PASS', 'completed_local': now(),
        'optimizer_updates': 0, 'frozen': frozen, 'source_hashes': sources, 'rows': rows,
        'batch_fingerprint': batch['fingerprint'],
        'estimated_1000_update_RX_training_hours_without_calibration': sum(row['microbatch_RX_forward_backward_seconds'] for row in rows) * 4000 / 3600,
        'scope': 'One fixed training example per arm; temporary fusion probe restored, no trained result or quality gate.'})


if __name__ == '__main__':
    main()
