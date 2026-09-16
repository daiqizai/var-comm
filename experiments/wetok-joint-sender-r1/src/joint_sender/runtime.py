"""Paired batch4 transmissions and microbatch image gradients through the trainable sender."""

import hashlib

import torch

from wetok_comm.interface_study import interface_losses


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def observation(system, inputs):
    signal = system.transmit(inputs['source_fq'], inputs['snrs'])
    if signal.shape != inputs['noise'].shape:
        raise ValueError('source transmission and paid AWGN shape differ')
    received = signal + inputs['noise'] * torch.pow(10., inputs['snrs'] / 10).rsqrt()[:, None, None]
    return signal, received


def gradient_norm(module):
    values = [value.grad.detach().square().sum() for value in module.parameters() if value.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def backward_paired_batch(system, inputs, decoder, perceptual, weights, loss_function=interface_losses):
    if len(inputs['images']) != 4:
        raise ValueError('the paired reference uses effective batch4 and microbatch1')
    system.train().zero_grad(set_to_none=True)
    signal, received = observation(system, inputs)
    receiver_input = received.detach().requires_grad_(received.requires_grad)
    power_error = float((signal.detach().square().sum(-1).mean(-1) - 2).abs().max())
    if power_error > 1e-5:
        raise RuntimeError('joint sender violated the fixed complex energy ledger')
    totals = {key: 0. for key in ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate')}
    for index in range(4):
        part = {key: value[index:index + 1] for key, value in inputs.items()}
        result = system.receive(receiver_input[index:index + 1], part['snrs'])
        objective, components, diagnostics = loss_function(result, part['source_fq'], part['images'], decoder, perceptual, weights)
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError('nonfinite communication image objective')
        (objective / 4).backward()
        totals['loss'] += float(objective.detach()) / 4
        for key, value in components.items():
            totals[key] += float(value.detach().sum()) / 4
        totals['bit_error_rate'] += float(diagnostics['bit_error_rate'].detach().sum()) / 4
        del part, objective, components, diagnostics, result
    waveform_norm = 0.
    if receiver_input.requires_grad:
        if receiver_input.grad is None or not bool(torch.isfinite(receiver_input.grad).all()):
            raise RuntimeError('image/receiver gradients did not reach the shared channel boundary')
        waveform_norm = float(receiver_input.grad.norm())
        received.backward(receiver_input.grad)
    encoder_norm, receiver_norm = gradient_norm(system.encoder), gradient_norm(system.receiver)
    if not system.update_sender and any(value.grad is not None for value in system.encoder.parameters()):
        raise RuntimeError('frozen sender control accumulated parameter gradients')
    return {**totals, 'encoder_gradient_norm': encoder_norm, 'receiver_gradient_norm': receiver_norm,
            'waveform_gradient_norm': waveform_norm,
            'power_max_error': power_error, 'transmitted_sha256': tensor_sha256(signal),
            'received_sha256': tensor_sha256(received), 'paired_standard_noise_sha256': tensor_sha256(inputs['noise'])}


@torch.no_grad()
def calibration(system, population, decoder, perceptual, reference_config, base, device, positions=None):
    from innovation_comm.runtime import calibrate

    return calibrate(system, system, population, decoder, perceptual, reference_config, base, device, positions)
