"""Image-only inference removes dead no-history auxiliary reads, never training supervision."""

import torch
from wetok_comm.native import hard_native_st


def receive_for_image(system, received, snrs):
    if torch.is_grad_enabled() or system.training:
        raise RuntimeError('image-only pruning is an evaluation path, not a training/calibration loss path')
    if system.variant != 'multiscale_no_history':
        return system.receive(received, snrs)
    if system.interface != 'continuous_mean':
        raise ValueError('only the registered continuous receiver interface can be pruned here')
    guessed, memory = system.receiver.prepare(received, snrs)
    logits = system.receiver.read(guessed, memory, snrs, 16)
    return {'logits': logits, 'native_fq': hard_native_st(logits), 'receiver_features': torch.tanh(logits / 2),
            'states': [], 'predicted_symbols': [], 'source_hypotheses': [], 'feature_gates': [], 'residual_noise_ratios': []}


def verify_pruned_result(system, result, received, snrs):
    if system.variant != 'multiscale_no_history':
        return 0.
    if torch.is_grad_enabled() or system.training:
        raise RuntimeError('pruning verification must not enter a training graph')
    reference = system.receive(received, snrs)
    if any(not torch.equal(result[key], reference[key]) for key in ('logits', 'native_fq', 'receiver_features')):
        raise RuntimeError('removing unused no-history stages changed the actual image input')
    return float((result['receiver_features'] - reference['receiver_features']).abs().max())
