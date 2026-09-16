"""Shared native-bit, coarse-state and final-image objectives."""

import torch
from torch.nn import functional


def losses(result, truth_fq, images, native_decoder, perceptual, weights):
    bit_target = truth_fq.add(1).mul(.5)
    bits = functional.binary_cross_entropy_with_logits(result['logits'], bit_target, reduction='none').flatten(1).mean(1)
    targets = [functional.adaptive_avg_pool2d(truth_fq, size) for size in (4, 8)]
    state_by_scale = torch.stack([(estimated - target).square().flatten(1).mean(1)
                                  for estimated, target in zip(result['states'], targets)], dim=1)
    state = state_by_scale.mean(1)
    components = {'bits': bits, 'state': state}
    if weights['mse'] or weights['lpips']:
        if not torch.all((result['native_fq'] == -1) | (result['native_fq'] == 1)):
            raise RuntimeError('the registered hard receiver produced off-native values')
        reconstruction = native_decoder.decode(result['native_fq'])
        result['image'] = reconstruction
        components['mse'] = (reconstruction - images).square().flatten(1).mean(1)
        components['lpips'] = perceptual(reconstruction * 2 - 1, images * 2 - 1).reshape(-1)
    else:
        components['mse'] = bits.new_zeros(bits.shape)
        components['lpips'] = bits.new_zeros(bits.shape)
    objective = sum(weights[name] * values for name, values in components.items()).mean()
    bit_errors = result['native_fq'].detach().ne(truth_fq).float()
    diagnostics = {'bit_error_rate': bit_errors.flatten(1).mean(1), 'state_by_scale': state_by_scale,
                   'group_error_rate': bit_errors.permute(0, 2, 3, 1).reshape(len(bits), 16, 16, 4, 8).amax(-1).flatten(1).mean(1)}
    return objective, components, diagnostics
