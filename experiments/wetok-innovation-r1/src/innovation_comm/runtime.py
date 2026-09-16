"""Shared-observation loss and calibration utilities for the frozen-TX study."""

import numpy as np
import torch

from wetok_comm.interface_study import interface_losses
from wetok_comm.native import indices_to_features
from wetok_comm.training import seeded_noise


def actual_observation(parent, inputs):
    with torch.no_grad():
        signal = parent.transmit(inputs['source_fq'], inputs['snrs'])
        observed = signal + inputs['noise'] * torch.pow(10., inputs['snrs'] / 10).rsqrt()[:, None, None]
    return signal, observed


def receiver_loss(system, observed, inputs, decoder, perceptual, weights):
    result = system.receive(observed, inputs['snrs'])
    objective, components, diagnostics = interface_losses(result, inputs['source_fq'], inputs['images'], decoder, perceptual, weights)
    return objective, components, diagnostics, result


@torch.no_grad()
def calibrate(system, parent, population, decoder, perceptual, config, base, device, positions=None):
    from pytorch_msssim import ssim

    system.eval()
    images, codes, identifiers = population
    positions = range(len(images)) if positions is None else positions
    rows = []
    for snr in base['channel']['snrs_db']:
        for index in positions:
            grouped = torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)).to(device)
            truth = indices_to_features(grouped)
            target = images[index:index + 1].to(device).float()
            if images.dtype == torch.uint8:
                target = target / 255
            inputs = {'source_fq': truth, 'images': target, 'snrs': torch.tensor([snr], device=device),
                      'noise': torch.tensor(seeded_noise(identifiers[index], base['calibration']['noise_seed'], (1, 3060, 2)), device=device)}
            signal, observed = actual_observation(parent, inputs)
            unused, components, diagnostics, result = receiver_loss(system, observed, inputs, decoder, perceptual, config['training']['weights'])
            mse = float(components['mse'][0])
            row = {'image_id': identifiers[index], 'snr_db': float(snr), 'psnr_db': float(-10 * np.log10(max(mse, 1e-12))),
                   'lpips': float(components['lpips'][0]), 'ssim': float(ssim(result['image'], target, data_range=1., size_average=True)),
                   'bits_BCE': float(components['bits'][0]), 'state_error': float(components['state'][0]),
                   'bit_error_rate': float(diagnostics['bit_error_rate'][0]), 'feature_mse': float(diagnostics['feature_mse'][0]),
                   'state_error_4': float(diagnostics['state_by_scale'][0, 0]), 'state_error_8': float(diagnostics['state_by_scale'][0, 1]),
                   'feature_gate_1': '', 'feature_gate_2': '', 'residual_noise_ratio_1': '', 'residual_noise_ratio_2': ''}
            for stage, (gate, ratio) in enumerate(zip(result['feature_gates'], result['residual_noise_ratios']), 1):
                row[f'feature_gate_{stage}'] = float(gate[0, 0])
                row[f'residual_noise_ratio_{stage}'] = float(ratio[0])
            if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
                raise RuntimeError('nonfinite receiver calibration outcome; no sample may be dropped')
            rows.append(row)
    return rows
