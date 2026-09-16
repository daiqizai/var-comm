"""Matched hard-gradient and continuous-feature receiver interfaces."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import EXPERIMENT, PROJECT, settings, sha256, verify_sources
from .model import NativeWeTokJSCC, communicate
from .native import hard_native_st, indices_to_features
from .objective import losses
from .training import seeded_noise


INTERFACES = ('hard_identity', 'hard_bounded', 'continuous_mean')


def receiver_features(logits, interface):
    if interface == 'hard_identity':
        return hard_native_st(logits)
    mean = torch.tanh(logits / 2)
    if interface == 'continuous_mean':
        return mean
    if interface == 'hard_bounded':
        hard = torch.where(logits > 0, torch.ones_like(logits), -torch.ones_like(logits))
        return hard + (mean - mean.detach()) if torch.is_grad_enabled() and logits.requires_grad else hard
    raise ValueError('unregistered receiver interface')


class InterfaceJSCC(NativeWeTokJSCC):
    def __init__(self, variant, config, interface):
        super().__init__(variant, config)
        if interface not in INTERFACES:
            raise ValueError('unregistered receiver interface')
        self.interface = interface

    def receive(self, received, snrs):
        result = super().receive(received, snrs)
        result['receiver_features'] = receiver_features(result['logits'], self.interface)
        return result


def load_interface_study(path=None):
    path = Path(path or EXPERIMENT / 'configs/interface_study.yaml')
    study = yaml.safe_load(path.read_text())
    base = settings(EXPERIMENT / study['base_config'])
    if tuple(study['interfaces']) != INTERFACES or study['variants'] != base['arms']:
        raise ValueError('registered interface/structure matrix changed')
    if study['weights'] != base['training']['joint_weights'] or study['parent_step'] != 2000:
        raise ValueError('this interface comparison does not change the joint loss or parent')
    parent = PROJECT / 'outputs' / study['parent_training']
    receipt_path = parent / 'milestones/step_0005000.json'
    if sha256(receipt_path) != study['parent_milestone_sha256']:
        raise RuntimeError('original training history receipt changed')
    verify_sources(json.loads(receipt_path.read_text())['source_hashes'])
    for variant, expected in study['parent_checkpoints'].items():
        if sha256(parent / variant / 'checkpoints/step_0002000.pt') != expected:
            raise RuntimeError('frozen representation parent changed')
    return study, base, parent


def interface_definitions(study):
    return {f'{interface}__{variant}': {'interface': interface, 'variant': variant}
            for variant in study['variants'] for interface in study['interfaces']}


def interface_output(study, kind):
    path = (PROJECT / 'outputs' / study['outputs'][kind]).resolve()
    if not path.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('interface output escapes project')
    return path


def make_interface_network(base, definition, state, device):
    torch.manual_seed(base['training']['initialization_seed'])
    network = InterfaceJSCC(definition['variant'], base['model'], definition['interface']).to(device)
    network.load_state_dict(state, strict=True)
    return network


def interface_losses(result, truth, images, decoder, perceptual, weights):
    unused, components, diagnostics = losses(result, truth, images, decoder, perceptual,
                                             {**weights, 'mse': 0., 'lpips': 0.})
    features = result['receiver_features']
    if not torch.isfinite(features).all() or bool((features.abs() > 1).any()):
        raise RuntimeError('registered receiver features escaped the finite bounded interface')
    if weights['mse'] or weights['lpips']:
        image = decoder.decode(features)
        result['image'] = image
        components['mse'] = (image - images).square().flatten(1).mean(1)
        components['lpips'] = perceptual(image * 2 - 1, images * 2 - 1).reshape(-1)
    objective = sum(weights[name] * values for name, values in components.items()).mean()
    diagnostics['feature_mse'] = (features.detach() - truth).square().flatten(1).mean(1)
    diagnostics['feature_abs_mean'] = features.detach().abs().flatten(1).mean(1)
    diagnostics['feature_saturation_fraction'] = features.detach().abs().ge(.99).float().flatten(1).mean(1)
    diagnostics['logit_abs_mean'] = result['logits'].detach().abs().flatten(1).mean(1)
    return objective, components, diagnostics


@torch.no_grad()
def interface_calibrate(network, population, decoder, perceptual, base, weights, device, positions=None):
    from pytorch_msssim import ssim

    network.eval()
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
            noise = torch.tensor(seeded_noise(identifiers[index], base['calibration']['noise_seed'], (1, 3060, 2)), device=device)
            result = communicate(network, truth, torch.tensor([snr], device=device), noise)
            unused, components, diagnostics = interface_losses(result, truth, target, decoder, perceptual, weights)
            mse = float(components['mse'][0])
            row = {'image_id': identifiers[index], 'snr_db': float(snr), 'interface': network.interface,
                   'psnr_db': float(-10 * np.log10(max(mse, 1e-12))), 'mse': mse,
                   'ssim': float(ssim(result['image'], target, data_range=1., size_average=True)),
                   'lpips': float(components['lpips'][0]), 'bits_BCE': float(components['bits'][0]),
                   'state_error': float(components['state'][0]),
                   **{key: float(diagnostics[key][0]) for key in ('bit_error_rate', 'group_error_rate',
                       'feature_mse', 'feature_abs_mean', 'feature_saturation_fraction', 'logit_abs_mean')},
                   'state_error_4': float(diagnostics['state_by_scale'][0, 0]),
                   'state_error_8': float(diagnostics['state_by_scale'][0, 1])}
            if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
                raise RuntimeError('nonfinite calibration values; no difficult sample may be dropped')
            rows.append(row)
    return rows
