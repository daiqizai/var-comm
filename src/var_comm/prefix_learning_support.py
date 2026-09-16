"""Frozen perceptual losses and trainable communication-only input/output helpers."""

from pathlib import Path

import lpips
import numpy as np
import torch
from torch.nn import functional as functional
import yaml

from .learned_prefix import PrefixJSCC
from .next_scale_prior import load_models
from .study import ROOT, sha256


def load_frozen_training_models(config, device):
    model_config = yaml.safe_load((ROOT / config['model_config']).read_text())
    quality = yaml.safe_load((ROOT / config['quality_config']).read_text())['quality']
    for paths, names in ((model_config['paths'], ('vae_checkpoint', 'var_checkpoint')), (quality, ('alexnet_checkpoint',))):
        for name in names:
            if sha256(paths[name]) != paths[name + '_sha256']:
                raise RuntimeError('a fixed training backbone checkpoint changed')
    vae, var = load_models(model_config['paths'], device)
    weights_path = Path(lpips.__file__).parent / 'weights/v0.1/alex.pth'
    perceptual = lpips.LPIPS(net='alex', pnet_rand=True, model_path=str(weights_path), verbose=False)
    weights = torch.load(quality['alexnet_checkpoint'], map_location='cpu', weights_only=True)
    translated = {name: weights['features.' + name.split('.', 1)[1]] for name in perceptual.net.state_dict()}
    perceptual.net.load_state_dict(translated, strict=True)
    return vae, var, perceptual.to(device).eval().requires_grad_(False)


def build_codec(config, vae, variant, device):
    with np.load(ROOT / config['data_cache'] / 'embedding_statistics.npz', allow_pickle=False) as cache:
        mean = torch.tensor(cache['mean'], device=device)
        std = torch.tensor(cache['std'], device=device)
    return PrefixJSCC(vae.quantize.embedding.weight, mean, std, variant,
                      width=config['model']['width'], layers=config['model']['layers']).to(device)


def communication_forward(codec, tokens, decoded_labels, snrs, data_noise, valid, vae, var,
                          teacher_tokens=None, teacher_mask=None):
    symbols = codec.transmit(tokens)
    gamma = torch.pow(10.0, snrs / 10.0)
    received = symbols + data_noise * gamma.rsqrt()[:, None, None]
    result = codec.receive(received, decoded_labels, snrs, vae, var, teacher_tokens, teacher_mask)
    result['image'] = torch.where(valid[:, None, None, None], result['image'], result['image'].new_full((), 0.5))
    result['symbols'] = symbols
    result['received'] = received
    return result


def image_losses(result, targets, tokens, valid, perceptual, config):
    mse = (result['image'] - targets).square().flatten(1).mean(1)
    lpips_values = perceptual(result['image'] * 2 - 1, targets * 2 - 1).flatten()
    token = functional.cross_entropy(result['logits'].transpose(1, 2), tokens, reduction='none').mean(1) * valid
    objective = (config['image_mse_weight'] * mse + config['image_lpips_weight'] * lpips_values + config['token_ce_weight'] * token).mean()
    return objective, {'mse': mse, 'lpips': lpips_values, 'token_ce': token}
