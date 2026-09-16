"""Paired data, frozen perceptual loss and calibration utilities."""

import hashlib
import json
from pathlib import Path

import lpips
import numpy as np
import torch
import yaml

from .common import EXPERIMENT, WORKSPACE, assets, load_module, output_path, sha256, verify_sources
from .model import NativeWeTokJSCC, communicate
from .native import indices_to_features
from .objective import losses


def module_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_lpips(device):
    binding = assets()
    quality = yaml.safe_load((WORKSPACE / 'var-next-scale-comm/configs/quality.yaml').read_text())['quality']
    path = Path(binding['alexnet_checkpoint'])
    if sha256(path) != quality['alexnet_checkpoint_sha256']:
        raise RuntimeError('frozen AlexNet metric weights changed')
    linear = Path(lpips.__file__).parent / 'weights/v0.1/alex.pth'
    network = lpips.LPIPS(net='alex', pnet_rand=True, model_path=str(linear), verbose=False)
    stored = torch.load(path, map_location='cpu', weights_only=True)
    translated = {name: stored['features.' + name.split('.', 1)[1]] for name in network.net.state_dict()}
    network.net.load_state_dict(translated, strict=True)
    return network.to(device).eval().requires_grad_(False)


def read_population(config, name):
    root = output_path(config, 'cache')
    receipt = json.loads((root / 'completion.json').read_text())
    if receipt['status'] != 'NATIVE_CACHE_COMPLETE' or receipt['counts'] != {'train': 20000, 'calibration': 1000, 'development': 100}:
        raise RuntimeError('complete native source cache is required')
    verify_sources(receipt['source_hashes'])
    metadata_path = root / 'metadata.json'
    if sha256(metadata_path) != receipt['output_hashes']['metadata.json']:
        raise RuntimeError('cache input binding changed')
    metadata = json.loads(metadata_path.read_text())
    if metadata['source_image_manifest_sha256'] != assets()['image_manifest_sha256']:
        raise RuntimeError('current image population differs from the encoded source cache')
    code_path = root / f'{name}_codes.npy'
    ids_path = root / f'{name}_ids.json'
    for path in (code_path, ids_path):
        if sha256(path) != receipt['output_hashes'][str(path.relative_to(root))]:
            raise RuntimeError('cached codes/IDs changed')
    indices = np.load(code_path, mmap_mode='r', allow_pickle=False)
    identifiers = json.loads(ids_path.read_text())
    if name in ('train', 'calibration'):
        module = load_module('wetok_cache_population_reader', EXPERIMENT / 'scripts/prepare_native_cache.py')
        images, original_ids = module.image_population(name, assets())
        if original_ids != identifiers:
            raise RuntimeError('native codes do not match the source image ordering')
    else:
        reference = Path(assets()['native_reference']) / 'float_images'
        images = torch.cat([torch.load(reference / f'{index:03d}.pt', map_location='cpu', weights_only=True)['source_01']
                            for index in range(100)])
    return images, indices, identifiers


def paired_batches(config, count, start_step, until_step):
    batch_size = config['training']['effective_batch_size']
    if count % batch_size or until_step <= start_step:
        raise ValueError('invalid complete-batch training interval')
    per_epoch = count // batch_size
    for epoch in range(start_step // per_epoch, (until_step - 1) // per_epoch + 1):
        ordering = torch.Generator().manual_seed(config['training']['order_seed'] + epoch)
        order = torch.randperm(count, generator=ordering)
        flips = torch.rand(count, generator=ordering) < .5
        channel = torch.Generator().manual_seed(config['training']['channel_seed'] + epoch)
        for offset in range(0, count, batch_size):
            step = epoch * per_epoch + offset // batch_size
            snr_indices = torch.randint(len(config['channel']['snrs_db']), (batch_size,), generator=channel)
            snrs = torch.tensor(config['channel']['snrs_db'])[snr_indices]
            noise = torch.randn((batch_size, 3060, 2), generator=channel)
            if step < start_step:
                continue
            if step >= until_step:
                return
            selected, flip = order[offset:offset + batch_size], flips[offset:offset + batch_size]
            digest = hashlib.sha256()
            for value in (selected, flip, snrs, noise):
                digest.update(value.numpy().tobytes())
            yield {'step': step, 'epoch': epoch, 'indices': selected, 'flip': flip, 'snrs': snrs,
                   'noise': noise, 'fingerprint': digest.hexdigest()}


def batch_inputs(population, batch, device):
    images, codes, identifiers = population
    selected = batch['indices'].numpy()
    flip = batch['flip']
    grouped = torch.from_numpy(np.array(codes[selected, flip.long().numpy()], copy=True)).to(device)
    target = images[batch['indices']].to(device).float().div(255)
    target = torch.where(flip.to(device)[:, None, None, None], target.flip(-1), target)
    return {'source_fq': indices_to_features(grouped), 'images': target,
            'snrs': batch['snrs'].to(device), 'noise': batch['noise'].to(device)}


def new_network(config, arm, device):
    torch.manual_seed(config['training']['initialization_seed'])
    return NativeWeTokJSCC(arm, config['model']).to(device)


def stage(config, completed, joint_rate=None):
    if completed < config['training']['representation_updates']:
        return 'representation', config['training']['representation_weights'], config['training']['representation_learning_rate']
    return 'joint', config['training']['joint_weights'], joint_rate or config['training']['image_learning_rate']


def seeded_noise(identifier, seed, shape):
    key = int.from_bytes(hashlib.sha256(f'{identifier}|{seed}'.encode()).digest()[:8], 'little')
    return np.random.default_rng(key).standard_normal(shape).astype(np.float32)


def monitoring_indices(identifiers, config):
    seed = config['calibration']['monitor_subset_seed']
    return sorted(range(len(identifiers)), key=lambda index: hashlib.sha256(f'{seed}|{identifiers[index]}'.encode()).digest())[:config['calibration']['monitor_images']]


@torch.no_grad()
def calibrate(network, population, native, perceptual, config, device, positions=None, snrs=None, noiseless=False):
    from pytorch_msssim import ssim

    network.eval()
    images, codes, identifiers = population
    positions = list(range(len(images))) if positions is None else list(positions)
    snrs = config['channel']['snrs_db'] if snrs is None else snrs
    rows = []
    for snr in snrs:
        for index in positions:
            native_indices = torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)).to(device)
            source = indices_to_features(native_indices)
            noise = torch.tensor(seeded_noise(identifiers[index], config['calibration']['noise_seed'], (1, 3060, 2)), device=device)
            target = images[index:index + 1].to(device).float()
            if images.dtype == torch.uint8:
                target = target.div(255)
            result = communicate(network, source, torch.tensor([snr], device=device), noise, noiseless)
            objective, components, diagnostic = losses(result, source, target, native, perceptual, config['training']['joint_weights'])
            mse = float(components['mse'][0])
            row = {'image_id': identifiers[index], 'snr_db': float(snr), 'noiseless': noiseless,
                   'psnr_db': float(-10 * np.log10(max(mse, 1e-12))), 'ssim': float(ssim(result['image'], target, data_range=1., size_average=True)),
                   'lpips': float(components['lpips'][0]), 'mse': mse, 'bits_BCE': float(components['bits'][0]),
                   'state_error': float(components['state'][0]), 'bit_error_rate': float(diagnostic['bit_error_rate'][0]),
                   'group_error_rate': float(diagnostic['group_error_rate'][0]),
                   'state_error_4': float(diagnostic['state_by_scale'][0, 0]), 'state_error_8': float(diagnostic['state_by_scale'][0, 1])}
            if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
                raise RuntimeError('nonfinite calibration metrics; never drop difficult samples')
            rows.append(row)
    return rows
