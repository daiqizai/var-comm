#!/usr/bin/env python3
"""Cache only the first eight frozen source scales, including true retokenized horizontal flips."""

import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.prefix_training_data import IMAGE_CACHE, MANIFEST_SHA, read_image_population
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_snapshot, write_json


def pixel_hash(image):
    return hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()


def main():
    output = create_output(ROOT / 'outputs/VAR-PREFIX-TRAINING-DATA-001')
    sources = snapshot(output, [Path(__file__), ROOT / 'src/var_comm/prefix_training_data.py'])
    started = time.perf_counter()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    config = yaml.safe_load((ROOT / 'configs/next_scale_prior_diagnostic.yaml').read_text())
    for name in ('vae_checkpoint', 'var_checkpoint'):
        if sha256(config['paths'][name]) != config['paths'][name + '_sha256']:
            raise RuntimeError('frozen checkpoint changed')
    vae, var = load_models(config['paths'], device)
    del var
    torch.cuda.empty_cache()
    before = state_sha256(vae)
    development_hashes = set()
    for index in range(100):
        with np.load(ROOT / 'outputs/VAR-PROGRESSIVE-CHANNEL-001/images' / f'{index:03d}' / 'reconstructions.npz', allow_pickle=False) as cache:
            pixels = torch.tensor(np.rint(cache['source'] * 255).astype(np.uint8))
        development_hashes.update((pixel_hash(pixels), pixel_hash(pixels.flip(-1))))
    calibration_images, calibration_labels, calibration_ids, calibration_bindings = read_image_population('calibration')
    calibration_hashes = [pixel_hash(image) for image in calibration_images]
    calibration_flip_hashes = [pixel_hash(image.flip(-1)) for image in calibration_images]
    keep_calibration = [index for index, digest in enumerate(calibration_hashes) if digest not in development_hashes and calibration_flip_hashes[index] not in development_hashes]
    blocked = development_hashes | set(calibration_hashes) | set(calibration_flip_hashes)
    counts = np.zeros(4096, dtype=np.int64)
    populations = []
    for name in ('calibration', 'train'):
        if name == 'calibration':
            images, labels, ids, bindings = calibration_images, calibration_labels, calibration_ids, calibration_bindings
            hashes, flip_hashes = calibration_hashes, calibration_flip_hashes
            kept = keep_calibration
        else:
            images, labels, ids, bindings = read_image_population('train')
            hashes = [pixel_hash(image) for image in images]
            flip_hashes = [pixel_hash(image.flip(-1)) for image in images]
            kept = [index for index, digest in enumerate(hashes) if digest not in blocked and flip_hashes[index] not in blocked]
        entries = [{**bindings[index], 'source_index': index, 'pixel_sha256': hashes[index], 'flip_pixel_sha256': flip_hashes[index]} for index in kept]
        prefix = np.empty((len(kept), 2 if name == 'train' else 1, 255), dtype=np.uint16)
        with torch.no_grad():
            for start in range(0, len(kept), 16):
                source = images[kept[start:start + 16]].to(device).float().div(127.5).sub(1)
                views = torch.cat((source, source.flip(-1)), dim=0) if name == 'train' else source
                scales = vae.img_to_idxBl(views)
                if len(scales) != 10:
                    raise RuntimeError('the official full scale schedule changed')
                values = torch.cat(scales[:8], dim=1).cpu().numpy().astype(np.uint16)
                count = len(source)
                prefix[start:start + count, 0] = values[:count]
                if name == 'train':
                    prefix[start:start + count, 1] = values[count:]
                if start % 1000 < 16:
                    print(f'prefix source cache {name} {min(start + count, len(kept))}/{len(kept)}', flush=True)
        if name == 'train':
            counts += np.bincount(prefix.ravel(), minlength=4096)
        np.savez(output / (name + '.npz'), tokens=prefix, source_indices=np.asarray(kept), labels=labels[kept].numpy(), image_ids=np.asarray([ids[index] for index in kept]))
        write_json(output / (name + '_manifest.json'), entries)
        populations.append({'name': name, 'source_count': len(images), 'count': len(kept), 'excluded_content_overlaps': len(images) - len(kept),
                            'class_count': int(len(torch.unique(labels[kept]))), 'views': prefix.shape[1]})
        if name == 'train':
            del images
    codebook = vae.quantize.embedding.weight.detach().cpu().numpy().astype(np.float64)
    probability = counts / counts.sum()
    mean = probability @ codebook
    std = np.sqrt(np.maximum(probability @ (codebook ** 2) - mean ** 2, 1e-8))
    np.savez(output / 'embedding_statistics.npz', counts=counts, mean=mean.astype(np.float32), std=std.astype(np.float32))
    if state_sha256(vae) != before:
        raise RuntimeError('tokenization modified VAE weights')
    verify_snapshot(sources)
    write_json(output / 'completion.json', {'status': 'PREFIX_TRAINING_DATA_READY', 'populations': populations, 'source_hashes': sources,
                                           'source_cache': str(IMAGE_CACHE), 'source_manifest_sha256': MANIFEST_SHA,
                                           'frozen_vae_state_sha256': before, 'no_cached_fhat_or_DINO_used': True,
                                           'full_ten_scale_tokenizer_first_eight_only': True, 'horizontal_flip_retokenized': True,
                                           'elapsed_seconds': time.perf_counter() - started, 'output_hashes': artifact_hashes(output)})
    print(populations, flush=True)


if __name__ == '__main__':
    main()
