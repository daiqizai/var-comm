"""Cache fixed WeTok native group indices; no communication optimizer is created."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np
import torch

from wetok_comm.common import PROJECT, REFERENCE, WORKSPACE, artifact_hashes, assets, configure_torch, now, output_path, settings, sha256, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok, indices_to_features


def image_population(name, binding):
    root = Path(binding['image_cache'])
    manifest = root / 'manifest.json'
    if sha256(manifest) != binding['image_manifest_sha256']:
        raise RuntimeError('original image manifest changed')
    record = json.loads(manifest.read_text())
    population = next(item for item in record['populations'] if item['name'] == name)
    images = torch.empty((population['count'], 3, 256, 256), dtype=torch.uint8)
    identifiers, offset = [], 0
    for shard in population['shards']:
        path = root / shard['path']
        if sha256(path) != shard['sha256']:
            raise RuntimeError('original training/calibration shard changed')
        values = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        count = len(values['targets_u8'])
        images[offset:offset + count].copy_(values['targets_u8'])
        identifiers.extend(values['image_ids'])
        offset += count
    if offset != len(images) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError('source population is incomplete or has duplicate IDs')
    return images, identifiers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/study.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config = settings(arguments.config)
    if not arguments.execute:
        print('PLAN ONLY: prepare native group-index cache; no model loaded')
        return
    configure_torch()
    binding = assets()
    output = output_path(config, 'cache')
    if (output / 'completion.json').exists():
        raise FileExistsError('native cache already complete; do not repeat it')
    if not arguments.resume:
        output.mkdir(parents=True, exist_ok=False)
        sources = snapshot(output, [Path(__file__), arguments.config, EXPERIMENT / 'docs/protocol.md',
            WORKSPACE / config['brief'], EXPERIMENT / 'src/wetok_comm/native.py', EXPERIMENT / 'src/wetok_comm/common.py',
            REFERENCE / 'scripts/wetok_adapter.py', Path(binding['wetok_config'])])
        metadata = {'created_local': now(), 'config_sha256': sha256(arguments.config), 'source_hashes': sources,
                    'source_image_manifest_sha256': binding['image_manifest_sha256'], 'optimizer_updates': 0,
                    'encoder_batch_size': 1, 'native_layout': [16, 16, 4], 'raw_bits_per_image': 8192}
        write_json(output / 'metadata.json', metadata)
    else:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if metadata['config_sha256'] != sha256(arguments.config):
            raise RuntimeError('cannot change cache configuration while resuming')
    progress_path = output / 'status.json'
    progress = json.loads(progress_path.read_text()) if arguments.resume and progress_path.exists() else {}
    completed = progress.get('completed', {'train': 0, 'calibration': 0, 'development': 0})
    started = time.time()
    def status(state, **fields):
        write_json(progress_path, {'status': state, 'local_time': now(), 'pid': os.getpid(),
                                  'completed': completed, 'optimizer_updates': 0, **fields})
    try:
        free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 8000:
            raise RuntimeError('authorized GPU currently lacks headroom; no other process will be stopped')
        status('LOADING_FROZEN_ENCODER')
        model = FrozenWeTok('cuda:0', 'encoder')
        write_json(output / 'native_model.json', model.metadata)
        populations = {}
        for name, count, views in (('train', 20000, 2), ('calibration', 1000, 1)):
            images, identifiers = image_population(name, binding)
            if len(images) != count:
                raise RuntimeError('registered data population size changed')
            populations[name] = identifiers
            write_json(output / f'{name}_ids.json', identifiers)
            path = output / f'{name}_codes.npy'
            codes = np.lib.format.open_memmap(path, mode='r+' if path.exists() and arguments.resume else 'w+',
                                              dtype=np.uint8, shape=(count, views, 16, 16, 4))
            for index in range(completed[name], count):
                image = images[index:index + 1].to('cuda:0').float().div(255)
                for view in range(views):
                    source = image.flip(-1) if view else image
                    codes[index, view] = model.encode(source)[0].cpu().numpy()
                completed[name] = index + 1
                if completed[name] % 100 == 0 or completed[name] == count:
                    codes.flush()
                    status('ENCODING_NATIVE_SOURCE', population=name, last_id=identifiers[index])
                    print(f'{name} native cache {completed[name]}/{count} views={views}', flush=True)
            codes.flush()
            del codes, images
        overlap = {Path(identifier).stem for identifier in populations['train']} & {Path(identifier).stem for identifier in populations['calibration']}
        if overlap:
            raise RuntimeError('training and calibration contain overlapping source IDs')
        reference = Path(binding['native_reference'])
        rows = [json.loads(line) for line in (reference / 'per_image.jsonl').read_text().splitlines()]
        by_index = {int(row['image_index']): row for row in rows}
        if len(by_index) != 100:
            raise RuntimeError('the original development population changed')
        model.codec.decoder.to('cuda:0')
        development_ids, identity_errors = [], []
        code_path = output / 'development_codes.npy'
        dev_codes = np.lib.format.open_memmap(code_path, mode='r+' if code_path.exists() and arguments.resume else 'w+',
                                              dtype=np.uint8, shape=(100, 1, 16, 16, 4))
        existing = output / 'development_identity.json'
        if existing.exists() and arguments.resume:
            identity_errors = json.loads(existing.read_text())
        for index in range(100):
            row = by_index[index]
            identifier = row.get('image_id', row.get('source_id', row.get('path')))
            if identifier is None:
                raise RuntimeError('reference does not identify its source image')
            development_ids.append(identifier)
            if index < completed['development']:
                continue
            payload = torch.load(reference / 'float_images' / f'{index:03d}.pt', map_location='cpu', weights_only=True)
            source = payload['source_01'].to('cuda:0')
            with torch.inference_mode():
                indices = model.encode(source)
                reconstructed = model.decode(indices_to_features(indices))
            originals = payload['images']
            reference_image = next(iter(originals.values())) if isinstance(originals, dict) else originals[0]
            reference_image = reference_image.to('cuda:0')
            error = float((reconstructed - reference_image).abs().max())
            if error > 1e-5:
                raise RuntimeError(f'native reference replay differs at development image {index}: {error}')
            dev_codes[index, 0] = indices[0].cpu().numpy()
            identity_errors.append({'image_index': index, 'image_id': identifier, 'max_absolute_image_error': error})
            completed['development'] = index + 1
            dev_codes.flush()
            write_json(output / 'development_identity.json', identity_errors)
            status('VERIFYING_NATIVE_REPLAY')
        populations['development'] = development_ids
        seen = [{Path(identifier).stem for identifier in population} for population in populations.values()]
        if any(seen[left] & seen[right] for left in range(3) for right in range(left + 1, 3)):
            raise RuntimeError('training/calibration/development source IDs overlap')
        write_json(output / 'development_ids.json', development_ids)
        verify_sources(metadata['source_hashes'])
        status('NATIVE_CACHE_COMPLETE')
        write_json(output / 'completion.json', {'status': 'NATIVE_CACHE_COMPLETE', 'completed_local': now(),
            'source_hashes': metadata['source_hashes'], 'config_sha256': metadata['config_sha256'],
            'counts': completed, 'raw_bits_per_image': 8192, 'encoder_batch_size': 1, 'native_identity_images': len(identity_errors),
            'maximum_identity_error': max(item['max_absolute_image_error'] for item in identity_errors),
            'optimizer_updates': 0, 'observed_wall_hours_this_session': (time.time() - started) / 3600,
            'checkpoint_sha256': model.metadata['checkpoint_sha256'], 'output_hashes': artifact_hashes(output)})
        print('NATIVE_CACHE_COMPLETE', flush=True)
    except BaseException as error:
        status('CACHE_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
