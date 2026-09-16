"""CPU arithmetic and pixel-pointer checks for new Grid quality and immutable references."""

import numpy as np
import torch

from grid_controls.evaluation import new_names
from grid_controls.references import ImageCache
from innovation_comm.archive_audit import check_features, check_image
from innovation_comm.evaluation import validate_population
from joint_sender.evaluation_io import digest
from wetok_comm.evaluation import raw_noise
from wetok_comm.native import indices_to_features
from wetok_comm.training import read_population


def waveform_key(name, snr, seed=None):
    return f'{name}__snr{float(snr)}' + ('' if seed is None else f'__seed{int(seed)}')


def check_channel(waveforms, rows, identifier, names, base):
    lookup = {(float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    maximum_noise_error, maximum_power_error = 0., 0.
    for snr in base['evaluation']['snrs_db']:
        for seed in base['evaluation']['noise_seeds']:
            noise = raw_noise(identifier, seed)
            for name in names:
                signal, received = waveforms[waveform_key(name, snr)], waveforms[waveform_key(name, snr, seed)]
                if any(value.dtype != np.float32 or value.shape != (1, 3060, 2) or not np.isfinite(value).all() for value in (signal, received)):
                    raise RuntimeError('Grid waveform shape, precision or finiteness changed')
                power = abs(float(np.mean(np.sum(signal.astype(np.float64) ** 2, axis=-1))) - 2)
                expected = signal + noise.astype(np.float32)[None] * np.float32(10 ** (-float(snr) / 20))
                error = float(np.max(np.abs(expected - received)))
                if power > 1e-5 or error > 2e-6:
                    raise RuntimeError('Grid archive violates the physical energy/AWGN ledger')
                row = lookup[float(snr), int(seed), name]
                if (row['image_id'] != identifier or row['noise_sha256'] != digest(noise) or
                    row['transmitted_sha256'] != digest(signal) or row['received_sha256'] != digest(received)):
                    raise RuntimeError('recorded Grid waveform is not the measured observation')
                maximum_noise_error, maximum_power_error = max(maximum_noise_error, error), max(maximum_power_error, power)
    return maximum_noise_error, maximum_power_error


def audit_saved(root, rows, clean, noiseless, support, config, base):
    torch.set_num_threads(2)
    images, codes, identifiers = read_population(base, 'development')
    validate_population(identifiers)
    names = new_names(config)
    by_source = {index: [] for index in range(100)}
    for row in rows + clean + noiseless + support:
        by_source[int(row['image_index'])].append(row)
    image_count, feature_count, max_noise, max_power, max_psnr = 0, 0, 0., 0., 0.
    for index, identifier in enumerate(identifiers):
        directory = root / 'images' / f'{index:03d}'
        source = images[index].numpy().astype(np.float32)
        if images.dtype == torch.uint8:
            source = source / 255
        truth = indices_to_features(torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)))[0].numpy()
        new_main = [row for row in by_source[index] if row.get('arm') in names and 'channel' not in row]
        with np.load(directory / 'waveforms.npz', allow_pickle=False) as waveforms:
            noise_error, power_error = check_channel(waveforms, new_main, identifier, names, base)
        max_noise, max_power = max(max_noise, noise_error), max(max_power, power_error)
        cache = ImageCache()
        with np.load(directory / 'receiver_features.npz', allow_pickle=False) as features:
            for row in by_source[index]:
                if row['image_id'] != identifier:
                    raise RuntimeError('a Grid/reference image belongs to another source')
                image = cache.image(row)
                max_psnr = max(max_psnr, check_image(image, source, row))
                image_count += 1
                if row.get('arm') in names:
                    if row['image_archive'] != str(directory / 'reconstructions.npz') or row['quality_origin'] != 'new_grid_measurement':
                        raise RuntimeError('new Grid image points outside its actual source archive')
                    suffix = '__noiseless19' if 'channel' in row else f'__snr{float(row["snr_db"])}__seed{int(row["seed"])}'
                    check_features(features[row['arm'] + suffix], truth, row)
                    feature_count += 1
        print(f'Grid saved quality CPU audit {index + 1}/100', flush=True)
    return {'status': 'GRID_SAVED_QUALITY_CPU_AUDIT_PASS', 'checked_image_rows': image_count,
        'checked_new_feature_rows': feature_count, 'maximum_PSNR_error_dB': max_psnr,
        'maximum_AWGN_reconstruction_error': max_noise, 'maximum_power_error': max_power,
        'GPU_used': False, 'new_inference': False, 'separate_timing_not_inferred_from_quality': True}
