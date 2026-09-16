"""CPU rechecks of the saved actual channel, images and receiver feature diagnostics."""

import hashlib

import numpy as np
import torch

from .evaluation import References, receiver_name, reference_names, validate_population
from .hardware import receiver_order
from wetok_comm.evaluation import raw_noise
from wetok_comm.native import indices_to_features
from wetok_comm.training import read_population


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def check_channel(signals, rows, identifier, base):
    maximum_error, maximum_power_error = 0., 0.
    lookup = {(float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    methods = sorted({row['arm'] for row in rows if row['arm'].startswith('receiver__')})
    if len(methods) != 6:
        raise RuntimeError('saved channel audit requires all six receivers')
    for snr in base['evaluation']['snrs_db']:
        signal = signals[f'shared__snr{snr}']
        if signal.dtype != np.float32 or signal.shape != (1, 3060, 2) or not np.isfinite(signal).all():
            raise RuntimeError('shared transmitted waveform changed shape or precision')
        power_error = abs(float(np.mean(np.sum(signal.astype(np.float64) ** 2, axis=-1))) - 2)
        maximum_power_error = max(maximum_power_error, power_error)
        if power_error > 1e-5:
            raise RuntimeError('saved transmitted waveform violates its power budget')
        signal_hash = digest(signal)
        for seed in base['evaluation']['noise_seeds']:
            noise = raw_noise(identifier, seed)
            noise_hash = digest(noise)
            received = signals[f'received__snr{snr}__seed{seed}']
            if received.dtype != np.float32 or received.shape != (1, 3060, 2) or not np.isfinite(received).all():
                raise RuntimeError('saved received waveform changed shape or precision')
            expected = signal + noise.astype(np.float32)[None] * np.float32(10 ** (-float(snr) / 20))
            error = float(np.max(np.abs(received - expected)))
            maximum_error = max(maximum_error, error)
            if error > 2e-6:
                raise RuntimeError('saved y does not equal the actual signal plus registered AWGN')
            for name in methods:
                row = lookup[snr, seed, name]
                if (row['image_id'] != identifier or row['noise_sha256'] != noise_hash or
                    row['shared_transmitted_sha256'] != signal_hash or row['shared_received_sha256'] != digest(received)):
                    raise RuntimeError('saved signal/observation is not the one scored by every receiver')
    return maximum_error, maximum_power_error


def check_image(image, source, row):
    if (image.dtype != np.float32 or image.shape != (3, 256, 256) or not np.isfinite(image).all() or
        image.min() < 0 or image.max() > 1 or digest(image) != row['image_sha256']):
        raise RuntimeError('saved reconstruction does not match its actual metric row')
    mse = float(np.mean((image.astype(np.float64) - source.astype(np.float64)) ** 2))
    error = abs(float(-10 * np.log10(max(mse, 1e-12))) - float(row['psnr_db']))
    if error > 1e-4:
        raise RuntimeError('CPU source-image PSNR differs from the GPU evaluation result')
    return error


def check_features(features, truth, row):
    if features.dtype != np.float32 or features.shape != (32, 16, 16) or not np.isfinite(features).all() or np.any(np.abs(features) > 1):
        raise RuntimeError('saved receiver features violate the registered continuous interface')
    errors = np.mean((features.astype(np.float64) - truth.astype(np.float64)) ** 2)
    signs = np.where(features > 0, 1., -1.)
    if abs(float(errors) - float(row['feature_mse'])) > 1e-6 or abs(float(np.mean(signs != truth)) - float(row['bit_error_rate'])) > 1e-10:
        raise RuntimeError('saved mean features do not reproduce the diagnostic state/bit errors')


def audit_saved_evaluation(root, rows, noiseless_rows, support_rows, evaluation, config, base):
    torch.set_num_threads(2)
    images, codes, identifiers = read_population(base, 'development')
    validate_population(identifiers)
    references = References(evaluation)
    methods = [receiver_name(name) for name in config['variants']]
    by_source, diagnostics, support = {}, {}, {}
    for key, collection in ((by_source, rows), (diagnostics, noiseless_rows), (support, support_rows)):
        for row in collection:
            key.setdefault(int(row['image_index']), []).append(row)
    checked, checked_features, maximum_psnr_error, maximum_channel_error, maximum_power_error = 0, 0, 0., 0., 0.
    for index, identifier in enumerate(identifiers):
        directory = root / f'images/{index:03d}'
        local_rows = by_source[index]
        for row in local_rows:
            if row['arm'] in methods:
                snr_position = base['evaluation']['snrs_db'].index(float(row['snr_db']))
                noise_position = base['evaluation']['noise_seeds'].index(int(row['seed']))
                frame_index = (index * len(base['evaluation']['snrs_db']) + snr_position) * len(base['evaluation']['noise_seeds']) + noise_position
                if receiver_order(methods, frame_index)[int(row['receiver_order_index'])] != row['arm']:
                    raise RuntimeError('balanced receiver timing order changed')
        with np.load(directory / 'waveforms.npz', allow_pickle=False) as signals:
            channel_error, power_error = check_channel(signals, local_rows, identifier, base)
        maximum_channel_error = max(maximum_channel_error, channel_error)
        maximum_power_error = max(maximum_power_error, power_error)
        source = images[index].numpy()
        grouped = torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True))
        truth = indices_to_features(grouped)[0].numpy()
        with np.load(directory / 'reconstructions.npz', allow_pickle=False) as archive:
            reconstructions = archive['images']
        with np.load(directory / 'receiver_features.npz', allow_pickle=False) as features:
            for row in local_rows + diagnostics[index] + support[index]:
                if row['image_id'] != identifier:
                    raise RuntimeError('saved result belongs to a different source')
                if row['arm'] in reference_names():
                    image, previous, archive = references.image(index, identifier, float(row['snr_db']), int(row['seed']), row['arm'], row['noise_sha256'])
                    image = image.numpy()
                else:
                    if row['image_archive'] != str(directory / 'reconstructions.npz'):
                        raise RuntimeError('new output was replaced by an external image archive')
                    image = reconstructions[int(row['image_ref'])]
                maximum_psnr_error = max(maximum_psnr_error, check_image(image, source, row))
                checked += 1
                if row['arm'] in methods:
                    suffix = '__noiseless19' if 'channel' in row else f'__snr{float(row["snr_db"])}__seed{int(row["seed"])}'
                    check_features(features[row['arm'] + suffix], truth, row)
                    checked_features += 1
        print(f'CPU archived channel/image audit {index + 1}/100', flush=True)
    return {'status': 'INNOVATION_SAVED_CHANNEL_IMAGE_CPU_AUDIT_PASS', 'checked_image_rows': checked,
        'checked_feature_rows': checked_features, 'maximum_PSNR_error_dB': maximum_psnr_error,
        'maximum_saved_y_reconstruction_error': maximum_channel_error, 'maximum_power_error': maximum_power_error,
        'GPU_used': False, 'scope': 'Independent arithmetic on saved outputs, not independent training or a new image-render implementation'}
