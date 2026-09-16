"""Independent saved-output arithmetic for Joint waveforms, frozen replays and image/source pairing."""

import numpy as np
import torch

from .evaluation import fresh_names, measurement_order
from .evaluation_io import digest
from .references import References
from innovation_comm.archive_audit import check_image, check_features
from innovation_comm.evaluation import validate_population
from wetok_comm.deep_support import SUPPORT_NAME
from wetok_comm.evaluation import raw_noise
from wetok_comm.native import indices_to_features
from wetok_comm.training import read_population


def waveform_key(name, snr, seed=None):
    sender = 'frozen_shared' if name.startswith('frozen__') else name
    return f'{sender}__snr{float(snr)}' + ('' if seed is None else f'__seed{int(seed)}')


def check_channel(waveforms, rows, identifier, config, reference, base):
    names = fresh_names(config, reference)
    lookup = {(float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    maximum_error, maximum_power_error = 0., 0.
    for snr in base['evaluation']['snrs_db']:
        for seed in base['evaluation']['noise_seeds']:
            noise = raw_noise(identifier, seed)
            for name in names:
                signal = waveforms[waveform_key(name, snr)]
                received = waveforms[waveform_key(name, snr, seed)]
                if any(value.dtype != np.float32 or value.shape != (1, 3060, 2) or not np.isfinite(value).all() for value in (signal, received)):
                    raise RuntimeError('actual Joint/frozen waveform shape, precision or finiteness differs')
                power_error = abs(float(np.mean(np.sum(signal.astype(np.float64) ** 2, axis=-1))) - 2)
                maximum_power_error = max(maximum_power_error, power_error)
                expected = signal + noise.astype(np.float32)[None] * np.float32(10 ** (-float(snr) / 20))
                error = float(np.max(np.abs(received - expected)))
                maximum_error = max(maximum_error, error)
                if power_error > 1e-5 or error > 2e-6:
                    raise RuntimeError('saved waveform does not satisfy the paid energy/AWGN model')
                row = lookup[snr, seed, name]
                if (row['image_id'] != identifier or row['noise_sha256'] != digest(noise) or
                    row['transmitted_sha256'] != digest(signal) or row['received_sha256'] != digest(received)):
                    raise RuntimeError('measured source/observation is not the one recorded in the result row')
    return maximum_error, maximum_power_error


def audit_saved(root, rows, noiseless, support, evaluation, config, reference, base):
    torch.set_num_threads(2)
    images, codes, identifiers = read_population(base, 'development')
    validate_population(identifiers)
    references = References(evaluation)
    measured = fresh_names(config, reference)
    local_main, local_noiseless, local_support = {}, {}, {}
    for mapping, values in ((local_main, rows), (local_noiseless, noiseless), (local_support, support)):
        for row in values:
            mapping.setdefault(int(row['image_index']), []).append(row)
    count, feature_count, max_psnr, max_noise, max_power = 0, 0, 0., 0., 0.
    for index, identifier in enumerate(identifiers):
        directory = root / f'images/{index:03d}'
        with np.load(directory / 'waveforms.npz', allow_pickle=False) as waveforms:
            noise_error, power_error = check_channel(waveforms, local_main[index], identifier, config, reference, base)
        max_noise, max_power = max(max_noise, noise_error), max(max_power, power_error)
        source = images[index].numpy()
        truth = indices_to_features(torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)))[0].numpy()
        with np.load(directory / 'reconstructions.npz', allow_pickle=False) as archive:
            new_images = archive['images']
        with np.load(directory / 'receiver_features.npz', allow_pickle=False) as features:
            for row in local_main[index] + local_noiseless[index] + local_support[index]:
                if row['image_id'] != identifier:
                    raise RuntimeError('archived Joint outcome belongs to a different source')
                name = row['arm']
                previous_image, previous = None, None
                if not name.startswith('joint__'):
                    if name == SUPPORT_NAME:
                        previous_image, previous = references.support(index, identifier, float(row['snr_db']), int(row['seed']), row['noise_sha256'])
                    elif 'channel' in row:
                        previous_image, previous = references.noiseless(index, identifier, name.split('__', 1)[1])
                    else:
                        previous_image, previous = references.image(index, identifier, float(row['snr_db']), int(row['seed']), name, row['noise_sha256'])
                if row['image_store'] == 'verified_existing_reference':
                    if previous is None or row['image_archive'] != previous['image_archive'] or int(row['image_ref']) != int(previous['image_ref']):
                        raise RuntimeError('existing image pointer is not the qualified reference for this result')
                    image = previous_image.numpy()
                elif row['image_store'] == 'new_measured_output':
                    if row['image_archive'] != str(directory / 'reconstructions.npz'):
                        raise RuntimeError('new Joint output points to an external image')
                    image = new_images[int(row['image_ref'])]
                else:
                    raise RuntimeError('unknown image ownership/provenance policy')
                max_psnr = max(max_psnr, check_image(image, source, row))
                if name.startswith('frozen__') and np.max(np.abs(image - previous_image.numpy())) > evaluation['frozen_pixel_max_error']:
                    raise RuntimeError('CPU replay of frozen receiver pixels differs from the qualified result')
                count += 1
                if name in measured:
                    suffix = '__noiseless19' if 'channel' in row else f'__snr{float(row["snr_db"])}__seed{int(row["seed"])}'
                    check_features(features[name + suffix], truth, row)
                    feature_count += 1
                    if 'channel' not in row:
                        snr_index = base['evaluation']['snrs_db'].index(float(row['snr_db']))
                        seed_index = base['evaluation']['noise_seeds'].index(int(row['seed']))
                        frame = (index * len(base['evaluation']['snrs_db']) + snr_index) * len(base['evaluation']['noise_seeds']) + seed_index
                        if measurement_order(measured, frame)[int(row['receiver_order_index'])] != name:
                            raise RuntimeError('registered balanced timing order changed')
        print(f'Joint actual-waveform/image CPU audit {index + 1}/100', flush=True)
    return {'status': 'JOINT_SAVED_WAVEFORM_IMAGE_CPU_AUDIT_PASS', 'checked_images': count, 'checked_feature_rows': feature_count,
        'maximum_PSNR_error_dB': max_psnr, 'maximum_AWGN_reconstruction_error': max_noise, 'maximum_power_error': max_power,
        'GPU_used': False, 'role': 'saved-output arithmetic_not_new_training_or_independent_holdout'}
