"""CPU verification of R2 images, saved physical observations and inherited reference pixels."""

import numpy as np
import torch

from grid_controls.archive_audit import check_channel
from grid_controls.references import ImageCache
from innovation_comm.archive_audit import check_features, check_image
from innovation_comm.evaluation import validate_population
from sufficiency.evaluation import new_names
from wetok_comm.native import indices_to_features
from wetok_comm.training import read_population


def audit_saved(root, rows, clean, noiseless, support, config, base, references):
    torch.set_num_threads(2)
    images, codes, identifiers = read_population(base, 'development')
    validate_population(identifiers)
    names = new_names(config)
    by_source = {index: [] for index in range(100)}
    for row in rows + clean + noiseless + support:
        by_source[int(row['image_index'])].append(row)
    count, feature_count, max_noise, max_power, max_psnr = 0, 0, 0., 0., 0.
    for index, identifier in enumerate(identifiers):
        source_tensor = images[index:index + 1].float()
        if images.dtype == torch.uint8:
            source_tensor = source_tensor / 255
        references.validate_source(index, identifier, source_tensor)
        source = source_tensor[0].numpy()
        truth = indices_to_features(torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)))[0].numpy()
        directory = root / 'images' / f'{index:03d}'
        new_main = [row for row in by_source[index] if row.get('arm') in names and 'channel' not in row]
        with np.load(directory / 'waveforms.npz', allow_pickle=False) as waves:
            noise_error, power_error = check_channel(waves, new_main, identifier, names, base)
        max_noise, max_power = max(max_noise, noise_error), max(max_power, power_error)
        cache = ImageCache()
        with np.load(directory / 'receiver_features.npz', allow_pickle=False) as features:
            for row in by_source[index]:
                if row['image_id'] != identifier:
                    raise RuntimeError('R2 archived row belongs to a different source')
                image = cache.image(row)
                max_psnr = max(max_psnr, check_image(image, source, row))
                count += 1
                if row.get('arm') in names:
                    if row['image_archive'] != str(directory / 'reconstructions.npz') or row['r2_quality_origin'] != 'new_r2_measurement':
                        raise RuntimeError('new R2 image does not point to its measured source archive')
                    suffix = '__noiseless19' if 'channel' in row else f'__snr{float(row["snr_db"])}__seed{int(row["seed"])}'
                    check_features(features[row['arm'] + suffix], truth, row)
                    feature_count += 1
        print(f'R2 saved quality CPU audit {index + 1}/100', flush=True)
    return {'status': 'R2_SAVED_QUALITY_CPU_AUDIT_PASS', 'checked_image_rows': count, 'checked_new_feature_rows': feature_count,
        'maximum_PSNR_error_dB': max_psnr, 'maximum_AWGN_reconstruction_error': max_noise,
        'maximum_power_error': max_power, 'GPU_used': False, 'new_inference': False, 'no_new_holdout': True}
