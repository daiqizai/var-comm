"""Same-frozen-model timing on a fixed source subset; never selects quality outputs."""

import numpy as np


def measurement_order(names, frame):
    names = list(names)
    if len(names) != 11 or len(set(names)) != 11 or frame < 0:
        raise ValueError('timing requires all eleven distinct learned models')
    shift = frame % len(names)
    return names[shift:] + names[:shift]


def validate_config(config):
    indices = config['timing_source_indices']
    expected = np.rint(np.linspace(0, 99, 32)).astype(int).tolist()
    if (indices != expected or config['timing_snrs_db'] != [1., 4., 7., 13., 19.] or
        config['timing_noise_seed'] != 2001 or config['timing_models'] != 11 or config['timing_rows'] != 1760):
        raise RuntimeError('predeclared timing sources, conditions or count changed')


def validate_rows(rows, config, names):
    validate_config(config)
    expected = {(index, snr, config['timing_noise_seed'], name) for index in config['timing_source_indices']
                for snr in config['timing_snrs_db'] for name in names}
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise RuntimeError('matched timing matrix is incomplete or duplicated')
    for position, index in enumerate(config['timing_source_indices']):
        for snr_position, snr in enumerate(config['timing_snrs_db']):
            frame = position * len(config['timing_snrs_db']) + snr_position
            order = measurement_order(names, frame)
            for name in names:
                row = lookup[index, snr, config['timing_noise_seed'], name]
                if (order[int(row['receiver_order_index'])] != name or int(row['total_complex_uses']) != 3060 or
                    abs(float(row['total_energy']) - 6120) > 1e-5 or row['timing_scope'] != 'separate_matched_uncontended_RX_including_visual_Decoder'):
                    raise RuntimeError('timing order, physical budget or scope changed')
                if any(not np.isfinite(float(row[key])) or float(row[key]) < 0 for key in ('online_TX_seconds', 'receiver_seconds')):
                    raise RuntimeError('invalid measured latency')
    return lookup


def summarize(rows, config, names, base):
    lookup = validate_rows(rows, config, names)
    indices = config['timing_source_indices']
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(len(indices),
        size=(base['evaluation']['bootstrap_resamples'], len(indices)))
    comparisons = [('joint__multiscale_state_history', 'grid__full_grid_state_history'),
                   ('grid__full_grid_innovation', 'grid__full_grid_state_history'),
                   ('grid__full_grid_innovation', 'frozen__full_grid_innovation')]
    summaries, paired = [], []
    for snrs in (base['evaluation']['primary_snrs_db'], config['timing_snrs_db'], *[[snr] for snr in config['timing_snrs_db']]):
        label = '+'.join(map(str, snrs))
        values = {}
        for name in names:
            subset = [lookup[index, snr, config['timing_noise_seed'], name] for index in indices for snr in snrs]
            summaries.append({'arm': name, 'snrs_db': label, 'source_images': len(indices), 'noise_seeds_per_source': 1,
                **{key + suffix: float(function([float(row[key]) for row in subset]))
                   for key in ('online_TX_seconds', 'receiver_seconds')
                   for suffix, function in (('_mean', np.mean), ('_median', np.median), ('_p95', lambda values: np.percentile(values, 95)))}})
            values[name] = np.array([np.mean([float(lookup[index, snr, config['timing_noise_seed'], name]['receiver_seconds']) for snr in snrs]) for index in indices])
        for method, control in comparisons:
            difference = values[method] - values[control]
            low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
            paired.append({'snrs_db': label, 'method': method, 'control': control, 'source_images': len(indices),
                'metric': 'receiver_seconds', 'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired
