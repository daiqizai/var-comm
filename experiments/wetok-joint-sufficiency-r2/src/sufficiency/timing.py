"""Separate timing for fifteen quality-selected models with explicit training-history labels."""

import numpy as np

from grid_controls.evaluation import audited_grid, model_registry as grid_registry
from joint_sender.evaluation import model_registry as joint_registry
from sufficiency.evaluation import learned_names, model_registry, previous_name
from wetok_comm.training import module_sha256


def measurement_order(names, frame):
    names = list(names)
    if len(names) != 15 or len(set(names)) != 15 or frame < 0:
        raise ValueError('R2 timing requires fifteen distinct learned models')
    shift = frame % len(names)
    return names[shift:] + names[:shift]


def validate_config(config):
    indices = np.rint(np.linspace(0, 99, 32)).astype(int).tolist()
    if (config['timing_source_indices'] != indices or config['timing_snrs_db'] != [1., 4., 7., 13., 19.] or
        config['timing_noise_seed'] != 2001 or config['timing_models'] != 15 or config['timing_rows'] != 2400):
        raise RuntimeError('R2 timing population, conditions or model count changed')


def all_models(config, original, grid, reference, base, milestone, device):
    parent, models, choices = model_registry(config, reference, base, milestone, device)
    grid_endpoint, unused_path, unused_review, bindings = audited_grid(grid, original, reference, 5000)
    older_parent, grid_models, grid_choices = grid_registry(grid, reference, base, grid_endpoint, device)
    oldest_parent, joint_models, joint_choices = joint_registry(original, reference, base, bindings['joint'], bindings['frozen'], device)
    if len({module_sha256(value) for value in (parent, older_parent, oldest_parent)}) != 1:
        raise RuntimeError('R2 timing models changed their declared common parent')
    models.update(grid_models)
    models.update(joint_models)
    choices.update(grid_choices)
    choices.update(joint_choices)
    if set(models) != set(learned_names(config, original, grid, reference)) or set(choices) != set(models):
        raise RuntimeError('R2 timing omitted a frozen learned reference')
    return parent, models, choices


def waveform_key(name, snr, seed=None):
    owner = 'frozen_shared' if name.startswith('frozen__') else name
    return f'{owner}__snr{float(snr)}' + ('' if seed is None else f'__seed{int(seed)}')


def archive_origin(name):
    if name.startswith('r2__'):
        return 'r2'
    if name.startswith('grid__'):
        return 'grid'
    if name.startswith(('joint__', 'frozen__')):
        return 'joint'
    raise ValueError('unknown R2 timing waveform owner')


def comparisons(names):
    pairs = [('r2__' + method, 'r2__' + control) for method, control in (
        ('multiscale_state_history', 'full_grid_state_history'), ('multiscale_state_history', 'single_pass'),
        ('full_grid_state_history', 'single_pass'), ('full_grid_innovation', 'single_pass'),
        ('full_grid_innovation', 'multiscale_state_history'), ('full_grid_innovation', 'full_grid_state_history'))]
    pairs.extend((name, previous_name(name.split('__', 1)[1])) for name in names if name.startswith('r2__'))
    if len(pairs) != 10 or any(method not in names or control not in names for method, control in pairs):
        raise RuntimeError('R2 timing comparisons lost a matched structure or own5000 reference')
    return pairs


def validate_rows(rows, config, names):
    validate_config(config)
    measurement_order(names, 0)
    expected = {(index, snr, config['timing_noise_seed'], name) for index in config['timing_source_indices']
        for snr in config['timing_snrs_db'] for name in names}
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise RuntimeError('R2 timing matrix is incomplete or duplicated')
    identifiers = set()
    for position, index in enumerate(config['timing_source_indices']):
        identities = {row['image_id'] for key, row in lookup.items() if key[0] == index}
        if len(identities) != 1 or next(iter(identities)) in identifiers:
            raise RuntimeError('R2 timing source identities were changed or duplicated')
        identifiers.update(identities)
        for snr_position, snr in enumerate(config['timing_snrs_db']):
            order = measurement_order(names, position * len(config['timing_snrs_db']) + snr_position)
            for name in names:
                row = lookup[index, snr, config['timing_noise_seed'], name]
                if (int(row['receiver_order_index']) != order.index(name) or int(row['total_complex_uses']) != 3060 or
                    abs(float(row['total_energy']) - 6120) > 1e-5 or
                    row['timing_scope'] != 'separate_matched_uncontended_RX_including_visual_Decoder'):
                    raise RuntimeError('R2 timing order, physical resources or measurement scope changed')
                if any(not np.isfinite(float(row[key])) or float(row[key]) <= 0 for key in ('online_TX_seconds', 'receiver_seconds')):
                    raise RuntimeError('R2 measured latency must be positive and finite')
    return lookup


def summarize(rows, config, names, base):
    lookup = validate_rows(rows, config, names)
    indices = config['timing_source_indices']
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(len(indices),
        size=(base['evaluation']['bootstrap_resamples'], len(indices)))
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
            values[name] = np.array([np.mean([float(lookup[index, snr, config['timing_noise_seed'], name]['receiver_seconds'])
                for snr in snrs]) for index in indices])
        for method, control in comparisons(names):
            difference = values[method] - values[control]
            low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
            paired.append({'snrs_db': label, 'method': method, 'control': control, 'source_images': len(indices),
                'comparison_scope': 'matched10000_structure_comparison' if control.startswith('r2__') else 'extra_training_effect_not_new_mechanism',
                'metric': 'receiver_seconds', 'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired
