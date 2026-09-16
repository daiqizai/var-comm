"""Separate sixteen-model timing with matched vector and historical comparisons distinguished."""

import numpy as np

from sufficiency.evaluation import audited_endpoint as r2_endpoint
from sufficiency.timing import all_models as r2_models, archive_origin as r2_archive_origin, waveform_key
from vector_control.evaluation import comparison_scope, learned_names, model_registry, new_names
from wetok_comm.training import module_sha256


def all_models(config, r2, original, grid, reference, base, milestone, device):
    parent, models, choices = model_registry(config, reference, base, milestone, device)
    old_milestone, unused_path, unused_review, unused_sources = r2_endpoint(r2, 10000)
    old_parent, old_models, old_choices = r2_models(r2, original, grid, reference, base, old_milestone, device)
    if module_sha256(parent) != module_sha256(old_parent):
        raise RuntimeError('R3 timing changed the declared original parent')
    models.update(old_models)
    choices.update(old_choices)
    if set(models) != set(learned_names(config, r2, original, grid, reference)) or set(choices) != set(models):
        raise RuntimeError('R3 timing omitted a registered learned reference')
    return parent, models, choices


def archive_origin(name):
    return 'r3' if name.startswith('r3__') else r2_archive_origin(name)


def measurement_order(names, frame):
    names = list(names)
    if len(names) != 16 or len(set(names)) != 16 or frame < 0:
        raise ValueError('R3 timing requires sixteen distinct registered models')
    shift = frame % len(names)
    return names[shift:] + names[:shift]


def validate_config(config):
    indices = np.rint(np.linspace(0, 99, 32)).astype(int).tolist()
    if (config['timing_source_indices'] != indices or config['timing_snrs_db'] != [1., 4., 7., 13., 19.] or
        config['timing_noise_seed'] != 2001 or config['timing_models'] != 16 or config['timing_rows'] != 2560):
        raise RuntimeError('R3 timing population, conditions or model count changed')


def validate_rows(rows, config, names):
    validate_config(config)
    measurement_order(names, 0)
    expected = {(index, snr, config['timing_noise_seed'], name) for index in config['timing_source_indices']
        for snr in config['timing_snrs_db'] for name in names}
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise RuntimeError('R3 timing matrix is incomplete or duplicated')
    identifiers = set()
    for position, index in enumerate(config['timing_source_indices']):
        identities = {row['image_id'] for key, row in lookup.items() if key[0] == index}
        if len(identities) != 1 or next(iter(identities)) in identifiers:
            raise RuntimeError('R3 timing source identities changed or duplicated')
        identifiers.update(identities)
        for snr_position, snr in enumerate(config['timing_snrs_db']):
            order = measurement_order(names, position * len(config['timing_snrs_db']) + snr_position)
            for name in names:
                row = lookup[index, snr, config['timing_noise_seed'], name]
                if (int(row['receiver_order_index']) != order.index(name) or int(row['total_complex_uses']) != 3060 or
                    abs(float(row['total_energy']) - 6120) > 1e-5 or
                    row['timing_scope'] != 'separate_matched_uncontended_RX_including_visual_Decoder'):
                    raise RuntimeError('R3 timing order, resources or scope changed')
                if any(not np.isfinite(float(row[key])) or float(row[key]) <= 0 for key in ('online_TX_seconds', 'receiver_seconds')):
                    raise RuntimeError('R3 measured latency must be positive and finite')
    return lookup


def summarize(rows, evaluation, names, base, config):
    lookup = validate_rows(rows, evaluation, names)
    indices = evaluation['timing_source_indices']
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(len(indices),
        size=(base['evaluation']['bootstrap_resamples'], len(indices)))
    method = new_names(config)[0]
    if method not in names or len(names) != 16:
        raise RuntimeError('R3 timing comparison omitted the new method or a reference')
    summaries, paired = [], []
    for snrs in (base['evaluation']['primary_snrs_db'], evaluation['timing_snrs_db'], *[[snr] for snr in evaluation['timing_snrs_db']]):
        label = '+'.join(map(str, snrs))
        values = {}
        for name in names:
            subset = [lookup[index, snr, evaluation['timing_noise_seed'], name] for index in indices for snr in snrs]
            summaries.append({'arm': name, 'snrs_db': label, 'source_images': len(indices), 'noise_seeds_per_source': 1,
                **{key + suffix: float(function([float(row[key]) for row in subset]))
                    for key in ('online_TX_seconds', 'receiver_seconds')
                    for suffix, function in (('_mean', np.mean), ('_median', np.median), ('_p95', lambda values: np.percentile(values, 95)))}})
            values[name] = np.array([np.mean([float(lookup[index, snr, evaluation['timing_noise_seed'], name]['receiver_seconds'])
                for snr in snrs]) for index in indices])
        for control in [name for name in names if name != method]:
            difference = values[method] - values[control]
            low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
            paired.append({'snrs_db': label, 'method': method, 'control': control, 'source_images': len(indices),
                'comparison_scope': comparison_scope(control), 'metric': 'receiver_seconds',
                'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired
