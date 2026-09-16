"""Complete fixed-budget grid quality comparisons, explicitly separate from timing."""

import json

import numpy as np
import torch
import yaml

from grid_controls.common import EXPERIMENT, completed_joint_reference, initial_system, load_parent, output_path, settings
from joint_sender.evaluation import all_names as original_names, fresh_names as original_learned_names
from joint_sender.evaluation_io import digest
from wetok_comm.common import sha256, verify_sources
from wetok_comm.evaluation import raw_noise
from wetok_comm.interface_evaluation import METRICS
from wetok_comm.training import module_sha256


def new_names(config):
    return ['grid__' + name for name in config['variants']]


def all_names(config, original, reference):
    return new_names(config) + original_names(original, reference)


def learned_names(config, original, reference):
    return new_names(config) + original_learned_names(original, reference)


def comparison_pairs(config, original, reference):
    primary = ('joint__multiscale_state_history', 'grid__full_grid_state_history')
    pairs = [primary, ('grid__full_grid_innovation', 'grid__full_grid_state_history')]
    pairs.extend((method, control) for method in new_names(config) for control in original_names(original, reference)
                 if (method, control) != primary[::-1])
    if len(pairs) != 33 or len({frozenset(pair) for pair in pairs}) != 33:
        raise RuntimeError('registered grid contrasts were omitted or duplicated')
    return pairs


def load_evaluation():
    config, original, reference, base, parent = settings()
    evaluation = yaml.safe_load((EXPERIMENT / 'configs/evaluation.yaml').read_text())
    if (evaluation['population'] != 'original_100_development_only' or evaluation['source_images'] != 100 or
        evaluation['required_updates_each'] != 5000 or evaluation['all_snrs_db'] != base['evaluation']['snrs_db'] or
        evaluation['primary_snrs_db'] != base['evaluation']['primary_snrs_db'] or
        len(all_names(config, original, reference)) != evaluation['all_methods'] or evaluation['main_rows'] != 37800 or
        evaluation['comparison_pairs'] != len(comparison_pairs(config, original, reference))):
        raise RuntimeError('registered grid comparison population or physics changed')
    return evaluation, config, original, reference, base, parent


def audited_grid(config, original, reference, step):
    if step != 5000:
        raise ValueError('grid development requires equal5000 update opportunities')
    references = completed_joint_reference(config, original, reference)
    root = output_path(config, 'training')
    path = root / 'milestones/step_0005000.json'
    record = json.loads(path.read_text())
    if (record['status'] != 'JOINT_GRID_MILESTONE_COMPLETE' or record['grid_updates_per_arm'] != step or
        record['reference_hashes'] != references['hashes'] or sha256(record['checkpoint']) != record['checkpoint_sha256']):
        raise RuntimeError('grid milestone or frozen references changed')
    verify_sources(record['source_hashes'])
    review_path = output_path(config, 'analysis') / 'calibration_0005000/completion.json'
    review = json.loads(review_path.read_text())
    if (review['actual_E_R_Adam_data_noise_power_selection_audit'] != 'PASS' or review['grid_updates'] != step or
        review['milestone_sha256'] != sha256(path)):
        raise RuntimeError('grid selection has not passed its completed audit')
    verify_sources(review['source_hashes'])
    for relative, expected in review['output_hashes'].items():
        if sha256(review_path.parent / relative) != expected:
            raise RuntimeError('grid calibration review artifact changed')
    return record, path, review_path, references


def model_registry(config, reference, base, milestone, device):
    parent = load_parent(reference, base, device)
    models, choices = {}, {}
    for variant in config['variants']:
        choice = milestone['selected'][variant]
        path = output_path(config, 'training') / choice['checkpoint']
        if sha256(path) != choice['checkpoint_sha256']:
            raise RuntimeError('selected grid checkpoint changed')
        stored = torch.load(path, map_location='cpu', weights_only=True)
        if stored['variant'] != variant or stored['step'] != choice['step'] or stored['encoder_trained'] is not True:
            raise RuntimeError('selected grid policy or step differs')
        model = initial_system(parent, variant, reference, device).eval()
        model.load_state_dict(stored['model'], strict=True)
        name = 'grid__' + variant
        models[name] = model
        choices[name] = {**choice, 'absolute_checkpoint': str(path), 'encoder_sha256': module_sha256(model.encoder),
            'communication_parameters': sum(parameter.numel() for parameter in model.parameters()),
            'optimized_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            'selected_encoder_differs_from_parent': module_sha256(model.encoder) != module_sha256(parent.encoder)}
    return parent, models, choices


def validate_rows(rows, config, original, reference, base, source_count=100):
    names = all_names(config, original, reference)
    snrs, seeds = base['evaluation']['snrs_db'], base['evaluation']['noise_seeds']
    expected = {(index, float(snr), int(seed), name) for index in range(source_count) for snr in snrs for seed in seeds for name in names}
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise RuntimeError('grid quality source/noise/SNR/method matrix is incomplete or duplicated')
    source_ids = set()
    for index in range(source_count):
        identifiers = {row['image_id'] for key, row in lookup.items() if key[0] == index}
        if len(identifiers) != 1:
            raise RuntimeError('grid methods do not use the same source')
        identifier = next(iter(identifiers))
        if identifier in source_ids:
            raise RuntimeError('a source image was duplicated under a different index')
        source_ids.add(identifier)
        for snr in snrs:
            for seed in seeds:
                noise_hash = digest(raw_noise(identifier, seed))
                for name in names:
                    row = lookup[index, float(snr), int(seed), name]
                    if (int(row['total_complex_uses']) != 3060 or abs(float(row['total_energy']) - 6120) > 1e-5 or
                        row['noise_sha256'] != noise_hash or not all(np.isfinite(float(row[metric])) for metric in METRICS)):
                        raise RuntimeError('grid physical resource/noise/metric ledger changed')
                    if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
                        raise RuntimeError('severe distortion rule changed')
                    if row.get('online_TX_seconds', '') != '' or row.get('receiver_seconds', '') != '':
                        raise RuntimeError('quality phase must not mix measured or shared GPU latency')
                    if name.startswith('grid__'):
                        if (row['quality_origin'] != 'new_grid_measurement' or int(row['available_updates']) != 5000 or
                            int(row['parent_updates']) != 7000 or int(row['header_uses']) != 0 or int(row['data_uses']) != 3060 or
                            int(row['raw_bits']) != 8192 or row['decoder_interface'] != 'continuous_mean'):
                            raise RuntimeError('new grid input/training/interface conditions changed')
                        for key in ('bit_error_rate', 'feature_mse', 'feature_abs_mean', 'feature_saturation_fraction'):
                            if not np.isfinite(float(row[key])):
                                raise RuntimeError('grid feature diagnostics are nonfinite')
            for name in new_names(config):
                if len({lookup[index, float(snr), int(seed), name]['transmitted_sha256'] for seed in seeds}) != 1:
                    raise RuntimeError('a Grid transmitter depended on its receiver noise seed')
    return lookup, names


def validate_selection(rows, milestone):
    for row in rows:
        if not row['arm'].startswith('grid__'):
            continue
        variant = row['arm'].split('__', 1)[1]
        selected = milestone['selected'][variant]
        if (int(row['selected_step']) != selected['step'] or int(row['selected_global_data_step']) != 7000 + selected['step'] or
            row['checkpoint_sha256'] != selected['checkpoint_sha256']):
            raise RuntimeError('quality evaluation changed its full-calibration checkpoint selection')


def validate_reference_diagnostics(noiseless, support, references):
    from .references import project_reference
    expected_noiseless = {(int(row['image_index']), row['arm']): project_reference(row, references.receipt_sha)
                          for row in references.noiseless}
    expected_support = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']):
                        project_reference(row, references.receipt_sha) for row in references.support}
    for row in noiseless:
        if row['arm'].startswith('grid__'):
            continue
        expected = expected_noiseless[int(row['image_index']), row['arm']]
        if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
            raise RuntimeError('a frozen noiseless reference field changed')
    for row in support:
        expected = expected_support[int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']]
        if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
            raise RuntimeError('a frozen fixed-support reference field changed')


def validate_diagnostics(rows, clean, noiseless, support, config, original, reference, base, source_count=100):
    anchors = {int(row['image_index']): row for row in clean}
    if len(clean) != source_count or set(anchors) != set(range(source_count)):
        raise RuntimeError('native quality anchors are incomplete or duplicated')
    expected_noiseless = {(index, name) for index in range(source_count) for name in learned_names(config, original, reference)}
    if len(noiseless) != len(expected_noiseless) or {(int(row['image_index']), row['arm']) for row in noiseless} != expected_noiseless:
        raise RuntimeError('a learned noiseless control was omitted')
    from wetok_comm.deep_support import SUPPORT_NAME
    expected_support = {(index, snr, int(seed), SUPPORT_NAME) for index in range(source_count) for snr in (5., 6.) for seed in base['evaluation']['noise_seeds']}
    if len(support) != len(expected_support) or {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']) for row in support} != expected_support:
        raise RuntimeError('fixed-support Deep controls are incomplete')
    for row in rows + noiseless + support:
        anchor = anchors[int(row['image_index'])]
        if row['image_id'] != anchor['image_id'] or abs(float(row['LPIPS_excess_from_native']) - (float(row['lpips']) - float(anchor['lpips']))) > 1e-7:
            raise RuntimeError('quality rows use a different source or native excess anchor')
        if not all(np.isfinite(float(row[metric])) for metric in METRICS):
            raise RuntimeError('nonfinite quality or diagnostic outcome')
        if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
            raise RuntimeError('diagnostic severe-distortion rule differs')


def support_statistics(rows, support, config, original, reference, base, source_count=100):
    from wetok_comm.deep_support import SUPPORT_NAME
    names = learned_names(config, original, reference) + [SUPPORT_NAME]
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows + support}
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(source_count,
        size=(base['evaluation']['bootstrap_resamples'], source_count))
    summaries, paired = [], []
    for snrs in ([5.], [6.], [5., 6.]):
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, snr, int(seed), name][metric]) for snr in snrs
            for seed in base['evaluation']['noise_seeds']]) for index in range(source_count)]) for metric in METRICS} for name in names}
        summaries.extend({'arm': name, 'snrs_db': label, 'source_images': source_count,
            **{metric: float(array.mean()) for metric, array in outcome.items()}} for name, outcome in values.items())
        for name in names[:-1]:
            for metric in METRICS:
                difference = values[name][metric] - values[SUPPORT_NAME][metric]
                low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
                paired.append({'snrs_db': label, 'method': name, 'control': SUPPORT_NAME, 'metric': metric,
                    'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired


def statistics(rows, config, original, reference, base, source_count=100):
    lookup, names = validate_rows(rows, config, original, reference, base, source_count)
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(source_count,
        size=(base['evaluation']['bootstrap_resamples'], source_count))
    summaries, paired = [], []
    for snrs in (base['evaluation']['primary_snrs_db'], *[[snr] for snr in base['evaluation']['snrs_db']]):
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, float(snr), int(seed), name][metric]) for snr in snrs
            for seed in base['evaluation']['noise_seeds']]) for index in range(source_count)]) for metric in METRICS} for name in names}
        summaries.extend({'arm': name, 'snrs_db': label, 'source_images': source_count,
            **{metric: float(array.mean()) for metric, array in outcomes.items()}} for name, outcomes in values.items())
        for method, control in comparison_pairs(config, original, reference):
            for metric in METRICS:
                difference = values[method][metric] - values[control][metric]
                low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
                paired.append({'snrs_db': label, 'method': method, 'control': control, 'metric': metric,
                    'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired
