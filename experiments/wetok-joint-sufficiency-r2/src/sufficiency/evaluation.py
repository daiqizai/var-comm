"""R2 quality contracts: matched10000 models versus one another, older trials explicitly historical."""

import json

import numpy as np
import torch
import yaml

from grid_controls.common import initial_system as grid_system
from grid_controls.evaluation import all_names as grid_reference_names, learned_names as grid_learned_names
from joint_sender.common import initial_system as basic_system
from joint_sender.evaluation_io import digest
from sufficiency.common import EXPERIMENT, load_parent, load_sources, output_path, settings
from wetok_comm.common import PROJECT, sha256, verify_sources
from wetok_comm.evaluation import raw_noise
from wetok_comm.interface_evaluation import METRICS
from wetok_comm.training import module_sha256


def new_names(config):
    return ['r2__' + name for name in config['variants']]


def all_names(config, original, grid, reference):
    return new_names(config) + grid_reference_names(grid, original, reference)


def learned_names(config, original, grid, reference):
    return new_names(config) + grid_learned_names(grid, original, reference)


def previous_name(variant):
    return ('joint__' if variant in ('single_pass', 'multiscale_state_history') else 'grid__') + variant


def comparison_pairs(config, original, grid, reference):
    pairs = [(method, control) for method in new_names(config) for control in grid_reference_names(grid, original, reference)]
    pairs.extend(('r2__' + first, 'r2__' + second) for first, second in (
        ('multiscale_state_history', 'full_grid_state_history'), ('multiscale_state_history', 'single_pass'),
        ('full_grid_state_history', 'single_pass'), ('full_grid_innovation', 'single_pass'),
        ('full_grid_innovation', 'multiscale_state_history'), ('full_grid_innovation', 'full_grid_state_history')))
    if len(pairs) != 78 or len({frozenset(pair) for pair in pairs}) != 78:
        raise RuntimeError('R2 comparisons were duplicated or omitted')
    return pairs


def load_evaluation():
    config, original, grid, reference, base, parent = settings()
    evaluation = yaml.safe_load((EXPERIMENT / 'configs/evaluation.yaml').read_text())
    if (evaluation['source_images'] != 100 or evaluation['population'] != 'original_100_development_only' or
        evaluation['required_total_updates'] != 10000 or evaluation['all_snrs_db'] != base['evaluation']['snrs_db'] or
        evaluation['primary_snrs_db'] != base['evaluation']['primary_snrs_db'] or evaluation['all_methods'] != 22 or
        len(all_names(config, original, grid, reference)) != 22 or evaluation['main_rows'] != 46200 or
        evaluation['comparison_pairs'] != len(comparison_pairs(config, original, grid, reference))):
        raise RuntimeError('R2 quality population, update opportunities or comparison grid changed')
    return evaluation, config, original, grid, reference, base, parent


def evaluation_output(evaluation, kind):
    path = (PROJECT / 'outputs' / evaluation['outputs'][kind]).resolve()
    if not path.is_relative_to(PROJECT / 'outputs'):
        raise ValueError('R2 evaluation output escaped project outputs')
    return path


def audited_endpoint(config, step):
    if step != 10000:
        raise ValueError('this R2 evaluation requires all four10000 opportunities')
    sources = load_sources(config)
    root = output_path(config, 'training')
    path = root / 'milestones' / f'step_{step:07d}.json'
    milestone = json.loads(path.read_text())
    if (milestone['status'] != 'SUFFICIENCY_MILESTONE_COMPLETE' or milestone['total_updates_per_arm'] != step or
        milestone['new_updates_per_arm'] != 5000 or milestone['global_data_step'] != 17000 or
        milestone['optimizer_reset'] or milestone['source_endpoint_hashes'] != sources['hashes'] or
        sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']):
        raise RuntimeError('R2 actual continuation endpoint is incomplete or changed')
    verify_sources(milestone['source_hashes'])
    review_path = output_path(config, 'analysis') / f'calibration_{step:07d}/completion.json'
    review = json.loads(review_path.read_text())
    if (review['status'] != 'R2_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE' or
        review['model_Adam_inheritance_new_data_power_selection_audit'] != 'PASS' or review['total_updates'] != step or
        review['milestone_sha256'] != sha256(path)):
        raise RuntimeError('R2 selection/optimizer/data audit is incomplete')
    verify_sources(review['source_hashes'])
    for relative, expected in review['output_hashes'].items():
        if sha256(review_path.parent / relative) != expected:
            raise RuntimeError('R2 completed calibration review changed')
    return milestone, path, review_path, sources


def model_registry(config, reference, base, milestone, device):
    parent = load_parent(reference, base, device)
    models, choices = {}, {}
    for variant in config['variants']:
        choice = milestone['selected'][variant]
        path = choice['checkpoint']
        if sha256(path) != choice['checkpoint_sha256']:
            raise RuntimeError('selected R2 checkpoint changed')
        stored = torch.load(path, map_location='cpu', weights_only=True)
        if stored['variant'] != variant or stored['step'] != choice['step']:
            raise RuntimeError('R2 checkpoint variant/step differs from full-calibration selection')
        factory = basic_system if config['origins'][variant] == 'joint' else grid_system
        model = factory(parent, variant, reference, device).eval()
        model.load_state_dict(stored['model'], strict=True)
        name = 'r2__' + variant
        models[name] = model
        choices[name] = {**choice, 'encoder_sha256': module_sha256(model.encoder),
            'communication_parameters': sum(parameter.numel() for parameter in model.parameters()),
            'optimized_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            'selected_encoder_differs_from_parent': module_sha256(model.encoder) != module_sha256(parent.encoder)}
    return parent, models, choices


def validate_rows(rows, config, original, grid, reference, base, source_count=100):
    names = all_names(config, original, grid, reference)
    snrs, seeds = base['evaluation']['snrs_db'], base['evaluation']['noise_seeds']
    expected = {(index, float(snr), int(seed), name) for index in range(source_count) for snr in snrs for seed in seeds for name in names}
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise RuntimeError('R2 full source/SNR/noise/method matrix is incomplete')
    identifiers = set()
    for index in range(source_count):
        identity = {row['image_id'] for key, row in lookup.items() if key[0] == index}
        if len(identity) != 1 or next(iter(identity)) in identifiers:
            raise RuntimeError('R2 sources were substituted or duplicated')
        identifier = next(iter(identity))
        identifiers.add(identifier)
        for snr in snrs:
            for seed in seeds:
                noise_hash = digest(raw_noise(identifier, seed))
                for name in names:
                    row = lookup[index, float(snr), int(seed), name]
                    if (int(row['total_complex_uses']) != 3060 or abs(float(row['total_energy']) - 6120) > 1e-5 or
                        row['noise_sha256'] != noise_hash or not all(np.isfinite(float(row[key])) for key in METRICS)):
                        raise RuntimeError('R2 physical/noise/quality ledger changed')
                    if row.get('receiver_seconds', '') != '' or row.get('online_TX_seconds', '') != '':
                        raise RuntimeError('R2 quality phase cannot contain current latency measurements')
                    if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
                        raise RuntimeError('R2 severe-distortion rule changed')
                    if name.startswith('r2__') and (int(row['available_updates']) != 10000 or int(row['parent_updates']) != 7000 or
                        int(row['new_update_opportunity']) != 5000 or row['r2_quality_origin'] != 'new_r2_measurement' or
                        int(row['header_uses']) != 0 or int(row['data_uses']) != 3060 or int(row['raw_bits']) != 8192 or
                        row['decoder_interface'] != 'continuous_mean'):
                        raise RuntimeError('R2 update, input or interface conditions changed')
            for name in new_names(config):
                if len({lookup[index, float(snr), int(seed), name]['transmitted_sha256'] for seed in seeds}) != 1:
                    raise RuntimeError('R2 transmitter used receiver noise information')
    return lookup, names


def validate_selection(rows, milestone):
    for row in rows:
        if row['arm'].startswith('r2__'):
            selected = milestone['selected'][row['arm'].split('__', 1)[1]]
            if (int(row['selected_step']) != selected['step'] or row['checkpoint_sha256'] != selected['checkpoint_sha256'] or
                int(row['selected_global_data_step']) != 7000 + selected['step']):
                raise RuntimeError('R2 quality used a different checkpoint selection')


def validate_diagnostics(rows, clean, noiseless, support, config, original, grid, reference, base, source_count=100):
    from wetok_comm.deep_support import SUPPORT_NAME
    anchors = {int(row['image_index']): row for row in clean}
    if len(clean) != source_count or set(anchors) != set(range(source_count)):
        raise RuntimeError('R2 native anchors are incomplete')
    expected = {(index, name) for index in range(source_count) for name in learned_names(config, original, grid, reference)}
    if len(noiseless) != len(expected) or {(int(row['image_index']), row['arm']) for row in noiseless} != expected:
        raise RuntimeError('R2 noiseless learned-model matrix is incomplete')
    expected_support = {(index, snr, int(seed), SUPPORT_NAME) for index in range(source_count) for snr in (5., 6.) for seed in base['evaluation']['noise_seeds']}
    if len(support) != len(expected_support) or {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']) for row in support} != expected_support:
        raise RuntimeError('R2 fixed-support Deep matrix is incomplete')
    for row in rows + noiseless + support:
        anchor = anchors[int(row['image_index'])]
        if row['image_id'] != anchor['image_id'] or abs(float(row['LPIPS_excess_from_native']) - (float(row['lpips']) - float(anchor['lpips']))) > 1e-7:
            raise RuntimeError('R2 quality source or native anchor changed')
        if not all(np.isfinite(float(row[metric])) for metric in METRICS):
            raise RuntimeError('nonfinite R2 quality/diagnostic outcome')
        if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
            raise RuntimeError('R2 diagnostic severe distortion rule changed')


def validate_reference_diagnostics(noiseless, support, references):
    from .references import project_reference
    old_noiseless = {(int(row['image_index']), row['arm']): project_reference(row, references.receipt_sha) for row in references.noiseless}
    old_support = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): project_reference(row, references.receipt_sha) for row in references.support}
    for row in noiseless:
        if not row['arm'].startswith('r2__'):
            expected = old_noiseless[int(row['image_index']), row['arm']]
            if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
                raise RuntimeError('R2 modified a historical noiseless outcome')
    for row in support:
        expected = old_support[int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']]
        if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
            raise RuntimeError('R2 modified a historical fixed-support outcome')


def support_statistics(rows, support, config, original, grid, reference, base, source_count=100):
    from wetok_comm.deep_support import SUPPORT_NAME
    names = learned_names(config, original, grid, reference) + [SUPPORT_NAME]
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows + support}
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(source_count,
        size=(base['evaluation']['bootstrap_resamples'], source_count))
    summaries, paired = [], []
    for snrs in ([5.], [6.], [5., 6.]):
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, snr, int(seed), name][metric]) for snr in snrs
            for seed in base['evaluation']['noise_seeds']]) for index in range(source_count)]) for metric in METRICS} for name in names}
        summaries.extend({'arm': name, 'snrs_db': label, 'source_images': source_count,
            **{metric: float(array.mean()) for metric, array in values[name].items()}} for name in names)
        for name in names[:-1]:
            for metric in METRICS:
                difference = values[name][metric] - values[SUPPORT_NAME][metric]
                low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
                paired.append({'snrs_db': label, 'method': name, 'control': SUPPORT_NAME, 'metric': metric,
                    'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired


def statistics(rows, config, original, grid, reference, base, source_count=100):
    lookup, names = validate_rows(rows, config, original, grid, reference, base, source_count)
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(source_count,
        size=(base['evaluation']['bootstrap_resamples'], source_count))
    summaries, paired = [], []
    for snrs in (base['evaluation']['primary_snrs_db'], *[[snr] for snr in base['evaluation']['snrs_db']]):
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, float(snr), int(seed), name][metric]) for snr in snrs
            for seed in base['evaluation']['noise_seeds']]) for index in range(source_count)]) for metric in METRICS} for name in names}
        summaries.extend({'arm': name, 'snrs_db': label, 'source_images': source_count,
            **{metric: float(array.mean()) for metric, array in values[name].items()}} for name in names)
        for method, control in comparison_pairs(config, original, grid, reference):
            for metric in METRICS:
                difference = values[method][metric] - values[control][metric]
                low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
                if control.startswith('r2__'):
                    scope = 'matched10000_structure_comparison'
                elif control == previous_name(method.split('__', 1)[1]):
                    scope = 'extra_training_effect_not_new_mechanism'
                else:
                    scope = 'historical_or_system_reference_not_equal10000_training'
                paired.append({'snrs_db': label, 'method': method, 'control': control, 'metric': metric,
                    'comparison_scope': scope, 'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired
