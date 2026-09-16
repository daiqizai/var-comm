"""Final R3 quality contracts and explicitly scoped comparisons against all twenty-two frozen references."""

import json

import numpy as np
import torch
import yaml

from joint_sender.evaluation_io import digest
from sufficiency.evaluation import all_names as r2_names, learned_names as r2_learned_names
from vector_control.common import EXPERIMENT, initial_system, load_parent, output_path, qualified_reference, settings
from wetok_comm.common import PROJECT, sha256, verify_sources
from wetok_comm.evaluation import raw_noise
from wetok_comm.interface_evaluation import METRICS
from wetok_comm.training import module_sha256


def new_names(config):
    return ['r3__' + config['variant']]


def all_names(config, r2, original, grid, reference):
    return new_names(config) + r2_names(r2, original, grid, reference)


def learned_names(config, r2, original, grid, reference):
    return new_names(config) + r2_learned_names(r2, original, grid, reference)


def comparison_scope(control):
    if control == 'r2__full_grid_innovation':
        return 'matched10000_vector_construction_parameter_and_E_call_comparison'
    if control.startswith('r2__'):
        return 'same10000_structure_reference_not_equal_parameters_or_E_calls'
    return 'historical_or_system_reference_not_equal10000_training'


def load_evaluation():
    config, r2, original, grid, reference, base, parent = settings()
    evaluation = yaml.safe_load((EXPERIMENT / 'configs/evaluation.yaml').read_text())
    if (evaluation['population'] != 'original_100_development_only' or evaluation['source_images'] != 100 or
        evaluation['required_total_updates'] != 10000 or evaluation['new_method'] != new_names(config)[0] or
        evaluation['all_snrs_db'] != base['evaluation']['snrs_db'] or evaluation['primary_snrs_db'] != base['evaluation']['primary_snrs_db'] or
        len(all_names(config, r2, original, grid, reference)) != evaluation['all_methods'] or evaluation['all_methods'] != 23 or
        evaluation['main_rows'] != 48300 or evaluation['new_main_rows'] != 2100 or evaluation['comparison_pairs'] != 22 or
        evaluation['matched_vector_control'] != 'r2__full_grid_innovation'):
        raise RuntimeError('R3 evaluation population, methods, budget opportunities or comparison changed')
    return evaluation, config, r2, original, grid, reference, base, parent


def evaluation_output(evaluation, kind):
    path = (PROJECT / 'outputs' / evaluation['outputs'][kind]).resolve()
    if not path.is_relative_to(PROJECT / 'outputs'):
        raise ValueError('R3 evaluation output escaped project outputs')
    return path


def audited_endpoint(config, r2, grid, step):
    if step != 10000:
        raise ValueError('R3 development requires the complete matched10000 opportunity')
    qualified = qualified_reference(config, r2, grid)
    root = output_path(config, 'training')
    path = root / 'milestones/step_0010000.json'
    milestone = json.loads(path.read_text())
    if (milestone['status'] != 'R3_VECTOR_CONTROL_MILESTONE_COMPLETE' or milestone['completed_updates'] != step or
        milestone['global_data_step'] != 17000 or milestone['reference_bindings'] != qualified['bindings'] or
        sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']):
        raise RuntimeError('R3 actual model/optimizer/data endpoint is incomplete or changed')
    verify_sources(milestone['source_hashes'])
    review_path = output_path(config, 'analysis') / 'calibration_0010000/completion.json'
    review = json.loads(review_path.read_text())
    if (review['status'] != 'R3_CALIBRATION_AND_HISTORY_AUDIT_PASS_NOT_RESEARCH_COMPLETE' or
        review['fresh_origin_actual_Adam_new_data_power_full_selection_audit'] != 'PASS' or
        review['completed_updates'] != step or review['milestone_sha256'] != sha256(path)):
        raise RuntimeError('R3 complete-history/selection audit is incomplete')
    verify_sources(review['source_hashes'])
    for relative, expected in review['output_hashes'].items():
        if sha256(review_path.parent / relative) != expected:
            raise RuntimeError('R3 completed calibration review changed')
    return milestone, path, review_path, qualified


def model_registry(config, reference, base, milestone, device):
    parent = load_parent(reference, base, device)
    choice = milestone['selected']
    if sha256(choice['checkpoint']) != choice['checkpoint_sha256']:
        raise RuntimeError('selected R3 checkpoint changed')
    saved = torch.load(choice['checkpoint'], map_location='cpu', weights_only=True)
    if saved['variant'] != config['variant'] or saved['step'] != choice['step']:
        raise RuntimeError('R3 selection points to a different model or update')
    model = initial_system(parent, reference, device).eval()
    model.load_state_dict(saved['model'], strict=True)
    name = new_names(config)[0]
    selection = {**choice, 'encoder_sha256': module_sha256(model.encoder),
        'communication_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'optimized_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'selected_encoder_differs_from_parent': module_sha256(model.encoder) != module_sha256(parent.encoder)}
    return parent, {name: model}, {name: selection}


def validate_rows(rows, config, r2, original, grid, reference, base, source_count=100):
    names = all_names(config, r2, original, grid, reference)
    snrs, seeds = base['evaluation']['snrs_db'], base['evaluation']['noise_seeds']
    expected = {(index, float(snr), int(seed), name) for index in range(source_count) for snr in snrs for seed in seeds for name in names}
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise RuntimeError('R3 full source/SNR/noise/method matrix is incomplete or duplicated')
    identifiers = set()
    for index in range(source_count):
        identity = {row['image_id'] for key, row in lookup.items() if key[0] == index}
        if len(identity) != 1 or next(iter(identity)) in identifiers:
            raise RuntimeError('R3 sources were substituted or duplicated')
        identifier = next(iter(identity))
        identifiers.add(identifier)
        for snr in snrs:
            for seed in seeds:
                noise_hash = digest(raw_noise(identifier, seed))
                for name in names:
                    row = lookup[index, float(snr), int(seed), name]
                    if (int(row['total_complex_uses']) != 3060 or abs(float(row['total_energy']) - 6120) > 1e-5 or
                        row['noise_sha256'] != noise_hash or not all(np.isfinite(float(row[key])) for key in METRICS)):
                        raise RuntimeError('R3 physical/noise/quality ledger changed')
                    if row.get('receiver_seconds', '') != '' or row.get('online_TX_seconds', '') != '':
                        raise RuntimeError('R3 quality must not contain current latency')
                    if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
                        raise RuntimeError('R3 severe-distortion rule changed')
                    if name.startswith('r3__') and (int(row['available_updates']) != 10000 or int(row['parent_updates']) != 7000 or
                        int(row['new_update_opportunity']) != 10000 or row['r3_quality_origin'] != 'new_r3_measurement' or
                        int(row['header_uses']) != 0 or int(row['data_uses']) != 3060 or int(row['raw_bits']) != 8192 or
                        row['decoder_interface'] != 'continuous_mean' or int(row['communication_parameters']) != config['expected_communication_parameters']):
                        raise RuntimeError('R3 initial history, parameter, input or interface conditions changed')
            if len({lookup[index, float(snr), int(seed), new_names(config)[0]]['transmitted_sha256'] for seed in seeds}) != 1:
                raise RuntimeError('R3 transmitter depended on receiver noise')
    return lookup, names


def validate_selection(rows, milestone):
    selected = milestone['selected']
    for row in rows:
        if row['arm'].startswith('r3__') and (int(row['selected_step']) != selected['step'] or
            row['checkpoint_sha256'] != selected['checkpoint_sha256'] or int(row['selected_global_data_step']) != 7000 + selected['step']):
            raise RuntimeError('R3 used a different model than complete-calibration selection')


def validate_diagnostics(rows, clean, noiseless, support, config, r2, original, grid, reference, base, source_count=100):
    from wetok_comm.deep_support import SUPPORT_NAME
    anchors = {int(row['image_index']): row for row in clean}
    if len(clean) != source_count or set(anchors) != set(range(source_count)):
        raise RuntimeError('R3 native anchors are incomplete')
    expected = {(index, name) for index in range(source_count) for name in learned_names(config, r2, original, grid, reference)}
    if len(noiseless) != len(expected) or {(int(row['image_index']), row['arm']) for row in noiseless} != expected:
        raise RuntimeError('R3 noiseless learned-model matrix is incomplete')
    expected_support = {(index, snr, int(seed), SUPPORT_NAME) for index in range(source_count) for snr in (5., 6.) for seed in base['evaluation']['noise_seeds']}
    if len(support) != len(expected_support) or {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']) for row in support} != expected_support:
        raise RuntimeError('R3 fixed-support Deep matrix is incomplete')
    for row in rows + noiseless + support:
        anchor = anchors[int(row['image_index'])]
        if row['image_id'] != anchor['image_id'] or abs(float(row['LPIPS_excess_from_native']) - (float(row['lpips']) - float(anchor['lpips']))) > 1e-7:
            raise RuntimeError('R3 diagnostic source or native quality anchor changed')
        if not all(np.isfinite(float(row[metric])) for metric in METRICS):
            raise RuntimeError('nonfinite R3 outcome')
        if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
            raise RuntimeError('R3 diagnostic severe-distortion rule changed')


def validate_reference_diagnostics(noiseless, support, references):
    from .references import project_reference
    old_noiseless = {(int(row['image_index']), row['arm']): project_reference(row, references.receipt_sha) for row in references.noiseless}
    old_support = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): project_reference(row, references.receipt_sha) for row in references.support}
    for row in noiseless:
        if not row['arm'].startswith('r3__'):
            expected = old_noiseless[int(row['image_index']), row['arm']]
            if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
                raise RuntimeError('R3 changed a frozen noiseless reference')
    for row in support:
        expected = old_support[int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']]
        if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
            raise RuntimeError('R3 changed a frozen fixed-support outcome')


def statistics(rows, config, r2, original, grid, reference, base, source_count=100):
    lookup, names = validate_rows(rows, config, r2, original, grid, reference, base, source_count)
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(source_count,
        size=(base['evaluation']['bootstrap_resamples'], source_count))
    summaries, paired = [], []
    method = new_names(config)[0]
    for snrs in (base['evaluation']['primary_snrs_db'], *[[snr] for snr in base['evaluation']['snrs_db']]):
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, float(snr), int(seed), name][metric]) for snr in snrs
            for seed in base['evaluation']['noise_seeds']]) for index in range(source_count)]) for metric in METRICS} for name in names}
        summaries.extend({'arm': name, 'snrs_db': label, 'source_images': source_count,
            **{metric: float(array.mean()) for metric, array in values[name].items()}} for name in names)
        for control in names[1:]:
            for metric in METRICS:
                difference = values[method][metric] - values[control][metric]
                low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
                paired.append({'snrs_db': label, 'method': method, 'control': control, 'metric': metric,
                    'comparison_scope': comparison_scope(control), 'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired


def support_statistics(rows, support, config, r2, original, grid, reference, base, source_count=100):
    from wetok_comm.deep_support import SUPPORT_NAME
    names = learned_names(config, r2, original, grid, reference) + [SUPPORT_NAME]
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
                    'measurement_scope': 'new_R3_comparison' if name.startswith('r3__') else 'unchanged_reference_comparison',
                    'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired
