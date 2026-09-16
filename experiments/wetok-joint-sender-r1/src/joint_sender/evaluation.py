"""Matched-opportunity Joint versus all six frozen-sender receivers, not a same-y claim."""

import json

import numpy as np
import torch
import yaml

from .common import EXPERIMENT, complete_control, initial_system, output_path, settings
from innovation_comm.evaluation import model_registry as frozen_registry, reference_names as system_names, validate_population
from wetok_comm.common import PROJECT, sha256, verify_sources
from wetok_comm.interface_evaluation import METRICS
from wetok_comm.training import module_sha256


def joint_name(variant):
    return 'joint__' + variant


def frozen_name(variant):
    return 'frozen__' + variant


def fresh_names(config, reference):
    return [joint_name(variant) for variant in config['variants']] + [frozen_name(variant) for variant in reference['variants']]


def all_names(config, reference):
    return fresh_names(config, reference) + system_names()


def comparison_pairs(config, reference):
    pairs = [(joint_name(variant), frozen_name(control)) for variant in config['variants'] for control in reference['variants']]
    pairs.extend((joint_name(variant), control) for variant in config['variants'] for control in system_names())
    pairs.extend((joint_name(method), joint_name(control)) for method, control in history_contrasts())
    if len(pairs) != 42 or len(set(pairs)) != 42:
        raise RuntimeError('registered Joint contrasts were added, duplicated or omitted')
    return pairs


def history_contrasts():
    return [('multiscale_state_history', 'multiscale_no_history'), ('multiscale_state_history', 'single_pass'),
            ('multiscale_no_history', 'single_pass')]


def load_evaluation():
    config, reference, base, parent = settings()
    evaluation = yaml.safe_load((EXPERIMENT / 'configs/evaluation.yaml').read_text())
    if (evaluation['required_updates_each'] != 5000 or evaluation['all_snrs_db'] != base['evaluation']['snrs_db'] or
        evaluation['primary_snrs_db'] != base['evaluation']['primary_snrs_db'] or evaluation['population'] != 'original_100_development_only' or
        evaluation['fresh_models'] != len(fresh_names(config, reference)) or evaluation['all_methods'] != len(all_names(config, reference)) or
        evaluation['main_rows'] != 33600 or evaluation['support_rows'] != 600):
        raise RuntimeError('Joint evaluation population, physical comparison or method grid changed')
    return evaluation, config, reference, base, parent


def matched_milestone(config, reference, step):
    if step != 5000:
        raise ValueError('final Joint development comparison requires 5000 updates of opportunity on both sides')
    control, control_path, control_review = complete_control(config, reference)
    training = output_path(config, 'training')
    metadata = json.loads((training / 'metadata.json').read_text())
    for path, expected in metadata['control_input_hashes'].items():
        if sha256(path) != expected:
            raise RuntimeError('sealed frozen control inputs changed')
    path = training / 'milestones' / f'step_{step:07d}.json'
    milestone = json.loads(path.read_text())
    if milestone['status'] != 'JOINT_SENDER_MILESTONE_COMPLETE' or milestone['joint_updates_per_arm'] != step:
        raise RuntimeError('Joint training opportunity is incomplete')
    verify_sources(milestone['source_hashes'])
    if sha256(milestone['checkpoint']) != milestone['checkpoint_sha256'] or milestone['control_milestone_sha256'] != sha256(control_path):
        raise RuntimeError('Joint endpoint or matched control changed')
    review_path = output_path(config, 'analysis') / f'calibration_{step:07d}/completion.json'
    review = json.loads(review_path.read_text())
    if (review['actual_E_R_Adam_data_noise_power_selection_audit'] != 'PASS' or review['milestone_sha256'] != sha256(path) or
        review['joint_updates'] != step or review['matched_opportunity_control_milestone_sha256'] != sha256(control_path)):
        raise RuntimeError('Joint matched-opportunity calibration audit has not passed')
    verify_sources(review['source_hashes'])
    for relative, expected in review['output_hashes'].items():
        if sha256(review_path.parent / relative) != expected:
            raise RuntimeError('Joint calibration audit artifact changed')
    return milestone, path, review_path, control, control_path, control_review


def model_registry(config, reference, base, milestone, control, device):
    parent, receivers, receiver_choices = frozen_registry(reference, base, control, device)
    networks, choices = {}, {}
    for variant in config['variants']:
        choice = milestone['selected'][variant]
        path = (output_path(config, 'training') / choice['checkpoint']).resolve()
        if (not path.is_relative_to(output_path(config, 'training')) or choice['scope'] != 'full' or choice['source_images'] != 1000 or
            choice['step'] not in config['full_calibration_steps'] or choice['step'] > 5000 or sha256(path) != choice['checkpoint_sha256']):
            raise RuntimeError('Joint checkpoint is not a complete-calibration eligible selection')
        stored = torch.load(path, map_location='cpu', weights_only=True)
        if stored['variant'] != variant or stored['step'] != choice['step'] or stored['encoder_trained'] is not True:
            raise RuntimeError('selected Joint policy/variant/step differs')
        model = initial_system(parent, variant, reference, device)
        model.load_state_dict(stored['model'], strict=True)
        name = joint_name(variant)
        networks[name] = model.eval()
        choices[name] = {**choice, 'absolute_checkpoint': str(path), 'training_policy': 'joint_E_R',
            'optimized_parameters': sum(value.numel() for value in model.parameters() if value.requires_grad)}
    for variant in reference['variants']:
        old = 'receiver__' + variant
        name = frozen_name(variant)
        networks[name] = receivers[old]
        choices[name] = {**receiver_choices[old], 'training_policy': 'receiver_only',
                         'optimized_parameters': receiver_choices[old]['receiver_trainable_parameters']}
    for name, model in networks.items():
        choices[name]['encoder_sha256'] = module_sha256(model.encoder)
        choices[name]['selected_encoder_differs_from_parent'] = choices[name]['encoder_sha256'] != module_sha256(parent.encoder)
    return parent, networks, choices


def measurement_order(names, frame_index):
    names = list(names)
    if len(names) != 9 or len(set(names)) != 9 or frame_index < 0:
        raise ValueError('Joint timing comparison requires nine distinct measured models')
    start = frame_index % len(names)
    return names[start:] + names[:start]


def validate_rows(rows, config, reference, base):
    measured, names = fresh_names(config, reference), all_names(config, reference)
    snrs, seeds = base['evaluation']['snrs_db'], base['evaluation']['noise_seeds']
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    expected = {(index, snr, seed, name) for index in range(100) for snr in snrs for seed in seeds for name in names}
    if len(rows) != len(lookup) or set(lookup) != expected:
        raise RuntimeError('Joint comparison requires all 33600 registered outcomes')
    source_ids, frozen_encoders = [], set()
    for index in range(100):
        identities, seed_noise = set(), {seed: set() for seed in seeds}
        for snr in snrs:
            sent_by_method = {name: set() for name in measured}
            for seed in seeds:
                frozen_signals, frozen_received = set(), set()
                for name in names:
                    row = lookup[index, snr, seed, name]
                    identities.add(row['image_id'])
                    seed_noise[seed].add(row['noise_sha256'])
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    if ((int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, header, 3060 - header) or
                        abs(float(row['total_energy']) - 6120) > 1e-5):
                        raise RuntimeError('Joint/reference N/E/header budget differs')
                    if not all(np.isfinite(float(row[metric])) for metric in METRICS):
                        raise RuntimeError('nonfinite or failed outcomes cannot be omitted')
                    if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
                        raise RuntimeError('severe-distortion definition changed')
                    if name in measured:
                        if int(row['available_updates']) != 5000 or row['decoder_interface'] != 'continuous_mean':
                            raise RuntimeError('Joint/frozen training opportunity or receiver interface differs')
                        if int(row['selected_step']) not in config['full_calibration_steps'] or int(row['selected_global_data_step']) != 7000 + int(row['selected_step']):
                            raise RuntimeError('Joint/frozen selected checkpoint was not eligible')
                        if any(len(str(row[key])) != 64 for key in ('transmitted_sha256', 'received_sha256', 'encoder_sha256', 'checkpoint_sha256')):
                            raise RuntimeError('missing actual transmitter/observation/selection evidence')
                        sent_by_method[name].add(row['transmitted_sha256'])
                        if 'no_history' in name and (row['receiver_forward_scope'] != 'fine_only_no_history_auxiliary_reads_pruned' or
                                                     float(row['prune_feature_max_error']) != 0):
                            raise RuntimeError('no-history was weakened or its image input was changed')
                        if not all(np.isfinite(float(row[key])) and float(row[key]) >= 0 for key in ('online_TX_seconds', 'receiver_seconds')):
                            raise RuntimeError('new timing observations are missing or invalid')
                        if name.startswith('frozen__'):
                            if row['training_policy'] != 'receiver_only':
                                raise RuntimeError('frozen control update policy differs')
                            frozen_encoders.add(row['encoder_sha256'])
                            frozen_signals.add(row['transmitted_sha256'])
                            frozen_received.add(row['received_sha256'])
                        elif row['training_policy'] != 'joint_E_R':
                            raise RuntimeError('new method is not the registered Joint training policy')
                if len(frozen_signals) != 1 or len(frozen_received) != 1:
                    raise RuntimeError('the six frozen-sender receivers did not use the same actual y')
            if any(len(values) != 1 for values in sent_by_method.values()):
                raise RuntimeError('a transmitter reads receiver noise or retained receiver state')
        if len(identities) != 1 or any(len(hashes) != 1 for hashes in seed_noise.values()):
            raise RuntimeError('source grouping or paired standard noise differs')
        source_ids.append(next(iter(identities)))
    validate_population(source_ids)
    if len(frozen_encoders) != 1:
        raise RuntimeError('frozen receivers have different sender parameters')
    return lookup, names


def validate_selection(rows, config, milestone, control):
    for row in rows:
        name = row['arm']
        if not name.startswith(('joint__', 'frozen__')):
            continue
        variant = name.split('__', 1)[1]
        selection = (milestone if name.startswith('joint__') else control)['selected'][variant]
        if int(row['selected_step']) != selection['step'] or row['checkpoint_sha256'] != selection['checkpoint_sha256']:
            raise RuntimeError('development output used a different full-calibration checkpoint')
        if name.startswith('frozen__') and row['encoder_sha256'] != control['frozen']['encoder']:
            raise RuntimeError('frozen-sender row does not match its qualified Encoder')


def statistics(rows, config, reference, base):
    lookup, names = validate_rows(rows, config, reference, base)
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100, size=(base['evaluation']['bootstrap_resamples'], 100))
    summaries, paired, interactions = [], [], []
    for snrs in (base['evaluation']['primary_snrs_db'], *[[snr] for snr in base['evaluation']['snrs_db']]):
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, snr, seed, name][metric]) for snr in snrs
            for seed in base['evaluation']['noise_seeds']]) for index in range(100)]) for metric in METRICS} for name in names}
        for name in names:
            row = {'arm': name, 'snrs_db': label, 'source_images': 100, **{metric: float(array.mean()) for metric, array in values[name].items()}}
            frames = [lookup[index, snr, seed, name] for index in range(100) for snr in snrs for seed in base['evaluation']['noise_seeds']]
            for field in ('online_TX_seconds', 'receiver_seconds'):
                durations = [float(frame[field]) for frame in frames if frame[field] != '']
                row[field + '_mean'] = float(np.mean(durations)) if durations else ''
                row[field + '_p95'] = float(np.percentile(durations, 95)) if durations else ''
            summaries.append(row)
        for method, control in comparison_pairs(config, reference):
            for metric in METRICS:
                difference = values[method][metric] - values[control][metric]
                low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
                paired.append({'snrs_db': label, 'method': method, 'control': control, 'metric': metric,
                    'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
        for method, control in history_contrasts():
            for metric in METRICS:
                difference = (values[joint_name(method)][metric] - values[joint_name(control)][metric]) - (
                    values[frozen_name(method)][metric] - values[frozen_name(control)][metric])
                low, high = np.percentile(difference[draws].mean(1), (2.5, 97.5))
                interactions.append({'snrs_db': label, 'method_structure': method, 'control_structure': control, 'metric': metric,
                    'joint_minus_frozen_structure_contrast': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired, interactions


def support_statistics(rows, support_rows, config, reference, base):
    from wetok_comm.deep_support import SUPPORT_NAME

    methods = fresh_names(config, reference)
    main = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    support = {(int(row['image_index']), float(row['snr_db']), int(row['seed'])): row for row in support_rows}
    seeds = base['evaluation']['noise_seeds']
    expected = {(index, snr, seed) for index in range(100) for snr in (5., 6.) for seed in seeds}
    if len(support_rows) != 600 or set(support) != expected:
        raise RuntimeError('Joint fixed-support comparison must contain all 600 outcomes')
    for (index, snr, seed), row in support.items():
        if (row['arm'] != SUPPORT_NAME or float(row['condition_snr_db']) != {5.: 4., 6.: 7.}[snr] or
            (int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, 0, 3060) or
            abs(float(row['total_energy']) - 6120) > 1e-5 or not all(np.isfinite(float(row[metric])) for metric in METRICS)):
            raise RuntimeError('Deep support policy, resources or retained outcome changed')
        for name in methods:
            paired = main[index, snr, seed, name]
            if row['image_id'] != paired['image_id'] or row['noise_sha256'] != paired['noise_sha256']:
                raise RuntimeError('fixed-support source/noise pairing differs')
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100, size=(base['evaluation']['bootstrap_resamples'], 100))
    summaries, paired = [], []
    for snrs in ([5.], [6.], [5., 6.]):
        values = {name: {metric: np.array([np.mean([float((support[index, snr, seed] if name == SUPPORT_NAME else
            main[index, snr, seed, name])[metric]) for snr in snrs for seed in seeds]) for index in range(100)]) for metric in METRICS}
            for name in methods + [SUPPORT_NAME]}
        label = '+'.join(map(str, snrs))
        summaries.extend({'arm': name, 'snrs_db': label, 'source_images': 100,
            **{metric: float(array.mean()) for metric, array in outcome.items()}} for name, outcome in values.items())
        for name in methods:
            for metric in METRICS:
                delta = values[name][metric] - values[SUPPORT_NAME][metric]
                low, high = np.percentile(delta[draws].mean(1), (2.5, 97.5))
                paired.append({'snrs_db': label, 'method': name, 'control': SUPPORT_NAME, 'metric': metric,
                    'delta': float(delta.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, paired


def validate_diagnostics(rows, clean_rows, noiseless_rows, support_rows, config, reference):
    clean = {int(row['image_index']): row for row in clean_rows}
    if len(clean_rows) != 100 or set(clean) != set(range(100)):
        raise RuntimeError('correct-native diagnostic must contain the whole source population')
    validate_population([row['image_id'] for row in clean_rows])
    if any(not np.isfinite(float(row[metric])) for row in clean_rows for metric in ('psnr_db', 'ssim', 'lpips', 'dino')):
        raise RuntimeError('native reference metrics must be finite')
    expected = {(index, name) for index in range(100) for name in fresh_names(config, reference)}
    if len(noiseless_rows) != 900 or {(int(row['image_index']), row['arm']) for row in noiseless_rows} != expected:
        raise RuntimeError('Joint/frozen noiseless diagnostic is incomplete')
    for row in rows + noiseless_rows + support_rows:
        anchor = clean[int(row['image_index'])]
        if (row['image_id'] != anchor['image_id'] or
            abs(float(row['lpips']) - float(anchor['lpips']) - float(row['LPIPS_excess_from_native'])) > 1e-10 or
            not all(np.isfinite(float(row[metric])) for metric in METRICS) or
            float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15)):
            raise RuntimeError('native-relative source quality changed')
    for row in noiseless_rows:
        if row['channel'] != 'noiseless_nominal19_not_wireless_ranking' or int(row['complex_uses']) != 3060:
            raise RuntimeError('noiseless output was relabelled as a wireless method')
