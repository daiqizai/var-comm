"""Evaluate full-calibration receiver selections on the same actual s/y, without training."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src'), str(REFERENCE / 'scripts')]

import numpy as np
import torch

from evaluate_interfaces import require_uncontended_gpu
from train_milestone import write_csv
from innovation_comm.common import output_path
from innovation_comm.hardware import gpu_telemetry, receiver_order
from innovation_comm.inference import receive_for_image, verify_pruned_result
from innovation_comm.evaluation import References, audited_milestone, load_evaluation, model_registry, reference_names, supplement_statistics, validate_native_reference, validate_population, validate_rows, validate_selection
from wetok_comm.common import PROJECT, WORKSPACE, artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.deep_support import SUPPORT_NAME, DeepSupportReferences, FrozenDeepSupport, add_actual_noise, load_deep_support
from wetok_comm.evaluation import metric_models, metrics, raw_noise
from wetok_comm.interface_evaluation import feature_diagnostics
from wetok_comm.native import FrozenWeTok, indices_to_features
from wetok_comm.training import module_sha256, read_population


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def array_sha256(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, base, parent_milestone = load_evaluation()
    if arguments.step not in config['calibration']['full_steps'] or arguments.step <= 0:
        raise ValueError('use a registered complete-calibration receiver milestone')
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'new_receivers': 6, 'references': 7, 'main_rows': 27300,
                          'fixed_support_rows': 600, 'selection': 'complete_calibration_only', 'shared_s_y': True}, indent=2))
        return
    require_uncontended_gpu()
    configure_torch()
    milestone, milestone_path, review_path = audited_milestone(config, arguments.step)
    deep_config = load_deep_support()
    output = (arguments.output_dir or output_path(config, 'evaluation') / f'step_{arguments.step:07d}').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()) or (output / 'completion.json').exists():
        raise ValueError('evaluation output escapes project or overwrites a frozen result')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if metadata['milestone_sha256'] != sha256(milestone_path) or metadata['calibration_review_sha256'] != sha256(review_path):
            raise RuntimeError('receiver milestone or calibration review changed during evaluation')
    else:
        output.mkdir(parents=True, exist_ok=False)
        sources = [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'configs/study.yaml',
            EXPERIMENT / 'docs/evaluation_protocol.md', *sorted((EXPERIMENT / 'src/innovation_comm').glob('*.py')),
            REFERENCE / 'scripts/evaluate_interfaces.py', REFERENCE / 'scripts/train_milestone.py',
            REFERENCE / 'configs/deep_support.yaml', *[WORKSPACE / relative for relative in deep_config['source_hashes']],
            *sorted((REFERENCE / 'src/wetok_comm').glob('*.py'))]
        metadata = {'created_local': now(), 'receiver_updates_per_arm': arguments.step,
            'milestone_sha256': sha256(milestone_path), 'calibration_review_sha256': sha256(review_path),
            'reference_receipt_sha256': evaluation['reference_receipt_sha256'],
            'deep_support_reference_sha256': deep_config['reference_receipt_sha256'],
            'selects_or_trains': False, 'source_hashes': snapshot(output, sources)}
        write_json(output / 'metadata.json', metadata)
    started = time.monotonic()
    session_started_local = now()
    session_path = output / 'sessions' / f'{time.time_ns()}_{os.getpid()}.json'
    completed_images = 0

    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'completed_images': completed_images, 'research_goal_complete': False, **extra})
        write_json(session_path, {'status': value, 'pid': os.getpid(), 'started_local': session_started_local,
            'last_recorded_local': now(), 'elapsed_seconds': time.monotonic() - started, 'resumed': arguments.resume,
            'completed_images': completed_images, **extra})

    try:
        status('LOADING_SELECTED_FIXED_TX_RECEIVERS')
        references = References(evaluation)
        support_references = DeepSupportReferences(deep_config, base)
        device = torch.device('cuda:0')
        native = FrozenWeTok(device, 'both')
        perceptual, dino = metric_models(device)
        support_model = FrozenDeepSupport(deep_config, device)
        parent, networks, selections = model_registry(config, base, milestone, device)
        objects = {'native': native.codec, 'lpips': perceptual, 'dino': dino, 'parent': parent,
                   'fixed_support_Deep': support_model.model, **networks}
        before = {name: module_sha256(model) for name, model in objects.items()}
        images, codes, identifiers = read_population(base, 'development')
        validate_population(identifiers)
        rows, clean_rows, noiseless_rows, support_rows, hardware_rows = [], [], [], [], []
        maximum_power_error, maximum_support_pixel_error, maximum_parent_signal_error = 0., 0., 0.
        native_warmed, warmed_receivers, warmed_support_snrs = False, set(), set()
        for index, identifier in enumerate(identifiers):
            directory = output / 'images' / f'{index:03d}'
            if (directory / 'receipt.json').exists():
                saved = json.loads((directory / 'receipt.json').read_text())
                if saved['image_id'] != identifier:
                    raise RuntimeError('resumed source identity changed')
                for relative, expected in saved['output_hashes'].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError('committed source-image evaluation changed')
                rows.extend(read_rows(directory / 'per_frame.csv'))
                clean_rows.extend(read_rows(directory / 'clean.csv'))
                noiseless_rows.extend(read_rows(directory / 'noiseless.csv'))
                support_rows.extend(read_rows(directory / 'deep_support.csv'))
                hardware_rows.extend(json.loads((directory / 'hardware.json').read_text()))
                maximum_power_error = max(maximum_power_error, saved['maximum_power_error'])
                maximum_support_pixel_error = max(maximum_support_pixel_error, saved['maximum_support_pixel_error'])
                maximum_parent_signal_error = max(maximum_parent_signal_error, saved['maximum_parent_signal_error'])
                completed_images = index + 1
                continue
            require_uncontended_gpu()
            directory.mkdir(parents=True, exist_ok=True)
            local_hardware = [{'image_index': index, 'image_id': identifier, 'phase': 'before_source', **gpu_telemetry()}]
            source = images[index:index + 1].to(device)
            grouped = torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)).to(device)
            if not native_warmed:
                warm_fq = indices_to_features(native.encode(source))
                native.decode(warm_fq)
                parent.transmit(warm_fq, torch.tensor([1.], device=device))
                native_warmed = True
            torch.cuda.synchronize()
            tick = time.perf_counter()
            actual_codes = native.encode(source)
            source_fq = indices_to_features(actual_codes)
            torch.cuda.synchronize()
            visual_tx_seconds = time.perf_counter() - tick
            if not torch.equal(actual_codes, grouped):
                raise RuntimeError('online source Encoder differs from the frozen source cache')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            clean = native.decode(source_fq)
            torch.cuda.synchronize()
            clean_seconds = time.perf_counter() - tick
            clean_score = metrics(source, [clean[0].cpu()], perceptual, dino)[0]
            clean_row = {'image_index': index, 'image_id': identifier, 'reference': 'correct_native_Fq_not_a_wireless_method',
                'raw_bits': 8192, 'online_visual_TX_seconds': visual_tx_seconds, 'native_decoder_seconds': clean_seconds, **clean_score}
            new_images, new_map, metric_images, metric_map = [], {}, [], {}
            local_rows, local_noiseless, local_support, expected_scores, support_scores = [], [], [], {}, {}
            waveforms, features = {}, {}
            local_power, local_support_error, local_parent_error = 0., 0., 0.

            def remember(image, save_new):
                tensor = image.detach().cpu().contiguous()
                digest = array_sha256(tensor)
                if digest not in metric_map:
                    metric_map[digest] = len(metric_images)
                    metric_images.append(tensor)
                if save_new and digest not in new_map:
                    new_map[digest] = len(new_images)
                    new_images.append(tensor)
                return new_map.get(digest, ''), digest

            for snr_position, snr in enumerate(base['evaluation']['snrs_db']):
                snrs = torch.tensor([snr], device=device)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                signal = parent.transmit(source_fq, snrs)
                torch.cuda.synchronize()
                tx_seconds = time.perf_counter() - tick
                signal_array = signal.cpu().numpy()
                signal_hash = array_sha256(signal_array)
                parent_error = float((signal.cpu() - references.parent_signal(index, snr)).abs().max())
                if parent_error > evaluation['parent_signal_replay_tolerance']:
                    raise RuntimeError('shared transmitter does not reproduce the pinned parent waveform')
                local_parent_error = max(local_parent_error, parent_error)
                local_power = max(local_power, float((signal.square().sum(-1).mean(-1) - 2).abs().max()))
                waveforms[f'shared__snr{snr}'] = signal_array
                for noise_position, seed in enumerate(base['evaluation']['noise_seeds']):
                    noise64 = raw_noise(identifier, seed)
                    noise_hash = array_sha256(noise64)
                    noise = torch.tensor(noise64[None], dtype=torch.float32, device=device)
                    received = signal + noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None]
                    received_hash = array_sha256(received)
                    waveforms[f'received__snr{snr}__seed{seed}'] = received.cpu().numpy()
                    common = {'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed,
                        'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': noise_hash}
                    frame_index = (index * len(base['evaluation']['snrs_db']) + snr_position) * len(base['evaluation']['noise_seeds']) + noise_position
                    for order_index, name in enumerate(receiver_order(networks, frame_index)):
                        network = networks[name]
                        if name not in warmed_receivers:
                            for warmup in range(2):
                                native.decode(receive_for_image(network, received, snrs)['receiver_features'])
                            warmed_receivers.add(name)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        result = receive_for_image(network, received, snrs)
                        image = native.decode(result['receiver_features'])
                        torch.cuda.synchronize()
                        rx_seconds = time.perf_counter() - tick
                        prune_error = verify_pruned_result(network, result, received, snrs)
                        if array_sha256(received) != received_hash:
                            raise RuntimeError('a receiver mutated the common observed waveform')
                        reference, digest = remember(image[0], True)
                        key = f'{name}__snr{snr}__seed{seed}'
                        features[key] = result['receiver_features'][0].cpu().numpy()
                        gates = {f'feature_gate_{stage}': '' for stage in (1, 2)}
                        gates.update({f'residual_noise_ratio_{stage}': '' for stage in (1, 2)})
                        for stage, (gate, ratio) in enumerate(zip(result['feature_gates'], result['residual_noise_ratios']), 1):
                            gates[f'feature_gate_{stage}'] = float(gate[0, 0])
                            gates[f'residual_noise_ratio_{stage}'] = float(ratio[0])
                        local_rows.append({**common, 'arm': name, 'variant': network.variant, 'receiver_order_index': order_index,
                            'receiver_forward_scope': 'fine_only_no_history_auxiliary_reads_pruned' if network.variant == 'multiscale_no_history' else 'full_declared_receiver',
                            'no_history_prune_feature_max_error': prune_error,
                            'available_receiver_updates': arguments.step, 'parent_training_updates': 7000,
                            'receiver_trainable_parameters': selections[name]['receiver_trainable_parameters'],
                            'communication_parameters': sum(value.numel() for value in network.parameters()),
                            'selected_receiver_step': selections[name]['step'], 'selected_global_data_step': 7000 + selections[name]['step'],
                            'selected_checkpoint_sha256': selections[name]['checkpoint_sha256'],
                            'encoder_frozen': True, 'shared_encoder_sha256': milestone['frozen']['encoder'],
                            'shared_transmitted_sha256': signal_hash, 'shared_received_sha256': received_hash,
                            'parent_signal_max_error': parent_error, 'header_uses': 0, 'data_uses': 3060,
                            'raw_bits': 8192, 'coded_bits': '', **feature_diagnostics(result, source_fq, 'continuous_mean'), **gates,
                            'crc_accepted': '', 'image_store': 'new', 'image_archive': str(directory / 'reconstructions.npz'),
                            'image_ref': reference, 'image_sha256': digest, 'online_TX_seconds': visual_tx_seconds + tx_seconds,
                            'receiver_seconds': rx_seconds, 'historical_TX_seconds': '', 'historical_RX_seconds': '',
                            'latency_source': 'current_host_RX_includes_internal_E_and_final_visual_Decoder'})
                    for name in reference_names():
                        image, previous, archive = references.image(index, identifier, snr, seed, name, noise_hash)
                        unused, digest = remember(image, False)
                        expected_scores[name, snr, seed] = previous
                        local_rows.append({**common, 'arm': name, 'variant': '', 'receiver_order_index': '', 'available_receiver_updates': '',
                            'receiver_forward_scope': 'frozen_reference', 'no_history_prune_feature_max_error': '',
                            'parent_training_updates': '', 'receiver_trainable_parameters': '', 'communication_parameters': '',
                            'selected_receiver_step': '', 'selected_global_data_step': previous.get('selected_global_data_step', previous['selected_step']),
                            'selected_checkpoint_sha256': '', 'encoder_frozen': '', 'shared_encoder_sha256': '',
                            'shared_transmitted_sha256': '', 'shared_received_sha256': '', 'parent_signal_max_error': '',
                            'header_uses': int(previous['header_uses']), 'data_uses': int(previous['data_uses']),
                            'raw_bits': previous['raw_bits'], 'coded_bits': previous['coded_bits'],
                            'decoder_interface': 'frozen_reference', 'decoded_feature_kind': 'historical_RGB_reference',
                            'bit_metric_role': 'historical_reference', 'bit_error_rate': previous['bit_error_rate'],
                            'feature_mse': '', 'feature_abs_mean': '', 'feature_saturation_fraction': '',
                            'feature_gate_1': '', 'feature_gate_2': '', 'residual_noise_ratio_1': '', 'residual_noise_ratio_2': '',
                            'crc_accepted': previous['crc_accepted'], 'image_store': 'frozen_reference',
                            'image_archive': archive, 'image_ref': int(previous['image_ref']), 'image_sha256': digest,
                            'online_TX_seconds': '', 'receiver_seconds': '', 'historical_TX_seconds': previous['online_TX_seconds'],
                            'historical_RX_seconds': previous['receiver_seconds'], 'latency_source': 'historical_not_a_latency_ranking'})
                    if float(snr) in deep_config['actual_to_condition_snr']:
                        if float(snr) not in warmed_support_snrs:
                            for warmup in range(2):
                                support_model.receive(add_actual_noise(support_model.transmit(source, snr), noise, snr), snr)
                            warmed_support_snrs.add(float(snr))
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        support_signal = support_model.transmit(source, snr)
                        torch.cuda.synchronize()
                        support_tx_seconds = time.perf_counter() - tick
                        support_received = add_actual_noise(support_signal, noise, snr)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        support_image = support_model.receive(support_received, snr)
                        torch.cuda.synchronize()
                        support_rx_seconds = time.perf_counter() - tick
                        original_image, original_row = support_references.image(index, identifier, snr, seed)
                        pixel_error = float((support_image[0].cpu() - original_image).abs().max())
                        if pixel_error > deep_config['pixel_replay_max_error']:
                            raise RuntimeError('fresh Deep support image differs from the frozen protocol')
                        local_support_error = max(local_support_error, pixel_error)
                        reference, digest = remember(support_image[0], True)
                        signal_key = f'{SUPPORT_NAME}__snr{snr}'
                        support_array = support_signal.cpu().numpy()
                        if signal_key in waveforms and not np.array_equal(waveforms[signal_key], support_array):
                            raise RuntimeError('fixed-support transmitter depends on receiver noise')
                        waveforms[signal_key] = support_array
                        local_power = max(local_power, float((support_signal.square().sum(-1).mean(-1) - 2).abs().max()))
                        support_scores[snr, seed] = {'psnr_db': float(original_row['psnr_db']),
                            'lpips': float(original_row['lpips_alex']), 'dino': float(original_row['dino_cosine'])}
                        local_support.append({**common, 'arm': SUPPORT_NAME,
                            'condition_snr_db': deep_config['actual_to_condition_snr'][float(snr)], 'header_uses': 0, 'data_uses': 3060,
                            'image_archive': str(directory / 'reconstructions.npz'), 'image_ref': reference, 'image_sha256': digest,
                            'transmitted_sha256': array_sha256(support_array), 'online_TX_seconds': support_tx_seconds,
                            'receiver_seconds': support_rx_seconds, 'latency_source': 'current_host_fixed_support_supplement',
                            'old_reference_pixel_max_error': pixel_error})
            for name, network in networks.items():
                snrs = torch.tensor([19.], device=device)
                clean_signal = parent.transmit(source_fq, snrs)
                result = receive_for_image(network, clean_signal, snrs)
                prune_error = verify_pruned_result(network, result, clean_signal, snrs)
                reference, digest = remember(native.decode(result['receiver_features'])[0], True)
                local_noiseless.append({'image_index': index, 'image_id': identifier, 'arm': name,
                    'channel': 'noiseless_nominal19_not_wireless_ranking', 'complex_uses': 3060,
                    'no_history_prune_feature_max_error': prune_error,
                    'image_archive': str(directory / 'reconstructions.npz'), 'image_ref': reference, 'image_sha256': digest,
                    **feature_diagnostics(result, source_fq, 'continuous_mean')})
                features[f'{name}__noiseless19'] = result['receiver_features'][0].cpu().numpy()
            scores = metrics(source, metric_images, perceptual, dino)
            for row in local_rows + local_noiseless + local_support:
                row.update(scores[metric_map[row['image_sha256']]])
                row['LPIPS_excess_from_native'] = row['lpips'] - clean_score['lpips']
                row['severe_distortion'] = int(row['LPIPS_excess_from_native'] >= .15)
                if row.get('image_store') == 'frozen_reference':
                    old = expected_scores[row['arm'], row['snr_db'], row['seed']]
                    for metric, tolerance in evaluation['reference_metric_tolerances'].items():
                        if abs(row[metric] - float(old[metric])) > tolerance:
                            raise RuntimeError(f'reference metric differs: {row["arm"]}, {metric}')
                if row['arm'] == SUPPORT_NAME:
                    old = support_scores[row['snr_db'], row['seed']]
                    for metric, tolerance in deep_config['metric_replay_tolerances'].items():
                        if abs(row[metric] - old[metric]) > tolerance:
                            raise RuntimeError(f'fixed-support replay metric differs: {metric}')
            if local_power > 1e-5:
                raise RuntimeError('physical energy constraint violated')
            require_uncontended_gpu()
            local_hardware.append({'image_index': index, 'image_id': identifier, 'phase': 'after_source', **gpu_telemetry()})
            write_csv(directory / 'per_frame.csv', local_rows)
            write_csv(directory / 'clean.csv', [clean_row])
            write_csv(directory / 'noiseless.csv', local_noiseless)
            write_csv(directory / 'deep_support.csv', local_support)
            write_json(directory / 'hardware.json', local_hardware)
            np.savez_compressed(directory / 'reconstructions.npz', images=torch.stack(new_images).numpy())
            np.savez_compressed(directory / 'receiver_features.npz', **features)
            np.savez(directory / 'waveforms.npz', **waveforms)
            write_json(directory / 'receipt.json', {'image_id': identifier, 'maximum_power_error': local_power,
                'maximum_support_pixel_error': local_support_error, 'maximum_parent_signal_error': local_parent_error,
                'reference_images_copied': 0, 'new_images': len(new_images), 'output_hashes': artifact_hashes(directory)})
            rows.extend(local_rows)
            clean_rows.append(clean_row)
            noiseless_rows.extend(local_noiseless)
            support_rows.extend(local_support)
            hardware_rows.extend(local_hardware)
            maximum_power_error = max(maximum_power_error, local_power)
            maximum_support_pixel_error = max(maximum_support_pixel_error, local_support_error)
            maximum_parent_signal_error = max(maximum_parent_signal_error, local_parent_error)
            completed_images = index + 1
            write_csv(output / 'per_frame.csv', rows)
            status('EVALUATING_FIXED_TX_RECEIVERS')
            print(f'receiver development {completed_images}/100 rows={len(rows)}', flush=True)
        validate_rows(rows, config, base)
        validate_selection(rows, milestone)
        validate_native_reference(rows + support_rows, clean_rows, noiseless_rows, config, base)
        supplement_statistics(rows, support_rows, config, base)
        after = {name: module_sha256(model) for name, model in objects.items()}
        if before != after:
            raise RuntimeError('evaluation changed model parameters')
        verify_sources(metadata['source_hashes'])
        write_csv(output / 'per_frame.csv', rows)
        write_csv(output / 'native_reference.csv', clean_rows)
        write_csv(output / 'noiseless_mapping.csv', noiseless_rows)
        write_csv(output / 'deep_support_supplement.csv', support_rows)
        write_csv(output / 'hardware_telemetry.csv', hardware_rows)
        output_hashes = artifact_hashes(output)
        status('INNOVATION_EVALUATION_COMPLETE_NOT_RESEARCH_COMPLETE')
        for changed in (output / 'status.json', session_path):
            output_hashes[str(changed.relative_to(output))] = sha256(changed)
        sessions = [json.loads(path.read_text()) for path in sorted((output / 'sessions').glob('*.json'))]
        terminal = {'INNOVATION_EVALUATION_COMPLETE_NOT_RESEARCH_COMPLETE', 'INNOVATION_EVALUATION_FAILED_OR_INTERRUPTED'}
        write_json(output / 'completion.json', {'status': 'INNOVATION_EVALUATION_COMPLETE', 'completed_local': now(),
            'rows': len(rows), 'noiseless_rows': len(noiseless_rows), 'separate_fixed_support_rows': len(support_rows),
            'source_hashes': metadata['source_hashes'], 'milestone_sha256': sha256(milestone_path),
            'reference_receipt_sha256': evaluation['reference_receipt_sha256'],
            'deep_support_reference_receipt_sha256': deep_config['reference_receipt_sha256'],
            'frozen_before': before, 'frozen_after': after, 'maximum_power_error': maximum_power_error,
            'maximum_support_pixel_error': maximum_support_pixel_error, 'maximum_parent_signal_error': maximum_parent_signal_error,
            'evaluation_wall_hours_this_session': (time.monotonic() - started) / 3600,
            'evaluation_observed_wall_hours_all_sessions': sum(session['elapsed_seconds'] for session in sessions) / 3600,
            'evaluation_sessions': len(sessions), 'evaluation_total_time_is_lower_bound': any(session['status'] not in terminal for session in sessions),
            'reference_images_copied': 0, 'shared_s_y_every_frame': True, 'research_goal_complete': False, 'output_hashes': output_hashes})
    except BaseException as error:
        status('INNOVATION_EVALUATION_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
