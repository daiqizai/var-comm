"""Fresh Joint/receiver-only quality and timing, preserving all six strong receiver controls."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch

from evaluate_interfaces import require_uncontended_gpu
from innovation_comm.evaluation import reference_names, validate_population
from innovation_comm.hardware import gpu_telemetry
from innovation_comm.inference import receive_for_image, verify_pruned_result
from joint_sender.common import output_path
from joint_sender.evaluation import load_evaluation, matched_milestone, measurement_order, model_registry, support_statistics, validate_diagnostics, validate_rows, validate_selection
from joint_sender.evaluation_io import SourceImages, digest, write_rows
from joint_sender.references import References, read_rows
from wetok_comm.common import PROJECT, WORKSPACE, artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.deep_support import SUPPORT_NAME, FrozenDeepSupport, add_actual_noise, load_deep_support
from wetok_comm.evaluation import metric_models, metrics, raw_noise
from wetok_comm.interface_evaluation import feature_diagnostics
from wetok_comm.native import FrozenWeTok, indices_to_features
from wetok_comm.training import module_sha256, read_population


def waveform_key(name, snr, seed=None):
    owner = 'frozen_shared' if name.startswith('frozen__') else name
    return f'{owner}__snr{float(snr)}' + ('' if seed is None else f'__seed{int(seed)}')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=5000)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, reference, base, parent_record = load_evaluation()
    if arguments.step != 5000:
        raise ValueError('Joint final development requires equal 5000-update opportunities')
    if not arguments.execute:
        print('PLAN ONLY: nine fresh models, all sixteen methods, 33600 main rows and 600 support rows; no new training')
        return
    require_uncontended_gpu()
    configure_torch()
    milestone, milestone_path, review_path, control, control_path, control_review = matched_milestone(config, reference, 5000)
    references = References(evaluation)
    deep_config = load_deep_support()
    output = (arguments.output_dir or output_path(config, 'evaluation') / 'step_0005000').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()) or (output / 'completion.json').exists():
        raise ValueError('output escapes project or overwrites a completed result')
    bindings = {'milestone_sha256': sha256(milestone_path), 'review_sha256': sha256(review_path),
        'control_milestone_sha256': sha256(control_path), 'control_review_sha256': sha256(control_review),
        'reference_qualification_sha256': sha256(references.qualification_path)}
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if any(metadata[key] != value for key, value in bindings.items()):
            raise RuntimeError('Joint selection/reference changed during evaluation')
    else:
        output.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
            *sorted((EXPERIMENT / 'src/joint_sender').glob('*.py')), *sorted((INNOVATION / 'src/innovation_comm').glob('*.py')),
            *sorted((BASE / 'src/wetok_comm').glob('*.py')), BASE / 'scripts/evaluate_interfaces.py',
            BASE / 'configs/deep_support.yaml', *[WORKSPACE / name for name in deep_config['source_hashes']]]
        metadata = {'created_local': now(), **bindings, 'selects_or_trains': False, 'source_hashes': snapshot(output, files)}
        write_json(output / 'metadata.json', metadata)
    started = time.monotonic()
    session_started = now()
    session_path = output / 'sessions' / f'{time.time_ns()}_{os.getpid()}.json'
    completed_images = 0

    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'completed_images': completed_images, 'research_goal_complete': False, **extra})
        write_json(session_path, {'status': value, 'pid': os.getpid(), 'started_local': session_started,
            'last_recorded_local': now(), 'elapsed_seconds': time.monotonic() - started,
            'resumed': arguments.resume, 'completed_images': completed_images, **extra})

    try:
        status('LOADING_JOINT_AND_ALL_FROZEN_RECEIVERS')
        device = torch.device('cuda:0')
        parent, networks, selections = model_registry(config, reference, base, milestone, control, device)
        native = FrozenWeTok(device, 'both')
        perceptual, dino = metric_models(device)
        deep = FrozenDeepSupport(deep_config, device)
        objects = {'parent': parent, 'native': native.codec, 'lpips': perceptual, 'dino': dino, 'deep_support': deep.model, **networks}
        before = {name: module_sha256(model) for name, model in objects.items()}
        images, codes, identifiers = read_population(base, 'development')
        validate_population(identifiers)
        rows, clean_rows, noiseless_rows, support_rows, hardware_rows = [], [], [], [], []
        maxima = {name: 0. for name in ('power_error', 'frozen_signal_error', 'frozen_observation_error', 'frozen_pixel_error', 'support_pixel_error')}
        native_warmed, warmed, support_warmed = False, set(), set()
        for index, identifier in enumerate(identifiers):
            directory = output / 'images' / f'{index:03d}'
            receipt_path = directory / 'receipt.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt['image_id'] != identifier:
                    raise RuntimeError('resumed source identity differs')
                for relative, expected in receipt['output_hashes'].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError('committed Joint source output changed')
                for destination, filename in ((rows, 'per_frame.csv'), (clean_rows, 'clean.csv'),
                                              (noiseless_rows, 'noiseless.csv'), (support_rows, 'deep_support.csv')):
                    destination.extend(read_rows(directory / filename))
                hardware_rows.extend(json.loads((directory / 'hardware.json').read_text()))
                maxima = {key: max(value, receipt['maxima'][key]) for key, value in maxima.items()}
                completed_images = index + 1
                continue
            require_uncontended_gpu()
            directory.mkdir(parents=True, exist_ok=True)
            hardware = [{'image_index': index, 'image_id': identifier, 'phase': 'before_source', **gpu_telemetry()}]
            source = images[index:index + 1].to(device)
            truth_codes = torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)).to(device)
            if not native_warmed:
                warm_fq = indices_to_features(native.encode(source))
                native.decode(warm_fq)
                warm_snr = torch.tensor([1.], device=device)
                parent.transmit(warm_fq, warm_snr)
                for name, model in networks.items():
                    if name.startswith('joint__'):
                        model.transmit(warm_fq, warm_snr)
                native_warmed = True
            torch.cuda.synchronize()
            tick = time.perf_counter()
            actual_codes = native.encode(source)
            source_fq = indices_to_features(actual_codes)
            torch.cuda.synchronize()
            visual_seconds = time.perf_counter() - tick
            if not torch.equal(actual_codes, truth_codes):
                raise RuntimeError('online visual Encoder differs from the fixed source cache')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            clean = native.decode(source_fq)
            torch.cuda.synchronize()
            clean_seconds = time.perf_counter() - tick
            clean_score = metrics(source, [clean[0].cpu()], perceptual, dino)[0]
            for metric, tolerance in evaluation['reference_metric_tolerances'].items():
                if abs(clean_score[metric] - float(references.clean_rows[index][metric])) > tolerance:
                    raise RuntimeError('the correct-native quality anchor changed')
            clean_row = {'image_index': index, 'image_id': identifier, 'reference': 'correct_native_Fq_not_a_wireless_method',
                'raw_bits': 8192, 'online_visual_TX_seconds': visual_seconds, 'native_decoder_seconds': clean_seconds, **clean_score}
            archive = SourceImages(directory)
            local_rows, local_noiseless, local_support, scores_to_replay = [], [], [], {}
            waveforms, features = {}, {}
            local_maxima = {key: 0. for key in maxima}
            for snr_index, snr in enumerate(base['evaluation']['snrs_db']):
                snrs = torch.tensor([snr], device=device)
                signals, tx_seconds = {}, {}
                torch.cuda.synchronize()
                tick = time.perf_counter()
                common_signal = parent.transmit(source_fq, snrs)
                torch.cuda.synchronize()
                common_tx_seconds = time.perf_counter() - tick
                signal_error = float((common_signal.cpu() - references.signal(index, identifier, snr)).abs().max())
                local_maxima['frozen_signal_error'] = max(local_maxima['frozen_signal_error'], signal_error)
                if signal_error > evaluation['frozen_signal_max_error']:
                    raise RuntimeError('frozen TX does not reproduce its qualified signal')
                for name, model in networks.items():
                    if name.startswith('frozen__'):
                        signals[name], tx_seconds[name] = common_signal, common_tx_seconds
                    else:
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        signals[name] = model.transmit(source_fq, snrs)
                        torch.cuda.synchronize()
                        tx_seconds[name] = time.perf_counter() - tick
                    waveforms[waveform_key(name, snr)] = signals[name].cpu().numpy()
                    local_maxima['power_error'] = max(local_maxima['power_error'], float((signals[name].square().sum(-1).mean(-1) - 2).abs().max()))
                for seed_index, seed in enumerate(base['evaluation']['noise_seeds']):
                    noise64 = raw_noise(identifier, seed)
                    noise_hash = digest(noise64)
                    noise = torch.tensor(noise64[None], dtype=torch.float32, device=device)
                    observations = {name: value + noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None] for name, value in signals.items()}
                    common = {'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed,
                        'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': noise_hash}
                    frame_index = (index * len(base['evaluation']['snrs_db']) + snr_index) * len(base['evaluation']['noise_seeds']) + seed_index
                    observed_error = float((observations['frozen__single_pass'].cpu() - references.observed(index, identifier, snr, seed)).abs().max())
                    local_maxima['frozen_observation_error'] = max(local_maxima['frozen_observation_error'], observed_error)
                    if observed_error > evaluation['frozen_signal_max_error']:
                        raise RuntimeError('frozen receiver observation differs from its qualified replay')
                    for position, name in enumerate(measurement_order(networks, frame_index)):
                        model, received = networks[name], observations[name]
                        received_hash = digest(received)
                        if name not in warmed:
                            for warmup in range(2):
                                native.decode(receive_for_image(model, received, snrs)['receiver_features'])
                            warmed.add(name)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        result = receive_for_image(model, received, snrs)
                        image = native.decode(result['receiver_features'])
                        torch.cuda.synchronize()
                        rx_seconds = time.perf_counter() - tick
                        prune_error = verify_pruned_result(model, result, received, snrs)
                        if digest(received) != received_hash:
                            raise RuntimeError('receiver mutated the actual observation')
                        previous, pixel_error = None, ''
                        if name.startswith('frozen__'):
                            old_image, previous = references.image(index, identifier, snr, seed, name, noise_hash)
                            pixel_error = float((image[0].cpu() - old_image).abs().max())
                            local_maxima['frozen_pixel_error'] = max(local_maxima['frozen_pixel_error'], pixel_error)
                            if pixel_error > evaluation['frozen_pixel_max_error']:
                                raise RuntimeError('fresh frozen receiver does not reproduce its qualified image')
                        locator = archive.add(image[0], previous)
                        if previous is not None:
                            scores_to_replay[locator['image_sha256']] = previous
                        waveforms[waveform_key(name, snr, seed)] = received.cpu().numpy()
                        features[f'{name}__snr{snr}__seed{seed}'] = result['receiver_features'][0].cpu().numpy()
                        local_rows.append({**common, 'arm': name, 'variant': model.variant, 'training_policy': selections[name]['training_policy'],
                            'available_updates': 5000, 'parent_updates': 7000, 'selected_step': selections[name]['step'],
                            'selected_global_data_step': 7000 + selections[name]['step'], 'checkpoint_sha256': selections[name]['checkpoint_sha256'],
                            'encoder_sha256': selections[name]['encoder_sha256'],
                            'selected_encoder_differs_from_parent': selections[name]['selected_encoder_differs_from_parent'],
                            'communication_parameters': sum(value.numel() for value in model.parameters()),
                            'optimized_parameters': selections[name]['optimized_parameters'], 'header_uses': 0, 'data_uses': 3060,
                            'raw_bits': 8192, 'coded_bits': '', 'transmitted_sha256': digest(signals[name]), 'received_sha256': received_hash,
                            'receiver_order_index': position, 'receiver_forward_scope': 'fine_only_no_history_auxiliary_reads_pruned'
                                if model.variant == 'multiscale_no_history' else 'full_declared_receiver',
                            'prune_feature_max_error': prune_error, 'frozen_pixel_max_error': pixel_error,
                            'online_TX_seconds': visual_seconds + tx_seconds[name], 'receiver_seconds': rx_seconds,
                            'latency_source': 'current_host_complete_RX_including_internal_E_and_visual_Decoder',
                            **feature_diagnostics(result, source_fq, 'continuous_mean'), **locator})
                    for name in reference_names():
                        image, previous = references.image(index, identifier, snr, seed, name, noise_hash)
                        locator = archive.add(image, previous)
                        scores_to_replay[locator['image_sha256']] = previous
                        local_rows.append({**common, 'arm': name, 'header_uses': int(previous['header_uses']),
                            'data_uses': int(previous['data_uses']), 'raw_bits': previous['raw_bits'], 'coded_bits': previous['coded_bits'],
                            'online_TX_seconds': '', 'receiver_seconds': '', 'latency_source': 'historical_reference_not_a_current_timing_ranking',
                            'historical_TX_seconds': previous.get('historical_TX_seconds', previous['online_TX_seconds']),
                            'historical_RX_seconds': previous.get('historical_RX_seconds', previous['receiver_seconds']),
                            'bit_error_rate': previous.get('bit_error_rate', ''), 'decoder_interface': 'frozen_system_reference', **locator})
                    if float(snr) in deep_config['actual_to_condition_snr']:
                        if snr not in support_warmed:
                            for warmup in range(2):
                                deep.receive(add_actual_noise(deep.transmit(source, snr), noise, snr), snr)
                            support_warmed.add(snr)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        deep_signal = deep.transmit(source, snr)
                        torch.cuda.synchronize()
                        deep_tx_seconds = time.perf_counter() - tick
                        deep_received = add_actual_noise(deep_signal, noise, snr)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        deep_image = deep.receive(deep_received, snr)
                        torch.cuda.synchronize()
                        deep_rx_seconds = time.perf_counter() - tick
                        old_image, previous = references.support(index, identifier, snr, seed, noise_hash)
                        pixel_error = float((deep_image[0].cpu() - old_image).abs().max())
                        local_maxima['support_pixel_error'] = max(local_maxima['support_pixel_error'], pixel_error)
                        if pixel_error > deep_config['pixel_replay_max_error']:
                            raise RuntimeError('Deep fixed-support image changed')
                        locator = archive.add(deep_image[0], previous)
                        scores_to_replay[locator['image_sha256']] = previous
                        local_support.append({**common, 'arm': SUPPORT_NAME, 'condition_snr_db': deep_config['actual_to_condition_snr'][snr],
                            'header_uses': 0, 'data_uses': 3060, 'online_TX_seconds': deep_tx_seconds, 'receiver_seconds': deep_rx_seconds,
                            'transmitted_sha256': digest(deep_signal), 'old_reference_pixel_max_error': pixel_error, **locator})
                        support_key = f'{SUPPORT_NAME}__snr{snr}'
                        if support_key in waveforms and not np.array_equal(waveforms[support_key], deep_signal.cpu().numpy()):
                            raise RuntimeError('Deep sender depends on its receiver noise seed')
                        waveforms[support_key] = deep_signal.cpu().numpy()
            for name, model in networks.items():
                snrs = torch.tensor([19.], device=device)
                result = receive_for_image(model, signals[name], snrs)
                prune_error = verify_pruned_result(model, result, signals[name], snrs)
                image = native.decode(result['receiver_features'])
                previous = None
                if name.startswith('frozen__'):
                    old_image, previous = references.noiseless(index, identifier, model.variant)
                    pixel_error = float((image[0].cpu() - old_image).abs().max())
                    if pixel_error > evaluation['frozen_pixel_max_error']:
                        raise RuntimeError('frozen noiseless image changed')
                locator = archive.add(image[0], previous)
                if previous is not None:
                    scores_to_replay[locator['image_sha256']] = previous
                local_noiseless.append({'image_index': index, 'image_id': identifier, 'arm': name,
                    'channel': 'noiseless_nominal19_not_wireless_ranking', 'complex_uses': 3060, 'prune_feature_max_error': prune_error,
                    **feature_diagnostics(result, source_fq, 'continuous_mean'), **locator})
                features[f'{name}__noiseless19'] = result['receiver_features'][0].cpu().numpy()
            scores = metrics(source, archive.metrics_images, perceptual, dino)
            for row in local_rows + local_support + local_noiseless:
                row.update(scores[archive.metrics_index[row['image_sha256']]])
                row['LPIPS_excess_from_native'] = row['lpips'] - clean_score['lpips']
                row['severe_distortion'] = int(row['LPIPS_excess_from_native'] >= .15)
                if row['image_sha256'] in scores_to_replay:
                    expected = scores_to_replay[row['image_sha256']]
                    for metric, tolerance in evaluation['reference_metric_tolerances'].items():
                        if abs(row[metric] - float(expected[metric])) > tolerance:
                            raise RuntimeError('qualified reference metric does not replay')
            if local_maxima['power_error'] > 1e-5:
                raise RuntimeError('actual Joint signal violates its energy budget')
            require_uncontended_gpu()
            hardware.append({'image_index': index, 'image_id': identifier, 'phase': 'after_source', **gpu_telemetry()})
            for filename, values in (('per_frame.csv', local_rows), ('clean.csv', [clean_row]), ('noiseless.csv', local_noiseless), ('deep_support.csv', local_support)):
                write_rows(directory / filename, values)
            write_json(directory / 'hardware.json', hardware)
            archive.save()
            np.savez(directory / 'waveforms.npz', **waveforms)
            np.savez_compressed(directory / 'receiver_features.npz', **features)
            write_json(receipt_path, {'image_id': identifier, 'maxima': local_maxima, 'new_images': len(archive.new_images),
                'existing_image_pointers_reused': archive.reused_frozen, 'output_hashes': artifact_hashes(directory)})
            rows.extend(local_rows)
            clean_rows.append(clean_row)
            noiseless_rows.extend(local_noiseless)
            support_rows.extend(local_support)
            hardware_rows.extend(hardware)
            maxima = {key: max(value, local_maxima[key]) for key, value in maxima.items()}
            completed_images = index + 1
            write_rows(output / 'per_frame.csv', rows)
            status('EVALUATING_JOINT_AND_ALL_FROZEN_RECEIVERS')
            print(f'Joint/frozen development {completed_images}/100 rows={len(rows)}', flush=True)
        validate_rows(rows, config, reference, base)
        validate_selection(rows, config, milestone, control)
        validate_diagnostics(rows, clean_rows, noiseless_rows, support_rows, config, reference)
        support_statistics(rows, support_rows, config, reference, base)
        after = {name: module_sha256(model) for name, model in objects.items()}
        if before != after:
            raise RuntimeError('Joint evaluation changed model parameters')
        verify_sources(metadata['source_hashes'])
        for filename, values in (('per_frame.csv', rows), ('native_reference.csv', clean_rows), ('noiseless_mapping.csv', noiseless_rows),
                                 ('deep_support_supplement.csv', support_rows), ('hardware_telemetry.csv', hardware_rows)):
            write_rows(output / filename, values)
        output_hashes = artifact_hashes(output)
        status('JOINT_EVALUATION_COMPLETE_NOT_RESEARCH_COMPLETE')
        for changed in (output / 'status.json', session_path):
            output_hashes[str(changed.relative_to(output))] = sha256(changed)
        sessions = [json.loads(path.read_text()) for path in sorted((output / 'sessions').glob('*.json'))]
        terminal = {'JOINT_EVALUATION_COMPLETE_NOT_RESEARCH_COMPLETE', 'JOINT_EVALUATION_FAILED_OR_INTERRUPTED'}
        write_json(output / 'completion.json', {'status': 'JOINT_EVALUATION_COMPLETE', 'completed_local': now(), **bindings,
            'rows': len(rows), 'noiseless_rows': len(noiseless_rows), 'support_rows': len(support_rows), 'maxima': maxima,
            'frozen_models_before': before, 'frozen_models_after': after, 'source_hashes': metadata['source_hashes'],
            'nine_models_freshly_timed': True, 'same_y_scope': 'only_the_six_frozen_sender_receivers',
            'evaluation_observed_wall_hours_all_sessions': sum(row['elapsed_seconds'] for row in sessions) / 3600,
            'time_lower_bound_due_to_unfinished_sessions': any(row['status'] not in terminal for row in sessions),
            'reference_unchanged': True, 'research_goal_complete': False, 'output_hashes': output_hashes})
    except BaseException as error:
        status('JOINT_EVALUATION_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
