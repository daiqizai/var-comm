"""Evaluate selected hard/continuous receivers without altering their deployed interface."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np
import torch

from train_milestone import write_csv
from wetok_comm.common import PROJECT, WORKSPACE, artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.deep_support import SUPPORT_NAME, DeepSupportReferences, FrozenDeepSupport, add_actual_noise, load_deep_support, supplement_statistics
from wetok_comm.evaluation import metric_models, metrics, raw_noise
from wetok_comm.interface_evaluation import FrozenImageReferences, feature_diagnostics, load_evaluation_config, reference_names, validate_grid
from wetok_comm.interface_study import interface_definitions, interface_output, make_interface_network
from wetok_comm.native import FrozenWeTok, indices_to_features
from wetok_comm.training import module_sha256, read_population


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def require_uncontended_gpu():
    output = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if any(not line.isdigit() for line in lines):
        raise RuntimeError('GPU process ownership cannot be verified')
    foreign = {int(line) for line in lines} - {os.getpid()}
    if foreign:
        raise RuntimeError(f'other GPU compute workloads are present: {sorted(foreign)}; not stopping them')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, study, base, parent = load_evaluation_config()
    definitions = interface_definitions(study)
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'new_arms': 9, 'frozen_references': 7, 'expected_wireless_rows': 33600,
                          'separate_fixed_support_rows': 600,
                          'continuous_interface': 'decode_receiver_features_not_native_signs', 'milestone': arguments.milestone}, indent=2))
        return
    require_uncontended_gpu()
    configure_torch()
    deep_config = load_deep_support()
    training = interface_output(study, 'training')
    milestone_path = training / 'milestones' / f'additional_{arguments.milestone:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    if milestone['status'] != 'INTERFACE_MILESTONE_COMPLETE' or milestone['additional_updates_per_arm'] != arguments.milestone:
        raise RuntimeError('interface training milestone incomplete')
    verify_sources(milestone['source_hashes'])
    review_path = interface_output(study, 'analysis') / f'calibration_{arguments.milestone:07d}/completion.json'
    review = json.loads(review_path.read_text())
    verify_sources(review['source_hashes'])
    if (review['status'] != 'INTERFACE_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE' or
        review['paired_data_Adam_energy_and_selection_audit'] != 'PASS' or review['milestone_sha256'] != sha256(milestone_path)):
        raise RuntimeError('full calibration selection and paired history have not been audited')
    output = (arguments.output_dir or interface_output(study, 'evaluation') / f'additional_{arguments.milestone:07d}').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()) or (output / 'completion.json').exists():
        raise ValueError('output escapes project or would overwrite a completed evaluation')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if metadata['milestone_sha256'] != sha256(milestone_path) or metadata['calibration_review_sha256'] != sha256(review_path):
            raise RuntimeError('selected milestone or calibration audit changed during evaluation')
    else:
        output.mkdir(parents=True, exist_ok=False)
        metadata = {'created_local': now(), 'milestone_sha256': sha256(milestone_path), 'calibration_review_sha256': sha256(review_path),
            'reference_receipt_sha256': evaluation['reference_receipt_sha256'], 'selects_or_trains': False,
            'deep_support_reference_sha256': deep_config['reference_receipt_sha256'],
            'source_hashes': snapshot(output, [Path(__file__), EXPERIMENT / 'scripts/train_milestone.py',
                EXPERIMENT / 'configs/interface_evaluation.yaml', EXPERIMENT / 'configs/interface_study.yaml',
                EXPERIMENT / 'configs/deep_support.yaml', EXPERIMENT / 'docs/deep_support_supplement.md',
                *[WORKSPACE / relative for relative in deep_config['source_hashes']],
                EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/interface_evaluation_protocol.md',
                *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])}
        write_json(output / 'metadata.json', metadata)
    started = time.time()
    monotonic_started = time.monotonic()
    session_started_local = now()
    session_path = output / 'sessions' / f'{time.time_ns()}_{os.getpid()}.json'
    completed_images = 0
    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'completed_images': completed_images, 'research_goal_complete': False, **extra})
        write_json(session_path, {'status': value, 'pid': os.getpid(), 'started_local': session_started_local,
            'last_recorded_local': now(), 'elapsed_seconds': time.monotonic() - monotonic_started,
            'resumed': arguments.resume, 'completed_images': completed_images, **extra})
    try:
        status('LOADING_SELECTED_INTERFACE_MODELS')
        references = FrozenImageReferences(evaluation, study)
        support_references = DeepSupportReferences(deep_config, base)
        device = torch.device('cuda:0')
        native = FrozenWeTok(device, 'both')
        perceptual, dino = metric_models(device)
        support_model = FrozenDeepSupport(deep_config, device)
        networks = {}
        for name, definition in definitions.items():
            selected = milestone['selected'][name]
            path = training / selected['checkpoint']
            if sha256(path) != selected['checkpoint_sha256']:
                raise RuntimeError('selected interface checkpoint changed')
            stored = torch.load(path, map_location='cpu', weights_only=True)
            if any(stored[key] != definition[key] for key in ('variant', 'interface')) or stored['additional_step'] != selected['step']:
                raise RuntimeError('checkpoint interface or calibration selection was swapped')
            networks[name] = make_interface_network(base, definition, stored['model'], device).eval().requires_grad_(False)
        objects = {'native': native.codec, 'lpips': perceptual, 'dino': dino, 'fixed_support_Deep': support_model.model, **networks}
        before = {name: module_sha256(model) for name, model in objects.items()}
        images, codes, identifiers = read_population(base, 'development')
        rows, clean_rows, noiseless_rows, support_rows = [], [], [], []
        maximum_power_error, maximum_support_pixel_error = 0., 0.
        native_warmed, warmed_interfaces, warmed_support_snrs = False, set(), set()
        for index, identifier in enumerate(identifiers):
            directory = output / 'images' / f'{index:03d}'
            if (directory / 'receipt.json').exists():
                saved = json.loads((directory / 'receipt.json').read_text())
                for relative, expected in saved['output_hashes'].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError('committed source-image evaluation changed')
                rows.extend(read_rows(directory / 'per_frame.csv'))
                clean_rows.extend(read_rows(directory / 'clean.csv'))
                noiseless_rows.extend(read_rows(directory / 'noiseless.csv'))
                support_rows.extend(read_rows(directory / 'deep_support.csv'))
                maximum_power_error = max(maximum_power_error, saved['maximum_power_error'])
                maximum_support_pixel_error = max(maximum_support_pixel_error, saved['maximum_support_pixel_error'])
                completed_images = index + 1
                continue
            require_uncontended_gpu()
            directory.mkdir(parents=True, exist_ok=True)
            source = images[index:index + 1].to(device)
            grouped = torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)).to(device)
            if not native_warmed:
                native.decode(indices_to_features(native.encode(source)))
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
            local_power, local_support_pixel_error = 0., 0.
            def remember(image, save_new):
                tensor = image.detach().cpu().contiguous()
                digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                if digest not in metric_map:
                    metric_map[digest] = len(metric_images)
                    metric_images.append(tensor)
                if save_new and digest not in new_map:
                    new_map[digest] = len(new_images)
                    new_images.append(tensor)
                return new_map.get(digest, ''), digest
            for snr in base['evaluation']['snrs_db']:
                snrs = torch.tensor([snr], device=device)
                for seed in base['evaluation']['noise_seeds']:
                    noise64 = raw_noise(identifier, seed)
                    noise_hash = hashlib.sha256(noise64.tobytes()).hexdigest()
                    noise = torch.tensor(noise64[None], dtype=torch.float32, device=device)
                    common = {'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed,
                        'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': noise_hash}
                    for name, network in networks.items():
                        if name not in warmed_interfaces:
                            for warmup in range(2):
                                warm_signal = network.transmit(source_fq, snrs)
                                warm = network.receive(warm_signal + noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None], snrs)
                                native.decode(warm['receiver_features'])
                            warmed_interfaces.add(name)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        signal = network.transmit(source_fq, snrs)
                        torch.cuda.synchronize()
                        tx_seconds = time.perf_counter() - tick
                        received = signal + noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None]
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        result = network.receive(received, snrs)
                        image = native.decode(result['receiver_features'])
                        torch.cuda.synchronize()
                        rx_seconds = time.perf_counter() - tick
                        diagnostic = feature_diagnostics(result, source_fq, definitions[name]['interface'])
                        reference, digest = remember(image[0], True)
                        signal_array = signal.detach().cpu().numpy()
                        signal_key = f'{name}__snr{snr}'
                        if signal_key in waveforms and not np.array_equal(waveforms[signal_key], signal_array):
                            raise RuntimeError('transmitter depends on receiver seed or retained receive state')
                        waveforms[signal_key] = signal_array
                        features[f'{name}__snr{snr}__seed{seed}'] = result['receiver_features'][0].cpu().numpy()
                        local_power = max(local_power, float((signal.square().sum(-1).mean(-1) - 2).abs().max()))
                        local_rows.append({**common, 'arm': name, 'header_uses': 0, 'data_uses': 3060, 'raw_bits': 8192,
                            'coded_bits': '', **diagnostic, 'crc_accepted': '', 'image_store': 'new',
                            'image_archive': str(directory / 'reconstructions.npz'), 'image_ref': reference, 'image_sha256': digest,
                            'online_TX_seconds': visual_tx_seconds + tx_seconds, 'receiver_seconds': rx_seconds,
                            'historical_TX_seconds': '', 'historical_RX_seconds': '', 'latency_source': 'current_host_complete_online_TX_RX',
                            'selected_additional_step': milestone['selected'][name]['step'],
                            'selected_global_data_step': study['parent_step'] + milestone['selected'][name]['step'],
                            'transmitted_sha256': hashlib.sha256(signal_array.tobytes()).hexdigest()})
                    for name in reference_names(study):
                        image, previous, archive = references.image(index, identifier, snr, seed, name, noise_hash)
                        unused, digest = remember(image, False)
                        expected_scores[name, snr, seed] = previous
                        local_rows.append({**common, 'arm': name, 'header_uses': int(previous['header_uses']),
                            'data_uses': int(previous['data_uses']), 'raw_bits': previous['raw_bits'], 'coded_bits': previous['coded_bits'],
                            'decoder_interface': 'frozen_reference', 'decoded_feature_kind': 'historical_RGB_reference',
                            'bit_metric_role': 'historical_reference', 'bit_error_rate': previous['bit_error_rate'],
                            'feature_mse': '', 'feature_abs_mean': '', 'feature_saturation_fraction': '',
                            'crc_accepted': previous['crc_accepted'], 'image_store': 'frozen_reference', 'image_archive': archive,
                            'image_ref': int(previous['image_ref']), 'image_sha256': digest, 'online_TX_seconds': '', 'receiver_seconds': '',
                            'historical_TX_seconds': previous['online_TX_seconds'], 'historical_RX_seconds': previous['receiver_seconds'],
                            'latency_source': 'historical_not_a_latency_ranking', 'selected_additional_step': '',
                            'selected_global_data_step': previous['selected_step'], 'transmitted_sha256': ''})
                    if float(snr) in deep_config['actual_to_condition_snr']:
                        if float(snr) not in warmed_support_snrs:
                            for warmup in range(2):
                                warm_signal = support_model.transmit(source, snr)
                                support_model.receive(add_actual_noise(warm_signal, noise, snr), snr)
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
                            raise RuntimeError('fresh Deep support image does not reproduce the frozen protocol')
                        local_support_pixel_error = max(local_support_pixel_error, pixel_error)
                        reference, digest = remember(support_image[0], True)
                        signal_array = support_signal.cpu().numpy()
                        signal_key = f'{SUPPORT_NAME}__snr{snr}'
                        if signal_key in waveforms and not np.array_equal(waveforms[signal_key], signal_array):
                            raise RuntimeError('fixed-support TX changed with receiver noise seed')
                        waveforms[signal_key] = signal_array
                        local_power = max(local_power, float((support_signal.square().sum(-1).mean(-1) - 2).abs().max()))
                        support_scores[snr, seed] = {'psnr_db': float(original_row['psnr_db']),
                            'lpips': float(original_row['lpips_alex']), 'dino': float(original_row['dino_cosine'])}
                        local_support.append({**common, 'arm': SUPPORT_NAME,
                            'condition_snr_db': deep_config['actual_to_condition_snr'][float(snr)], 'header_uses': 0, 'data_uses': 3060,
                            'image_archive': str(directory / 'reconstructions.npz'), 'image_ref': reference, 'image_sha256': digest,
                            'transmitted_sha256': hashlib.sha256(signal_array.tobytes()).hexdigest(),
                            'online_TX_seconds': support_tx_seconds, 'receiver_seconds': support_rx_seconds,
                            'latency_source': 'current_host_fixed_support_supplement', 'old_reference_pixel_max_error': pixel_error})
            for name, network in networks.items():
                snrs = torch.tensor([19.], device=device)
                result = network.receive(network.transmit(source_fq, snrs), snrs)
                reference, digest = remember(native.decode(result['receiver_features'])[0], True)
                local_noiseless.append({'image_index': index, 'image_id': identifier, 'arm': name,
                    'channel': 'noiseless_nominal19_not_wireless_ranking', 'complex_uses': 3060,
                    'image_archive': str(directory / 'reconstructions.npz'), 'image_ref': reference, 'image_sha256': digest,
                    **feature_diagnostics(result, source_fq, definitions[name]['interface'])})
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
                            raise RuntimeError(f'reference metric replay differs: {row["arm"]}, {metric}')
                if row['arm'] == SUPPORT_NAME:
                    old = support_scores[row['snr_db'], row['seed']]
                    for metric, tolerance in deep_config['metric_replay_tolerances'].items():
                        if abs(row[metric] - old[metric]) > tolerance:
                            raise RuntimeError(f'fixed-support replay metric differs: {metric}')
            if local_power > 1e-5:
                raise RuntimeError('physical energy constraint violated')
            require_uncontended_gpu()
            write_csv(directory / 'per_frame.csv', local_rows)
            write_csv(directory / 'clean.csv', [clean_row])
            write_csv(directory / 'noiseless.csv', local_noiseless)
            write_csv(directory / 'deep_support.csv', local_support)
            np.savez_compressed(directory / 'reconstructions.npz', images=torch.stack(new_images).numpy())
            np.savez_compressed(directory / 'receiver_features.npz', **features)
            np.savez(directory / 'waveforms.npz', **waveforms)
            write_json(directory / 'receipt.json', {'image_id': identifier, 'maximum_power_error': local_power,
                'maximum_support_pixel_error': local_support_pixel_error,
                'reference_images_copied': 0, 'new_images': len(new_images), 'output_hashes': artifact_hashes(directory)})
            rows.extend(local_rows)
            clean_rows.append(clean_row)
            noiseless_rows.extend(local_noiseless)
            support_rows.extend(local_support)
            maximum_power_error = max(maximum_power_error, local_power)
            maximum_support_pixel_error = max(maximum_support_pixel_error, local_support_pixel_error)
            completed_images = index + 1
            write_csv(output / 'per_frame.csv', rows)
            status('EVALUATING_INTERFACES')
            print(f'interface development {completed_images}/100 rows={len(rows)}', flush=True)
        validate_grid(rows, study, base)
        supplement_statistics(rows, support_rows, study, base)
        after = {name: module_sha256(model) for name, model in objects.items()}
        if before != after:
            raise RuntimeError('evaluation changed model parameters')
        verify_sources(metadata['source_hashes'])
        write_csv(output / 'per_frame.csv', rows)
        write_csv(output / 'native_reference.csv', clean_rows)
        write_csv(output / 'noiseless_mapping.csv', noiseless_rows)
        write_csv(output / 'deep_support_supplement.csv', support_rows)
        output_hashes = artifact_hashes(output)
        status('INTERFACE_EVALUATION_COMPLETE_NOT_RESEARCH_COMPLETE')
        for changed in (output / 'status.json', session_path):
            output_hashes[str(changed.relative_to(output))] = sha256(changed)
        sessions = [json.loads(path.read_text()) for path in sorted((output / 'sessions').glob('*.json'))]
        terminal = {'INTERFACE_EVALUATION_COMPLETE_NOT_RESEARCH_COMPLETE', 'INTERFACE_EVALUATION_FAILED_OR_INTERRUPTED'}
        write_json(output / 'completion.json', {'status': 'INTERFACE_EVALUATION_COMPLETE', 'completed_local': now(),
            'rows': len(rows), 'noiseless_rows': len(noiseless_rows), 'source_hashes': metadata['source_hashes'],
            'separate_fixed_support_rows': len(support_rows), 'maximum_support_pixel_error': maximum_support_pixel_error,
            'deep_support_reference_receipt_sha256': deep_config['reference_receipt_sha256'],
            'milestone_sha256': sha256(milestone_path), 'reference_receipt_sha256': evaluation['reference_receipt_sha256'],
            'frozen_before': before, 'frozen_after': after, 'maximum_power_error': maximum_power_error,
            'evaluation_wall_hours_this_session': (time.time() - started) / 3600, 'reference_images_copied': 0,
            'evaluation_observed_wall_hours_all_sessions': sum(session['elapsed_seconds'] for session in sessions) / 3600,
            'evaluation_sessions': len(sessions),
            'evaluation_total_time_is_lower_bound': any(session['status'] not in terminal for session in sessions),
            'research_goal_complete': False, 'output_hashes': output_hashes})
    except BaseException as error:
        status('INTERFACE_EVALUATION_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
