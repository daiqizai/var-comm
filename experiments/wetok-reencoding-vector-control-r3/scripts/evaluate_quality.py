"""Measure the selected R3 vector control while preserving all twenty-two sealed quality references."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np
import torch

from grid_controls.hardware import admit_quality, quality_telemetry
from innovation_comm.evaluation import validate_population
from joint_sender.evaluation_io import SourceImages, digest, write_rows
from vector_control.evaluation import audited_endpoint, evaluation_output, load_evaluation, model_registry, new_names, validate_diagnostics, validate_reference_diagnostics, validate_rows, validate_selection
from vector_control.references import References, project_reference, read_rows
from wetok_comm.common import artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.evaluation import metric_models, metrics, raw_noise
from wetok_comm.interface_evaluation import feature_diagnostics
from wetok_comm.native import FrozenWeTok, indices_to_features
from wetok_comm.training import module_sha256, read_population


def waveform_key(name, snr, seed=None):
    return f'{name}__snr{float(snr)}' + ('' if seed is None else f'__seed{int(seed)}')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--shared-gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, r2, original, grid, reference, base, parent_record = load_evaluation()
    if arguments.step != evaluation['required_total_updates']:
        raise ValueError('R3 quality requires the complete matched10000 opportunity')
    if not arguments.execute:
        print('PLAN ONLY: 2100 new R3 rows plus46200 sealed reference rows; no training, current latency or new test')
        return
    configure_torch()
    milestone, milestone_path, review_path, qualified = audited_endpoint(config, r2, grid, arguments.step)
    references = References(evaluation, config, r2, original, grid, reference)
    output = evaluation_output(evaluation, 'quality')
    if (output / 'completion.json').exists():
        raise RuntimeError('R3 quality is already complete')
    bindings = {'r3_milestone_sha256': sha256(milestone_path), 'r3_review_sha256': sha256(review_path),
        'reference_quality_sha256': references.receipt_sha, 'source_tensor_table_sha256': references.source_table_sha}
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if any(metadata[key] != value for key, value in bindings.items()) or metadata['shared_start_requested'] != arguments.shared_gpu:
            raise RuntimeError('R3 quality resume changed selection, sources or admission policy')
    else:
        output.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
            *sorted((EXPERIMENT / 'src/vector_control').glob('*.py')),
            *sorted((EXPERIMENT.parent / 'wetok-joint-grid-controls-r1/src/grid_controls').glob('*.py'))]
        metadata = {'created_local': now(), **bindings, 'source_hashes': snapshot(output, files),
            'shared_start_requested': arguments.shared_gpu, 'quality_only_no_current_timing': True}
        write_json(output / 'metadata.json', metadata)
    started = time.monotonic()
    session = output / 'sessions' / f'{time.time_ns()}_{os.getpid()}.json'
    completed = 0

    def status(value, **extra):
        record = {'status': value, 'pid': os.getpid(), 'local_time': now(), 'completed_images': completed,
            'observed_process_seconds': time.monotonic() - started, 'resumed': arguments.resume,
            'research_goal_complete': False, **extra}
        write_json(output / 'status.json', record)
        write_json(session, record)

    try:
        status('QUALIFYING_R3_QUALITY_GPU_ADMISSION')
        admission = admit_quality(evaluation, arguments.shared_gpu)
        write_json(output / 'admissions' / (session.stem + '.json'), admission)
        device = torch.device('cuda:0')
        parent, models, choices = model_registry(config, reference, base, milestone, device)
        native = FrozenWeTok(device, 'both')
        perceptual, dino = metric_models(device)
        objects = {'parent': parent, 'native': native.codec, 'lpips': perceptual, 'dino': dino, **models}
        before = {name: module_sha256(model) for name, model in objects.items()}
        if any(before[key] != references.frozen_models[key] for key in ('native', 'lpips', 'dino')):
            raise RuntimeError('R3 changed the visual codec or metric weights')
        images, codes, identifiers = read_population(base, 'development')
        validate_population(identifiers)
        rows, clean_rows, noiseless_rows, hardware_rows = [], [], [], []
        support_rows = [project_reference(row, references.receipt_sha) for row in references.support]
        for index, identifier in enumerate(identifiers):
            directory = output / 'images' / f'{index:03d}'
            receipt_path = directory / 'receipt.json'
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text())
                if saved['image_id'] != identifier:
                    raise RuntimeError('resumed R3 source identity changed')
                for relative, expected in saved['output_hashes'].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError('a committed R3 source artifact changed')
                rows.extend(read_rows(directory / 'per_frame.csv'))
                clean_rows.extend(read_rows(directory / 'clean.csv'))
                noiseless_rows.extend(read_rows(directory / 'noiseless.csv'))
                hardware_rows.extend(json.loads((directory / 'hardware.json').read_text()))
                completed = index + 1
                continue
            hardware = [{'image_index': index, 'image_id': identifier, 'phase': 'before_source', **quality_telemetry(evaluation)}]
            directory.mkdir(parents=True, exist_ok=True)
            source = images[index:index + 1].to(device).float()
            if images.dtype == torch.uint8:
                source = source / 255
            references.validate_source(index, identifier, source)
            actual_codes = native.encode(source)
            truth_codes = torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True)).to(device)
            if not torch.equal(actual_codes, truth_codes):
                raise RuntimeError('R3 source tokenization does not replay the native cache')
            truth = indices_to_features(actual_codes)
            archive = SourceImages(directory)
            native_locator = archive.add(native.decode(truth)[0], references.clean[index])
            local_rows = references.source_rows(index)
            local_noiseless = [project_reference(row, references.receipt_sha) for row in references.noiseless if int(row['image_index']) == index]
            fresh_rows, fresh_noiseless, waveforms, features, signal19 = [], [], {}, {}, {}
            maximum_power_error = 0.
            for snr in base['evaluation']['snrs_db']:
                snrs = torch.tensor([snr], device=device)
                signals = {name: model.transmit(truth, snrs) for name, model in models.items()}
                for name, signal in signals.items():
                    maximum_power_error = max(maximum_power_error, float((signal.square().sum(-1).mean(-1) - 2).abs().max()))
                    waveforms[waveform_key(name, snr)] = signal.cpu().numpy()
                    if snr == 19.:
                        signal19[name] = signal
                for seed in base['evaluation']['noise_seeds']:
                    noise64 = raw_noise(identifier, seed)
                    noise = torch.tensor(noise64[None], dtype=torch.float32, device=device)
                    for name, model in models.items():
                        received = signals[name] + noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None]
                        received_hash = digest(received)
                        result = model.receive(received, snrs)
                        image = native.decode(result['receiver_features'])
                        if digest(received) != received_hash:
                            raise RuntimeError('R3 receiver modified its observation')
                        choice = choices[name]
                        fresh_rows.append({'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed,
                            'arm': name, 'variant': model.variant, 'training_policy': 'joint_E_R_vector_control_complete_history',
                            'available_updates': 10000, 'new_update_opportunity': 10000, 'parent_updates': 7000,
                            'selected_step': choice['step'], 'selected_global_data_step': 7000 + choice['step'],
                            'checkpoint_sha256': choice['checkpoint_sha256'], 'encoder_sha256': choice['encoder_sha256'],
                            'selected_encoder_differs_from_parent': choice['selected_encoder_differs_from_parent'],
                            'communication_parameters': choice['communication_parameters'], 'optimized_parameters': choice['optimized_parameters'],
                            'total_complex_uses': 3060, 'total_energy': 6120., 'header_uses': 0, 'data_uses': 3060,
                            'raw_bits': 8192, 'coded_bits': '', 'noise_sha256': digest(noise64),
                            'transmitted_sha256': digest(signals[name]), 'received_sha256': received_hash,
                            'r3_quality_origin': 'new_r3_measurement', 'r3_reference_receipt_sha256': '',
                            'quality_origin': 'new_r3_measurement', 'online_TX_seconds': '', 'receiver_seconds': '',
                            'latency_source': 'not_measured_in_this_quality_phase',
                            **feature_diagnostics(result, truth, 'continuous_mean'), **archive.add(image[0])})
                        waveforms[waveform_key(name, snr, seed)] = received.cpu().numpy()
                        features[f'{name}__snr{float(snr)}__seed{int(seed)}'] = result['receiver_features'][0].cpu().numpy()
            for name, model in models.items():
                result = model.receive(signal19[name], torch.tensor([19.], device=device))
                image = native.decode(result['receiver_features'])
                fresh_noiseless.append({'image_index': index, 'image_id': identifier, 'arm': name,
                    'channel': 'noiseless_nominal19_not_wireless_ranking', 'complex_uses': 3060,
                    'r3_quality_origin': 'new_r3_measurement', 'quality_origin': 'new_r3_measurement',
                    'online_TX_seconds': '', 'receiver_seconds': '', 'latency_source': 'not_measured_in_this_quality_phase',
                    **feature_diagnostics(result, truth, 'continuous_mean'), **archive.add(image[0])})
                features[name + '__noiseless19'] = result['receiver_features'][0].cpu().numpy()
            scores = metrics(source, archive.metrics_images, perceptual, dino)
            measured_native = scores[archive.metrics_index[native_locator['image_sha256']]]
            anchor = references.clean[index]
            if anchor['image_id'] != identifier or any(abs(measured_native[key] - float(anchor[key])) > tolerance
                for key, tolerance in evaluation['native_metric_tolerances'].items()):
                raise RuntimeError('R3 changed the native source quality anchor')
            clean = {**project_reference(anchor, references.receipt_sha), **native_locator}
            for row in fresh_rows + fresh_noiseless:
                row.update(scores[archive.metrics_index[row['image_sha256']]])
                row['LPIPS_excess_from_native'] = row['lpips'] - float(anchor['lpips'])
                row['severe_distortion'] = int(row['LPIPS_excess_from_native'] >= .15)
            local_rows.extend(fresh_rows)
            local_noiseless.extend(fresh_noiseless)
            references.validate_reuse(local_rows)
            if maximum_power_error > 1e-5:
                raise RuntimeError('R3 quality waveform violated the energy ledger')
            hardware.append({'image_index': index, 'image_id': identifier, 'phase': 'after_source', **quality_telemetry(evaluation)})
            for filename, values in (('per_frame.csv', local_rows), ('clean.csv', [clean]), ('noiseless.csv', local_noiseless)):
                write_rows(directory / filename, values)
            write_json(directory / 'hardware.json', hardware)
            archive.save()
            np.savez(directory / 'waveforms.npz', **waveforms)
            np.savez_compressed(directory / 'receiver_features.npz', **features)
            write_json(receipt_path, {'image_id': identifier, 'new_quality_rows': len(fresh_rows), 'maximum_power_error': maximum_power_error,
                'reference_quality_sha256': references.receipt_sha, 'output_hashes': artifact_hashes(directory)})
            rows.extend(local_rows)
            clean_rows.append(clean)
            noiseless_rows.extend(local_noiseless)
            hardware_rows.extend(hardware)
            completed = index + 1
            write_rows(output / 'per_frame.csv', rows)
            status('EVALUATING_R3_QUALITY_WITHOUT_LATENCY')
            print(f'R3 quality {completed}/100 rows={len(rows)}', flush=True)
        validate_rows(rows, config, r2, original, grid, reference, base)
        validate_selection(rows, milestone)
        validate_diagnostics(rows, clean_rows, noiseless_rows, support_rows, config, r2, original, grid, reference, base)
        validate_reference_diagnostics(noiseless_rows, support_rows, references)
        references.validate_reuse(rows)
        if len(noiseless_rows) != evaluation['noiseless_rows'] or len(support_rows) != evaluation['support_rows'] or len(clean_rows) != evaluation['source_images']:
            raise RuntimeError('R3 quality diagnostics or reference population are incomplete')
        after = {name: module_sha256(model) for name, model in objects.items()}
        if before != after:
            raise RuntimeError('R3 quality inference changed model parameters')
        verify_sources(metadata['source_hashes'])
        for filename, values in (('per_frame.csv', rows), ('native_reference.csv', clean_rows), ('noiseless_mapping.csv', noiseless_rows), ('deep_support_supplement.csv', support_rows)):
            write_rows(output / filename, values)
        write_json(output / 'hardware.json', hardware_rows)
        status('R3_QUALITY_COMPLETE_NOT_RESEARCH_COMPLETE')
        sessions = [json.loads(path.read_text()) for path in sorted((output / 'sessions').glob('*.json'))]
        terminal = {'R3_QUALITY_COMPLETE_NOT_RESEARCH_COMPLETE', 'R3_QUALITY_FAILED_OR_INTERRUPTED'}
        write_json(output / 'completion.json', {'status': 'R3_QUALITY_COMPLETE', 'completed_local': now(), **bindings,
            'rows': len(rows), 'new_rows': sum(row['arm'] in new_names(config) for row in rows), 'noiseless_rows': len(noiseless_rows),
            'support_rows': len(support_rows), 'source_hashes': metadata['source_hashes'], 'frozen_models_before': before,
            'frozen_models_after': after, 'shared_start_requested': arguments.shared_gpu, 'quality_only_no_current_timing': True,
            'observed_process_hours_all_sessions': sum(row['observed_process_seconds'] for row in sessions) / 3600,
            'time_lower_bound_due_to_unfinished_sessions': any(row['status'] not in terminal for row in sessions),
            'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    except BaseException as error:
        status('R3_QUALITY_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
