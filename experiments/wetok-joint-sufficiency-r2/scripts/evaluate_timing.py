"""Replay the selected R2 and eleven frozen learned controls in a separate timing phase."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(JOINT / 'scripts'),
    str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch
import yaml

from evaluate_interfaces import require_uncontended_gpu
from finish_registered_trial import validate_receipt
from innovation_comm.hardware import gpu_telemetry
from innovation_comm.inference import receive_for_image, verify_pruned_result
from joint_sender.evaluation_io import digest, write_rows
from sufficiency.evaluation import audited_endpoint, evaluation_output, learned_names, load_evaluation
from sufficiency.references import ImageCache, References, read_rows
from sufficiency.timing import all_models, archive_origin, measurement_order, summarize, validate_config, waveform_key
from wetok_comm.common import artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.evaluation import raw_noise
from wetok_comm.native import FrozenWeTok, indices_to_features
from wetok_comm.training import module_sha256, read_population


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, original, grid, reference, base, parent_record = load_evaluation()
    validate_config(evaluation)
    if arguments.step != evaluation['required_total_updates']:
        raise ValueError('R2 timing requires all four10000 opportunities')
    if not arguments.execute:
        print('PLAN ONLY: fixed32 sources, five SNRs, seed2001, fifteen models; no quality reselection')
        return
    configure_torch()
    milestone, milestone_path, review_path, source_binding = audited_endpoint(config, arguments.step)
    references = References(evaluation, config, original, grid, reference)
    quality = evaluation_output(evaluation, 'quality')
    quality_receipt = quality / 'completion.json'
    validate_receipt(quality_receipt, 'R2_QUALITY_COMPLETE', {'r2_milestone_sha256': sha256(milestone_path),
        'r2_review_sha256': sha256(review_path), 'reference_quality_sha256': references.receipt_sha})
    output = evaluation_output(evaluation, 'timing')
    if (output / 'completion.json').exists():
        raise RuntimeError('R2 timing is already complete')
    bindings = {'r2_milestone_sha256': sha256(milestone_path), 'quality_receipt_sha256': sha256(quality_receipt),
        'reference_quality_sha256': references.receipt_sha}
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if any(metadata[key] != value for key, value in bindings.items()):
            raise RuntimeError('R2 timing resume changed frozen quality or selection')
    else:
        output.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
            *sorted((EXPERIMENT / 'src/sufficiency').glob('*.py')), INNOVATION / 'src/innovation_comm/inference.py',
            BASE / 'scripts/evaluate_interfaces.py']
        metadata = {'created_local': now(), **bindings, 'source_hashes': snapshot(output, files),
            'quality_rows_changed': False, 'subset_selected_from_quality': False,
            'process_checks_do_not_prove_continuous_exclusivity': True}
        write_json(output / 'metadata.json', metadata)
    started = time.monotonic()
    session = output / 'sessions' / f'{time.time_ns()}_{os.getpid()}.json'
    completed = 0

    def status(value, **extra):
        record = {'status': value, 'pid': os.getpid(), 'local_time': now(), 'completed_sources': completed,
            'observed_process_seconds': time.monotonic() - started, 'research_goal_complete': False, **extra}
        write_json(output / 'status.json', record)
        write_json(session, record)

    try:
        status('LOADING_SEPARATE_R2_TIMING')
        require_uncontended_gpu()
        device = torch.device('cuda:0')
        parent, models, choices = all_models(config, original, grid, reference, base, milestone, device)
        names = learned_names(config, original, grid, reference)
        native = FrozenWeTok(device, 'both')
        objects = {'parent': parent, 'native': native.codec, **models}
        before = {name: module_sha256(model) for name, model in objects.items()}
        if before['native'] != references.frozen_models['native']:
            raise RuntimeError('R2 timing changed the frozen visual codec')
        encoder_hashes = {name: module_sha256(model.encoder) for name, model in models.items()}
        tolerance = yaml.safe_load((JOINT / 'configs/evaluation.yaml').read_text())
        quality_rows = read_rows(quality / 'per_frame.csv')
        lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in quality_rows}
        images, codes, identifiers = read_population(base, 'development')
        rows, warmed = [], set()
        seed = evaluation['timing_noise_seed']
        archive_roots = {'r2': quality, 'grid': references.root, 'joint': references.previous.root}
        for position, index in enumerate(evaluation['timing_source_indices']):
            directory = output / 'images' / f'{index:03d}'
            source_receipt = directory / 'receipt.json'
            if source_receipt.exists():
                record = json.loads(source_receipt.read_text())
                if record['image_id'] != identifiers[index]:
                    raise RuntimeError('R2 timing source changed during resume')
                for relative, expected in record['output_hashes'].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError('a committed R2 timing source changed')
                rows.extend(read_rows(directory / 'per_frame.csv'))
                completed += 1
                continue
            require_uncontended_gpu()
            directory.mkdir(parents=True, exist_ok=True)
            hardware = [{'phase': 'before_source', **gpu_telemetry()}]
            source = images[index:index + 1].to(device).float()
            if images.dtype == torch.uint8:
                source = source / 255
            references.validate_source(index, identifiers[index], source)
            native.encode(source)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            encoded = native.encode(source)
            torch.cuda.synchronize()
            visual_seconds = time.perf_counter() - tick
            if not torch.equal(encoded.cpu(), torch.from_numpy(np.array(codes[index:index + 1, 0], copy=True))):
                raise RuntimeError('R2 timing tokenization differs from its source cache')
            truth = indices_to_features(encoded)
            cache, local = ImageCache(), []
            with ExitStack() as archives:
                waves = {owner: archives.enter_context(np.load(root / f'images/{index:03d}/waveforms.npz', allow_pickle=False))
                    for owner, root in archive_roots.items()}
                for snr_position, snr in enumerate(evaluation['timing_snrs_db']):
                    snrs = torch.tensor([snr], device=device)
                    frame = position * len(evaluation['timing_snrs_db']) + snr_position
                    for order_index, name in enumerate(measurement_order(names, frame)):
                        expected = lookup[index, snr, seed, name]
                        model = models[name]
                        if choices[name]['checkpoint_sha256'] != expected['checkpoint_sha256'] or encoder_hashes[name] != expected['encoder_sha256']:
                            raise RuntimeError('R2 timing used a different selected model than quality')
                        archive = waves[archive_origin(name)]
                        saved_signal, saved_received = archive[waveform_key(name, snr)], archive[waveform_key(name, snr, seed)]
                        if digest(saved_signal) != expected['transmitted_sha256'] or digest(saved_received) != expected['received_sha256']:
                            raise RuntimeError('R2 timing changed the saved quality waveform')
                        if expected['noise_sha256'] != digest(raw_noise(identifiers[index], seed)):
                            raise RuntimeError('R2 timing noise pairing changed')
                        received = torch.tensor(saved_received, device=device)
                        require_uncontended_gpu()
                        if name not in warmed:
                            for warmup in range(2):
                                model.transmit(truth, snrs)
                                native.decode(receive_for_image(model, received, snrs)['receiver_features'])
                            warmed.add(name)
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        signal = model.transmit(truth, snrs)
                        torch.cuda.synchronize()
                        transmitter_seconds = time.perf_counter() - tick
                        signal_error = float((signal.cpu() - torch.from_numpy(saved_signal)).abs().max())
                        if signal_error > tolerance['frozen_signal_max_error']:
                            raise RuntimeError('R2 timing transmitter failed to replay its quality waveform')
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        result = receive_for_image(model, received, snrs)
                        image = native.decode(result['receiver_features'])
                        torch.cuda.synchronize()
                        receiver_seconds = time.perf_counter() - tick
                        require_uncontended_gpu()
                        prune_error = verify_pruned_result(model, result, received, snrs)
                        image_error = float(np.max(np.abs(image[0].cpu().numpy() - cache.image(expected))))
                        if image_error > tolerance['frozen_pixel_max_error'] or digest(received) != expected['received_sha256']:
                            raise RuntimeError('R2 timing receiver image or observation differs from quality')
                        local.append({'image_index': index, 'image_id': identifiers[index], 'snr_db': snr, 'seed': seed, 'arm': name,
                            'total_complex_uses': 3060, 'total_energy': 6120., 'checkpoint_sha256': expected['checkpoint_sha256'],
                            'received_sha256': expected['received_sha256'], 'quality_image_sha256': expected['image_sha256'],
                            'replayed_image_max_error': image_error, 'replayed_signal_max_error': signal_error,
                            'prune_feature_max_error': prune_error, 'receiver_order_index': order_index,
                            'online_TX_seconds': visual_seconds + transmitter_seconds, 'receiver_seconds': receiver_seconds,
                            'timing_scope': 'separate_matched_uncontended_RX_including_visual_Decoder'})
            require_uncontended_gpu()
            hardware.append({'phase': 'after_source', **gpu_telemetry()})
            write_rows(directory / 'per_frame.csv', local)
            write_json(directory / 'hardware.json', hardware)
            write_json(source_receipt, {'image_id': identifiers[index], 'timing_rows': len(local), 'output_hashes': artifact_hashes(directory)})
            rows.extend(local)
            completed += 1
            write_rows(output / 'per_frame.csv', rows)
            status('MEASURING_SEPARATE_R2_TIMING')
            print(f'R2 timing {completed}/32 rows={len(rows)}', flush=True)
        summary, paired = summarize(rows, evaluation, names, base)
        if before != {name: module_sha256(model) for name, model in objects.items()}:
            raise RuntimeError('R2 timing changed model parameters')
        verify_sources(metadata['source_hashes'])
        write_rows(output / 'per_frame.csv', rows)
        write_rows(output / 'summary.csv', summary)
        write_rows(output / 'paired.csv', paired)
        status('R2_TIMING_COMPLETE_NOT_RESEARCH_COMPLETE')
        sessions = [json.loads(path.read_text()) for path in sorted((output / 'sessions').glob('*.json'))]
        terminal = {'R2_TIMING_COMPLETE_NOT_RESEARCH_COMPLETE', 'R2_TIMING_FAILED_OR_INTERRUPTED'}
        write_json(output / 'completion.json', {'status': 'R2_TIMING_COMPLETE', 'completed_local': now(), **bindings,
            'rows': len(rows), 'source_images': completed, 'source_hashes': metadata['source_hashes'],
            'observed_process_hours_all_sessions': sum(row['observed_process_seconds'] for row in sessions) / 3600,
            'time_lower_bound_due_to_unfinished_sessions': any(row['status'] not in terminal for row in sessions),
            'quality_selection_changed': False, 'frozen_model_hashes': before, 'research_goal_complete': False,
            'process_checks_do_not_prove_continuous_exclusivity': True, 'output_hashes': artifact_hashes(output)})
    except BaseException as error:
        status('R2_TIMING_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
