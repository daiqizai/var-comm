"""Separately measure eleven frozen receivers, using exact saved quality observations."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(JOINT / 'scripts'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch
import yaml

from evaluate_interfaces import require_uncontended_gpu
from finish_registered_trial import validate_receipt
from grid_controls.common import output_path
from grid_controls.evaluation import audited_grid, learned_names, load_evaluation, model_registry
from grid_controls.references import ImageCache, References, read_rows
from grid_controls.timing import measurement_order, summarize, validate_config
from innovation_comm.hardware import gpu_telemetry
from innovation_comm.inference import receive_for_image, verify_pruned_result
from joint_sender.evaluation import model_registry as old_model_registry
from joint_sender.evaluation_io import digest, write_rows
from wetok_comm.common import artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.evaluation import raw_noise
from wetok_comm.native import FrozenWeTok, indices_to_features
from wetok_comm.training import module_sha256, read_population


def key(name, snr, seed=None):
    owner = 'frozen_shared' if name.startswith('frozen__') else name
    return f'{owner}__snr{float(snr)}' + ('' if seed is None else f'__seed{int(seed)}')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=5000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, original, reference, base, parent_record = load_evaluation()
    validate_config(evaluation)
    if not arguments.execute:
        print('PLAN ONLY: fixed32 sources, five SNRs, seed2001, eleven-model timing; no quality reselection')
        return
    configure_torch()
    milestone, milestone_path, review_path, original_binding = audited_grid(config, original, reference, arguments.step)
    references = References(evaluation, config, original, reference)
    quality = output_path(config, 'evaluation') / 'quality_0005000'
    quality_receipt = quality / 'completion.json'
    validate_receipt(quality_receipt, 'GRID_QUALITY_COMPLETE', {'grid_milestone_sha256': sha256(milestone_path)})
    output = output_path(config, 'evaluation') / 'timing_0005000'
    if (output / 'completion.json').exists():
        raise RuntimeError('timing is already complete')
    bindings = {'grid_milestone_sha256': sha256(milestone_path), 'quality_receipt_sha256': sha256(quality_receipt),
                'original_joint_evaluation_sha256': references.receipt_sha}
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if any(metadata[key] != value for key, value in bindings.items()):
            raise RuntimeError('timing resume changed frozen quality or selection')
    else:
        output.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
            *sorted((EXPERIMENT / 'src/grid_controls').glob('*.py')), INNOVATION / 'src/innovation_comm/inference.py',
            BASE / 'scripts/evaluate_interfaces.py']
        metadata = {'created_local': now(), **bindings, 'source_hashes': snapshot(output, files),
            'quality_rows_changed': False, 'subset_selected_from_quality': False}
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
        status('LOADING_SEPARATE_GRID_TIMING')
        require_uncontended_gpu()
        device = torch.device('cuda:0')
        parent, models, choices = model_registry(config, reference, base, milestone, device)
        old_parent, old_models, old_choices = old_model_registry(original, reference, base,
            original_binding['joint'], original_binding['frozen'], device)
        if module_sha256(parent) != module_sha256(old_parent):
            raise RuntimeError('timing models do not share the declared original parent')
        models.update(old_models)
        choices.update(old_choices)
        names = learned_names(config, original, reference)
        if set(models) != set(names):
            raise RuntimeError('an original strong learned control is missing from timing')
        native = FrozenWeTok(device, 'both')
        objects = {'parent': parent, 'native': native.codec, **models}
        before = {name: module_sha256(model) for name, model in objects.items()}
        if before['native'] != references.frozen_models['native']:
            raise RuntimeError('timing visual codec differs from frozen quality')
        encoder_hashes = {name: module_sha256(model.encoder) for name, model in models.items()}
        tolerance = yaml.safe_load((JOINT / 'configs/evaluation.yaml').read_text())
        quality_rows = read_rows(quality / 'per_frame.csv')
        lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in quality_rows}
        images, codes, identifiers = read_population(base, 'development')
        rows, warmed = [], set()
        seed = evaluation['timing_noise_seed']
        for position, index in enumerate(evaluation['timing_source_indices']):
            directory = output / 'images' / f'{index:03d}'
            source_receipt = directory / 'receipt.json'
            if source_receipt.exists():
                record = json.loads(source_receipt.read_text())
                if record['image_id'] != identifiers[index]:
                    raise RuntimeError('timing source changed during resume')
                for relative, expected in record['output_hashes'].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError('committed timing changed')
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
                raise RuntimeError('timing tokenization differs from source cache')
            truth = indices_to_features(encoded)
            cache = ImageCache()
            local = []
            with np.load(quality / 'images' / f'{index:03d}' / 'waveforms.npz', allow_pickle=False) as new_waves, \
                    np.load(references.root / 'images' / f'{index:03d}' / 'waveforms.npz', allow_pickle=False) as old_waves:
                for snr_position, snr in enumerate(evaluation['timing_snrs_db']):
                    snrs = torch.tensor([snr], device=device)
                    frame = position * len(evaluation['timing_snrs_db']) + snr_position
                    for order_index, name in enumerate(measurement_order(names, frame)):
                        expected_row = lookup[index, snr, seed, name]
                        model = models[name]
                        if choices[name]['checkpoint_sha256'] != expected_row['checkpoint_sha256'] or encoder_hashes[name] != expected_row['encoder_sha256']:
                            raise RuntimeError('timing used a different model than quality')
                        archive = new_waves if name.startswith('grid__') else old_waves
                        saved_signal, saved_received = archive[key(name, snr)], archive[key(name, snr, seed)]
                        if digest(saved_signal) != expected_row['transmitted_sha256'] or digest(saved_received) != expected_row['received_sha256']:
                            raise RuntimeError('timing is not using the exact saved quality waveform')
                        if expected_row['noise_sha256'] != digest(raw_noise(identifiers[index], seed)):
                            raise RuntimeError('timing noise pairing changed')
                        received = torch.tensor(saved_received, device=device)
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
                            raise RuntimeError('timing transmitter does not replay its quality waveform')
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        result = receive_for_image(model, received, snrs)
                        image = native.decode(result['receiver_features'])
                        torch.cuda.synchronize()
                        receiver_seconds = time.perf_counter() - tick
                        prune_error = verify_pruned_result(model, result, received, snrs)
                        image_error = float(np.max(np.abs(image[0].cpu().numpy() - cache.image(expected_row))))
                        if image_error > tolerance['frozen_pixel_max_error'] or digest(received) != expected_row['received_sha256']:
                            raise RuntimeError('timing receiver image or input differs from frozen quality')
                        local.append({'image_index': index, 'image_id': identifiers[index], 'snr_db': snr, 'seed': seed, 'arm': name,
                            'total_complex_uses': 3060, 'total_energy': 6120., 'checkpoint_sha256': expected_row['checkpoint_sha256'],
                            'received_sha256': expected_row['received_sha256'], 'quality_image_sha256': expected_row['image_sha256'],
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
            status('MEASURING_SEPARATE_GRID_TIMING')
            print(f'Grid timing {completed}/32 rows={len(rows)}', flush=True)
        summary, paired = summarize(rows, evaluation, names, base)
        if before != {name: module_sha256(model) for name, model in objects.items()}:
            raise RuntimeError('timing changed model parameters')
        verify_sources(metadata['source_hashes'])
        write_rows(output / 'summary.csv', summary)
        write_rows(output / 'paired.csv', paired)
        status('GRID_TIMING_COMPLETE_NOT_RESEARCH_COMPLETE')
        write_json(output / 'completion.json', {'status': 'GRID_TIMING_COMPLETE', 'completed_local': now(), **bindings,
            'rows': len(rows), 'source_images': completed, 'source_hashes': metadata['source_hashes'],
            'quality_selection_changed': False, 'frozen_model_hashes': before, 'research_goal_complete': False,
            'output_hashes': artifact_hashes(output)})
    except BaseException as error:
        status('GRID_TIMING_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
