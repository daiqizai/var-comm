"""Continue four frozen-architecture models with their actual5000 Adam states and new paired data."""

import argparse
import copy
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
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch

from evaluate_interfaces import require_uncontended_gpu
from joint_sender.runtime import backward_paired_batch, calibration, tensor_sha256
from sufficiency.common import load_parent, load_sources, output_path, restore_systems, settings, tree_digest, validate_optimizer
from train_milestone import choose, write_csv
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, monitoring_indices, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until-update', type=int, default=7500)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, original, grid, reference, base, parent_record = settings()
    until, initial = arguments.until_update, config['initial_total_updates']
    if until not in config['new_full_steps'] or until <= initial:
        raise ValueError('use the registered7500 or10000 full-calibration milestone')
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'variants': config['variants'], 'from_actual_updates': initial,
            'until_total_updates': until, 'optimizer_reset': False, 'new_data_start': config['initial_global_data_step']}, indent=2))
        return
    require_uncontended_gpu()
    configure_torch()
    sources = load_sources(config)
    preparation_path = output_path(config, 'preparation') / 'resume_initialization.json'
    preparation = json.loads(preparation_path.read_text())
    if (preparation['status'] != 'FOUR_MODEL_AND_ADAM_RESTORATION_PASS_NO_NEW_UPDATES' or preparation['optimizer_updates'] != 0 or
        preparation['source_endpoint_hashes'] != sources['hashes']):
        raise RuntimeError('CPU exact-Adam restoration qualification is missing or stale')
    verify_sources(preparation['source_hashes'])
    output = output_path(config, 'training')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/protocol.md',
            *sorted((EXPERIMENT / 'src/sufficiency').glob('*.py')), JOINT / 'src/joint_sender/model.py',
            JOINT / 'src/joint_sender/runtime.py', GRID / 'src/grid_controls/model.py',
            INNOVATION / 'src/innovation_comm/model.py', INNOVATION / 'src/innovation_comm/runtime.py',
            BASE / 'src/wetok_comm/training.py', BASE / 'src/wetok_comm/interface_study.py', BASE / 'src/wetok_comm/native.py']
        metadata = {'created_local': now(), 'source_hashes': snapshot(output, files), 'source_endpoint_hashes': sources['hashes'],
            'preparation_sha256': sha256(preparation_path), 'inherited_calibration_files': sources['calibration_files'],
            'inherited_history_sha256': {name: tree_digest(rows) for name, rows in sources['histories'].items()},
            'prior_training_and_calibration_seconds_by_arm': sources['prior_timings'],
            'initial_total_updates': initial, 'initial_global_data_step': config['initial_global_data_step'], 'optimizer_reset': False}
        write_json(output / 'metadata.json', metadata)
    if metadata['source_endpoint_hashes'] != sources['hashes'] or metadata['preparation_sha256'] != sha256(preparation_path):
        raise RuntimeError('source endpoints or preparation changed during R2')
    completed, elapsed_prior, started = initial, 0., time.monotonic()

    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'completed_total_updates': completed, 'completed_new_updates': completed - initial,
            'global_data_step': config['global_data_offset'] + completed, 'milestone': until,
            'research_goal_complete': False, **extra})

    try:
        status('RESTORING_FOUR_ACTUAL_MODELS_AND_ADAM')
        device = torch.device('cuda:0')
        parent = load_parent(reference, base, device)
        decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
        if frozen != sources['frozen']:
            raise RuntimeError('R2 changed a frozen visual model or original parent')
        saved = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True) if arguments.resume else None
        if saved is not None:
            completed = saved['completed_total_updates']
            if (completed < initial or completed >= until or saved['frozen'] != frozen or
                saved['source_endpoint_hashes'] != sources['hashes'] or saved['global_data_step'] != config['global_data_offset'] + completed):
                raise RuntimeError('invalid actual R2 resume endpoint')
        systems, optimizers, restoration = restore_systems(config, reference, base, sources, parent, device, saved)
        if saved is None:
            if restoration != preparation['restored']:
                raise RuntimeError('GPU Adam/model restoration differs from the qualified CPU source')
            histories, selected = sources['histories'], sources['selected']
            summaries, calibrated = sources['summaries'], sources['calibrated']
            calibration_files = copy.deepcopy(sources['calibration_files'])
            timings = {'training': {name: 0. for name in systems}, 'calibration': {name: 0. for name in systems}}
            write_json(output / 'initialization.json', {'restored': restoration, 'frozen': frozen,
                'source_endpoint_hashes': sources['hashes'], 'starting_optimizer_steps': initial,
                'inherited_selection': selected, 'old_full_calibration_reused_not_rerun': True})
        else:
            histories, selected = saved['histories'], saved['selected']
            summaries, calibrated, calibration_files = saved['summaries'], saved['calibrated'], saved['calibration_files']
            timings, elapsed_prior = saved['timings'], saved['elapsed_seconds']
        for name in systems:
            if len(histories[name]) != completed or tree_digest(histories[name][:initial]) != metadata['inherited_history_sha256'][name]:
                raise RuntimeError('R2 altered inherited training history or update count')
            (output / name / 'checkpoints').mkdir(parents=True, exist_ok=True)
        population = read_population(base, 'train')
        calibration_population = read_population(base, 'calibration')
        monitor = monitoring_indices(calibration_population[2], base)

        def persist():
            for name in systems:
                write_csv(output / name / 'training.csv', histories[name])
                write_json(output / name / 'selected.json', selected[name])
            write_csv(output / 'calibration_summary.csv', summaries)
            state = {'completed_total_updates': completed, 'completed_new_updates': completed - initial,
                'global_data_step': config['global_data_offset'] + completed,
                'models': {name: model.state_dict() for name, model in systems.items()},
                'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                'histories': histories, 'selected': selected, 'summaries': summaries, 'calibrated': calibrated,
                'calibration_files': calibration_files, 'timings': timings, 'frozen': frozen,
                'source_endpoint_hashes': sources['hashes'], 'elapsed_seconds': elapsed_prior + time.monotonic() - started}
            temporary = output / 'resume.pending.pt'
            torch.save(state, temporary)
            temporary.replace(output / 'resume.pt')

        def calibration_point():
            if completed in calibrated or completed % config['monitor_every']:
                return
            full = completed in config['new_full_steps']
            scope = 'full' if full else 'monitor'
            for name, model in systems.items():
                status('CALIBRATING_FOUR_ARM_CONTINUATION', variant=name, scope=scope)
                tick = time.perf_counter()
                rows = calibration(model, calibration_population, decoder, perceptual, reference, base, device, None if full else monitor)
                table = output / name / f'{scope}_{completed:07d}.csv'
                write_csv(table, rows)
                metrics = [key for key in rows[0] if key not in ('image_id', 'snr_db') and rows[0][key] != '']
                means = {key: float(np.mean([row[key] for row in rows])) for key in metrics}
                for snr in base['channel']['snrs_db']:
                    subset = [row for row in rows if row['snr_db'] == snr]
                    summaries.append({'step': completed, 'variant': name, 'scope': scope, 'snr_db': snr, 'source_images': len(subset),
                        **{key: float(np.mean([row[key] for row in subset])) if key in metrics else ''
                           for key in rows[0] if key not in ('image_id', 'snr_db')}})
                if full:
                    path = output / name / 'checkpoints' / f'step_{completed:07d}.pt'
                    torch.save({'model': model.state_dict(), 'variant': name, 'step': completed, 'encoder_trained': True,
                        'optimizer_continued_from': initial}, path)
                    selected[name] = choose(selected[name], {'step': completed, 'scope': 'full', 'source_images': 1000,
                        'checkpoint': str(path), 'checkpoint_origin': 'r2', 'checkpoint_sha256': sha256(path), **means})
                    calibration_files[name][str(completed)] = {'path': str(table), 'sha256': sha256(table)}
                timings['calibration'][name] += time.perf_counter() - tick
                print(f'{scope} R2 {name} total={completed} LPIPS={means["lpips"]:.6f} PSNR={means["psnr_db"]:.5f}', flush=True)
            calibrated.append(completed)
            persist()

        if initial not in calibrated:
            raise RuntimeError('R2 cannot silently redo or omit its inherited5000 full calibration')
        persist()
        status('TRAINING_FOUR_ARM_CONTINUATION')
        start = config['global_data_offset'] + completed
        for batch in paired_batches(base, 20000, start, config['global_data_offset'] + until):
            if batch['step'] != config['global_data_offset'] + completed:
                raise RuntimeError('R2 sampler moved to a different global data position')
            if completed == initial and (batch['fingerprint'] != preparation['first_new_batch_sha256'] or
                tensor_sha256(batch['noise']) != preparation['first_new_noise_sha256']):
                raise RuntimeError('first R2 batch did not continue from the qualified global12000 boundary')
            inputs = batch_inputs(population, batch, device)
            expected_noise = tensor_sha256(batch['noise'])
            for name, model in systems.items():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                totals = backward_paired_batch(model, inputs, decoder, perceptual, config['unchanged_loss'])
                if totals['paired_standard_noise_sha256'] != expected_noise:
                    raise RuntimeError('R2 variants did not share the same actual standard noise')
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'], error_if_nonfinite=True))
                optimizers[name].step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                timings['training'][name] += seconds
                histories[name].append({'step': completed + 1, 'global_data_step': batch['step'] + 1,
                    'batch_sha256': batch['fingerprint'], 'gradient_norm': norm, 'step_seconds': seconds,
                    'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), **totals})
            completed += 1
            if completed == initial + 1 or completed % 100 == 0:
                persist()
                status('TRAINING_FOUR_ARM_CONTINUATION')
                print(f'R2 total updates {completed}/{until} new={completed-initial} ' +
                    ' '.join(f'{name}={histories[name][-1]["loss"]:.6f}' for name in systems), flush=True)
            calibration_point()
        if completed != until or completed not in calibrated:
            raise RuntimeError('R2 milestone did not complete its full calibration')
        for name, model in systems.items():
            validate_optimizer(optimizers[name].state_dict(), [parameter for parameter in model.parameters() if parameter.requires_grad], config, base, completed)
        if frozen != {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('R2 changed frozen source/visual/metric parameters')
        verify_sources(metadata['source_hashes'])
        persist()
        directory = output / 'milestones'
        directory.mkdir(exist_ok=True)
        checkpoint = directory / f'step_{completed:07d}_optimizer.pt'
        os.link(output / 'resume.pt', checkpoint)
        write_json(directory / f'step_{completed:07d}.json', {'status': 'SUFFICIENCY_MILESTONE_COMPLETE', 'completed_local': now(),
            'total_updates_per_arm': completed, 'new_updates_per_arm': completed - initial, 'global_data_step': config['global_data_offset'] + completed,
            'selected': selected, 'calibration_files': calibration_files, 'frozen': frozen, 'timings': timings,
            'source_endpoint_hashes': sources['hashes'], 'source_hashes': metadata['source_hashes'],
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'observed_additional_wall_GPU_hours': (elapsed_prior + time.monotonic() - started) / 3600,
            'optimizer_reset': False, 'research_goal_complete': False})
        status('SUFFICIENCY_MILESTONE_READY_NOT_RESEARCH_COMPLETE')
    except BaseException as error:
        status('SUFFICIENCY_TRAINING_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
