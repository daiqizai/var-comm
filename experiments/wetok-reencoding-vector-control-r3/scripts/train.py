"""Train the one registered prediction-vector control from fresh Adam0 with the full matched history."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT.parent / 'wetok-comm-v2-20260912/scripts')]

import torch

from evaluate_interfaces import require_uncontended_gpu
from joint_sender.runtime import backward_paired_batch, calibration
from train_milestone import choose, write_csv
from vector_control.common import calibration_summary, initial_system, load_parent, output_path, qualified_reference, settings, tree_digest, validate_optimizer
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, monitoring_indices, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until-update', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, r2, original, grid, reference, base, parent_record = settings()
    until, variant = arguments.until_update, config['variant']
    if until not in config['full_calibration_steps'] or until <= 0:
        raise ValueError('use a registered positive full-calibration milestone')
    if not arguments.execute:
        print(f'PLAN ONLY: R3 fresh0 to {until} matched updates, not residual-checkpoint fine-tuning')
        return
    require_uncontended_gpu()
    configure_torch()
    qualified = qualified_reference(config, r2, grid)
    qualification_path = output_path(config, 'preparation') / 'initialization.json'
    qualification = json.loads(qualification_path.read_text())
    verify_sources(qualification['source_hashes'])
    if qualification['status'] != 'R3_ORIGINAL_INITIALIZATION_AND_COMPLETE_HISTORY_PASS' or qualification['reference_bindings'] != qualified['bindings']:
        raise RuntimeError('R3 original initialization qualification changed')
    profile_path = output_path(config, 'profile') / 'profile.json'
    profile = json.loads(profile_path.read_text())
    if (profile['status'] != 'R3_REAL_INITIAL_FUNCTION_AND_IMAGE_GRADIENT_PROFILE_PASS' or profile['optimizer_updates'] != 0 or
        profile['qualification_sha256'] != sha256(qualification_path) or profile['reference_bindings'] != qualified['bindings']):
        raise RuntimeError('R3 initial/function/image-gradient profile is incomplete or changed')
    verify_sources(profile['source_hashes'])
    output = output_path(config, 'training')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        dependencies = [EXPERIMENT.parent / 'wetok-joint-sender-r1/src/joint_sender/runtime.py',
            EXPERIMENT.parent / 'wetok-innovation-r1/src/innovation_comm/runtime.py',
            EXPERIMENT.parent / 'wetok-innovation-r1/src/innovation_comm/model.py',
            EXPERIMENT.parent / 'wetok-joint-grid-controls-r1/src/grid_controls/model.py']
        metadata = {'created_local': now(), 'reference_bindings': qualified['bindings'],
            'profile_sha256': sha256(profile_path), 'qualification_sha256': sha256(qualification_path),
            'original_parent_global_step': config['global_data_offset'], 'fresh_Adam_initial_states': 0,
            'initial_calibration_reused_only_after_function_qualification': qualified['zero_calibration'],
            'source_hashes': snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/protocol.md',
                *sorted((EXPERIMENT / 'src/vector_control').glob('*.py')), *dependencies])}
        write_json(output / 'metadata.json', metadata)
    if (metadata['profile_sha256'] != sha256(profile_path) or metadata['qualification_sha256'] != sha256(qualification_path) or
        metadata['reference_bindings'] != qualified['bindings']):
        raise RuntimeError('R3 source/profile/history changed during training')
    completed, elapsed_prior, started = 0, 0., time.monotonic()

    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(), 'variant': variant,
            'completed_updates': completed, 'global_data_step': config['global_data_offset'] + completed,
            'milestone': until, 'research_goal_complete': False, **extra})

    try:
        status('LOADING_MATCHED_R3_PREDICTION_VECTOR')
        device = torch.device('cuda:0')
        parent = load_parent(reference, base, device)
        model = initial_system(parent, reference, device)
        decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
        if frozen != qualified['frozen'] or module_sha256(model) != profile['initial_model_sha256']:
            raise RuntimeError('R3 initial or visual/metric weights changed')
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], betas=tuple(base['training']['betas']), weight_decay=base['training']['weight_decay'])
        (output / 'checkpoints').mkdir(exist_ok=True)
        if arguments.resume:
            saved = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True)
            completed = saved['completed_updates']
            if (completed >= until or saved['global_data_step'] != config['global_data_offset'] + completed or
                saved['frozen'] != frozen or saved['reference_bindings'] != qualified['bindings']):
                raise RuntimeError('R3 must resume the actual model/Adam/data endpoint')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            if tree_digest(model.state_dict()) != tree_digest(saved['model']) or tree_digest(optimizer.state_dict()) != tree_digest(saved['optimizer']):
                raise RuntimeError('R3 model/optimizer restoration changed values')
            history, selected, summaries = saved['history'], saved['selected'], saved['summaries']
            calibrated, calibration_files, timings = saved['calibrated'], saved['calibration_files'], saved['timings']
            elapsed_prior = saved['elapsed_seconds']
            if completed:
                validate_optimizer(optimizer.state_dict(), parameters, config, base, completed)
        else:
            if optimizer.state:
                raise RuntimeError('R3 fresh initialization inherited optimizer moments')
            means, summaries = calibration_summary(qualified['zero_rows'], 0, variant, 'full')
            checkpoint = output / 'checkpoints/step_0000000.pt'
            torch.save({'model': model.state_dict(), 'variant': variant, 'step': 0, 'encoder_trained': True}, checkpoint)
            selected = choose(None, {'step': 0, 'scope': 'full', 'source_images': 1000, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': sha256(checkpoint), **means})
            history, calibrated = [], [0]
            calibration_files = {'0': copy.deepcopy(qualified['zero_calibration'])}
            timings = {'training': 0., 'calibration': 0.}
            write_json(output / 'initialization.json', {'model_sha256': module_sha256(model), 'frozen': frozen,
                'fresh_Adam_initial_states': 0, 'initial_calibration_reused': qualified['zero_calibration'],
                'reference_bindings': qualified['bindings']})
        if len(history) != completed or 0 not in calibrated:
            raise RuntimeError('R3 resume lost its history or qualified initial calibration')
        for row, expected in zip(history, qualified['trace']):
            if any(row[key] != expected[key] for key in ('step', 'global_data_step', 'batch_sha256', 'paired_standard_noise_sha256')):
                raise RuntimeError('R3 saved training prefix changed the paired data')
        population = read_population(base, 'train')
        calibration_population = read_population(base, 'calibration')
        monitor = monitoring_indices(calibration_population[2], base)

        def persist():
            write_csv(output / 'training.csv', history)
            write_csv(output / 'calibration_summary.csv', summaries)
            write_json(output / 'selected.json', selected)
            state = {'completed_updates': completed, 'global_data_step': config['global_data_offset'] + completed,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history, 'selected': selected,
                'summaries': summaries, 'calibrated': calibrated, 'calibration_files': calibration_files, 'timings': timings,
                'frozen': frozen, 'reference_bindings': qualified['bindings'], 'elapsed_seconds': elapsed_prior + time.monotonic() - started}
            pending = output / 'resume.pending.pt'
            torch.save(state, pending)
            pending.replace(output / 'resume.pt')

        def calibrate_point():
            nonlocal selected
            if completed in calibrated or completed % config['monitor_every']:
                return
            full = completed in config['full_calibration_steps']
            scope = 'full' if full else 'monitor'
            status('CALIBRATING_R3_PREDICTION_VECTOR', scope=scope)
            tick = time.perf_counter()
            rows = calibration(model, calibration_population, decoder, perceptual, reference, base, device, None if full else monitor)
            table = output / f'{scope}_{completed:07d}.csv'
            write_csv(table, rows)
            means, summary = calibration_summary(rows, completed, variant, scope)
            summaries.extend(summary)
            if full:
                checkpoint = output / 'checkpoints' / f'step_{completed:07d}.pt'
                torch.save({'model': model.state_dict(), 'variant': variant, 'step': completed, 'encoder_trained': True}, checkpoint)
                selected = choose(selected, {'step': completed, 'scope': 'full', 'source_images': 1000,
                    'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), **means})
                calibration_files[str(completed)] = {'path': str(table), 'sha256': sha256(table)}
            timings['calibration'] += time.perf_counter() - tick
            calibrated.append(completed)
            persist()
            print(f'{scope} R3 {variant} step={completed} LPIPS={means["lpips"]:.6f} PSNR={means["psnr_db"]:.5f}', flush=True)

        persist()
        status('TRAINING_R3_MATCHED_PREDICTION_VECTOR')
        for batch in paired_batches(base, 20000, config['global_data_offset'] + completed, config['global_data_offset'] + until):
            expected = qualified['trace'][completed]
            if batch['step'] != config['global_data_offset'] + completed or batch['fingerprint'] != expected['batch_sha256']:
                raise RuntimeError('R3 data/augmentation/SNR order differs from the residual reference')
            inputs = batch_inputs(population, batch, device)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            metrics = backward_paired_batch(model, inputs, decoder, perceptual, config['unchanged_loss'])
            if metrics['paired_standard_noise_sha256'] != expected['paired_standard_noise_sha256']:
                raise RuntimeError('R3 standard noise differs from residual reference')
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'], error_if_nonfinite=True))
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            timings['training'] += seconds
            completed += 1
            history.append({'step': completed, 'global_data_step': batch['step'] + 1, 'batch_sha256': batch['fingerprint'],
                'gradient_norm': norm, 'step_seconds': seconds, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), **metrics})
            if completed == 1 or completed % 100 == 0:
                persist()
                status('TRAINING_R3_MATCHED_PREDICTION_VECTOR')
                print(f'R3 updates {completed}/{until} loss={metrics["loss"]:.6f}', flush=True)
            calibrate_point()
        if completed != until or completed not in calibrated:
            raise RuntimeError('R3 milestone lacks its complete calibration')
        validate_optimizer(optimizer.state_dict(), parameters, config, base, completed)
        if frozen != {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('R3 changed frozen visual models or parent')
        verify_sources(metadata['source_hashes'])
        persist()
        milestones = output / 'milestones'
        milestones.mkdir(exist_ok=True)
        checkpoint = milestones / f'step_{completed:07d}_optimizer.pt'
        os.link(output / 'resume.pt', checkpoint)
        write_json(milestones / f'step_{completed:07d}.json', {'status': 'R3_VECTOR_CONTROL_MILESTONE_COMPLETE', 'completed_local': now(),
            'completed_updates': completed, 'global_data_step': config['global_data_offset'] + completed, 'selected': selected,
            'frozen': frozen, 'timings': timings, 'source_hashes': metadata['source_hashes'], 'reference_bindings': qualified['bindings'],
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'observed_process_hours': (elapsed_prior + time.monotonic() - started) / 3600, 'research_goal_complete': False})
        status('R3_MILESTONE_READY_NOT_RESEARCH_COMPLETE')
    except BaseException as error:
        status('R3_TRAINING_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
