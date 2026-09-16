"""Matched six-arm continuation changing only native bit BCE weight."""

import argparse
import csv
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

from train_milestone import choose, component_gradient_rows, write_csv
from wetok_comm.bit_support import arm_definitions, clone_optimizer, cloned_arm, load_repair, optimizer_digest, repair_output
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.model import communicate
from wetok_comm.native import FrozenWeTok
from wetok_comm.objective import losses
from wetok_comm.training import batch_inputs, calibrate, load_lpips, module_sha256, monitoring_indices, paired_batches, read_population


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/bit_support.yaml')
    parser.add_argument('--until-additional', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    repair, base, parent, parent_receipt = load_repair(arguments.config)
    definitions = arm_definitions(repair)
    until = arguments.until_additional or repair['training']['first_additional_milestone']
    if until <= 0 or until % 5000:
        raise ValueError('use complete research milestones; 5000 is not a total research cap')
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'parent_step': repair['parent_step'], 'additional_milestone': until,
                          'arms': definitions, 'only_changed_loss': 'bits_BCE_0p01_to_1'}, indent=2))
        return
    configure_torch()
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < 12000:
        raise RuntimeError('authorized GPU lacks headroom; no other process will be stopped')
    output = repair_output(repair, 'training')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
        if metadata['config_sha256'] != sha256(arguments.config):
            raise RuntimeError('repair configuration changed during continuation')
    else:
        output.mkdir(parents=True, exist_ok=False)
        paths = [Path(__file__), arguments.config, EXPERIMENT / repair['base_config'],
                 EXPERIMENT / 'docs/bit_support_protocol.md', EXPERIMENT / 'scripts/train_milestone.py',
                 *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))]
        metadata = {'created_local': now(), 'source_hashes': snapshot(output, paths), 'config_sha256': sha256(arguments.config),
                    'parent_milestone_sha256': repair['parent_milestone_sha256'],
                    'parent_optimizer_sha256': repair['parent_optimizer_checkpoint_sha256'], 'definitions': definitions}
        write_json(output / 'metadata.json', metadata)
    for name in definitions:
        (output / name / 'checkpoints').mkdir(parents=True, exist_ok=True)
    completed, started = 0, time.time()
    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
                   'additional_updates_per_arm': completed, 'global_step': repair['parent_step'] + completed,
                   'current_additional_milestone': until, **extra})
    try:
        status('LOADING_SHARED_PARENT')
        device = torch.device('cuda:0')
        native, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'native_codec': module_sha256(native.codec), 'lpips': module_sha256(perceptual)}
        saved = torch.load(parent / repair['parent_optimizer_checkpoint'], map_location='cpu', weights_only=True)
        if saved['completed_updates'] != repair['parent_step'] or saved['frozen'] != frozen:
            raise RuntimeError('parent models or frozen visual system mismatch')
        networks, optimizers = {}, {}
        for name, definition in definitions.items():
            networks[name], optimizers[name] = cloned_arm(base, definition['variant'], saved, repair['training']['learning_rate'], device)
        hashes = {name: module_sha256(network) for name, network in networks.items()}
        optimizer_hashes = {name: optimizer_digest(optimizer.state_dict()) for name, optimizer in optimizers.items()}
        for variant in repair['variants']:
            names = [f'{recipe}__{variant}' for recipe in repair['recipes']]
            if len({hashes[name] for name in names}) != 1 or len({optimizer_hashes[name] for name in names}) != 1:
                raise RuntimeError('variant pair is not initialized from identical model/Adam state')
        for optimizer in optimizers.values():
            if any(int(value['step']) != repair['parent_step'] for value in optimizer.state.values()):
                raise RuntimeError('parent optimizer counter was reset')
        write_json(output / 'initialization.json', {'models': hashes, 'optimizers': optimizer_hashes,
                   'parent_step': repair['parent_step'], 'optimizer_deep_copies': True, 'only_changed_loss': 'bits'})
        del saved
        population, calibration = read_population(base, 'train'), read_population(base, 'calibration')
        monitor_indices = monitoring_indices(calibration[2], base)
        histories = {name: [] for name in definitions}
        selected = {name: None for name in definitions}
        calibrated, summaries, gradients = [], [], []
        elapsed_prior = 0.
        timings = {'training': {name: 0. for name in definitions}, 'calibration': {name: 0. for name in definitions}}
        if arguments.resume:
            resumed = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True)
            completed = resumed['completed_additional_updates']
            if completed >= until:
                raise ValueError('requested milestone is already complete')
            for name in definitions:
                networks[name].load_state_dict(resumed['models'][name], strict=True)
                optimizers[name] = clone_optimizer(networks[name], resumed['optimizers'][name], repair['training']['learning_rate'])
            histories, selected, calibrated = resumed['histories'], resumed['selected'], resumed['calibrated']
            summaries, gradients, timings = resumed['summaries'], resumed['gradients'], resumed['timings']
            elapsed_prior = resumed['elapsed_seconds']
            del resumed
        resumed_from = completed

        def persist():
            for name in definitions:
                write_csv(output / name / 'training.csv', histories[name])
                write_json(output / name / 'selected.json', selected[name])
            write_csv(output / 'calibration_summary.csv', summaries)
            write_csv(output / 'gradient_monitor.csv', gradients)
            state = {'completed_additional_updates': completed, 'global_step': repair['parent_step'] + completed,
                     'models': {name: network.state_dict() for name, network in networks.items()},
                     'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                     'histories': histories, 'selected': selected, 'calibrated': calibrated, 'summaries': summaries,
                     'gradients': gradients, 'timings': timings, 'frozen': frozen,
                     'elapsed_seconds': elapsed_prior + time.time() - started}
            temporary = output / 'resume.pending.pt'
            torch.save(state, temporary)
            temporary.replace(output / 'resume.pt')

        def calibration_point():
            if completed in calibrated or (completed % 1000 and completed % 2500):
                return
            full = completed % 2500 == 0
            scope = 'full' if full else 'monitor'
            for name, definition in definitions.items():
                status('CALIBRATING', arm=name, scope=scope)
                tick = time.perf_counter()
                if completed == 0:
                    rows = read_csv(parent / definition['variant'] / 'full_0005000.csv')
                    reused = True
                else:
                    rows = calibrate(networks[name], calibration, native, perceptual, base, device,
                                     None if full else monitor_indices)
                    reused = False
                write_csv(output / name / f'{scope}_{completed:07d}.csv', rows)
                means = {key: float(np.mean([float(row[key]) for row in rows])) for key in ('psnr_db', 'ssim', 'lpips', 'bits_BCE', 'state_error', 'bit_error_rate', 'group_error_rate')}
                for snr in base['channel']['snrs_db']:
                    subset = [row for row in rows if float(row['snr_db']) == snr]
                    summaries.append({'additional_step': completed, 'global_step': repair['parent_step'] + completed, 'arm': name,
                        'variant': definition['variant'], 'recipe': definition['recipe'], 'scope': scope, 'snr_db': snr,
                        'source_images': len(subset), 'reused_parent_forward': reused,
                        **{key: float(np.mean([float(row[key]) for row in subset])) for key in means}})
                if full:
                    path = output / name / 'checkpoints' / f'additional_{completed:07d}.pt'
                    torch.save({'model': networks[name].state_dict(), 'arm': name, 'variant': definition['variant'],
                                'recipe': definition['recipe'], 'additional_step': completed}, path)
                    candidate = {'step': completed, 'global_step': repair['parent_step'] + completed, 'scope': 'full',
                                 'source_images': 1000, 'checkpoint': str(path.relative_to(output)), 'checkpoint_sha256': sha256(path), **means}
                    selected[name] = choose(selected[name], candidate)
                timings['calibration'][name] += time.perf_counter() - tick
                print(f'{scope} {name} additional={completed} PSNR={means["psnr_db"]:.5f} LPIPS={means["lpips"]:.6f} BER={means["bit_error_rate"]:.6f}', flush=True)
            calibrated.append(completed)
            persist()

        calibration_point()
        persist()
        status('TRAINING')
        for batch in paired_batches(base, 20000, repair['parent_step'] + completed, repair['parent_step'] + until):
            if batch['step'] != repair['parent_step'] + completed:
                raise RuntimeError('continuation data position differs from the shared parent')
            inputs = batch_inputs(population, batch, device)
            if completed == 0 and not gradients:
                for name, definition in definitions.items():
                    networks[name].train().zero_grad(set_to_none=True)
                    gradients.extend({'arm': name, 'additional_step': 0, **row}
                                     for row in component_gradient_rows(networks[name], inputs, native, perceptual, definition['weights']))
            for name, definition in definitions.items():
                network, optimizer = networks[name], optimizers[name]
                network.train()
                optimizer.zero_grad(set_to_none=True)
                weights = definition['weights']
                totals = {key: 0. for key in ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate', 'group_error_rate')}
                power_error = 0.
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                for index in range(4):
                    part = {key: value[index:index + 1] for key, value in inputs.items()}
                    result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
                    objective, components, diagnostic = losses(result, part['source_fq'], part['images'], native, perceptual, weights)
                    if not torch.isfinite(objective):
                        raise FloatingPointError('nonfinite repair objective')
                    (objective / 4).backward()
                    totals['loss'] += float(objective.detach()) / 4
                    for key, values in components.items():
                        totals[key] += float(values.detach().sum()) / 4
                    for key in ('bit_error_rate', 'group_error_rate'):
                        totals[key] += float(diagnostic[key].detach().sum()) / 4
                    power_error = max(power_error, float((result['transmitted'].detach().square().sum(-1).mean(-1) - 2).abs().max()))
                    del part, result, objective, components, diagnostic
                norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                if power_error > 1e-5:
                    raise RuntimeError('repair changed the physical energy budget')
                timings['training'][name] += seconds
                histories[name].append({'additional_step': completed + 1, 'global_step': batch['step'] + 1,
                    'optimizer_step': batch['step'] + 1, 'batch_sha256': batch['fingerprint'], 'learning_rate': repair['training']['learning_rate'],
                    'bits_weight': weights['bits'], 'gradient_norm': float(norm), 'step_seconds': seconds,
                    'power_max_error': power_error, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), **totals})
            completed += 1
            if completed == resumed_from + 1 or completed % 250 == 0:
                persist()
                status('TRAINING')
                print(f'repair paired updates {completed}/{until} global={repair["parent_step"] + completed} ' +
                      ' '.join(f'{name}={histories[name][-1]["loss"]:.5f}' for name in definitions), flush=True)
            calibration_point()
        if completed != until or completed not in calibrated:
            raise RuntimeError('repair milestone incomplete')
        for optimizer in optimizers.values():
            if any(int(value['step']) != repair['parent_step'] + completed for value in optimizer.state.values()):
                raise RuntimeError('optimizer history was reset or shared between comparison arms')
        if frozen != {'native_codec': module_sha256(native.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('frozen visual models changed')
        verify_sources(metadata['source_hashes'])
        persist()
        status('REPAIR_MILESTONE_COMPLETE_NOT_RESEARCH_FINISHED')
        milestone_path = output / 'milestones' / f'additional_{completed:07d}.json'
        optimizer_path = output / 'milestones' / f'additional_{completed:07d}_optimizer.pt'
        optimizer_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(output / 'resume.pt', optimizer_path)
        write_json(milestone_path, {'status': 'REPAIR_MILESTONE_COMPLETE', 'completed_local': now(), 'additional_updates_per_arm': completed,
            'global_step': repair['parent_step'] + completed, 'source_hashes': metadata['source_hashes'], 'selected': selected,
            'definitions': definitions, 'frozen': frozen, 'timings': timings, 'parent_optimizer_sha256': repair['parent_optimizer_checkpoint_sha256'],
            'optimizer_checkpoint': str(optimizer_path.relative_to(output)), 'optimizer_checkpoint_sha256': sha256(optimizer_path),
            'observed_GPU_hours': (elapsed_prior + time.time() - started) / 3600, 'research_goal_complete': False})
        print('REPAIR_MILESTONE_COMPLETE_REVIEW_BEFORE_NEXT_DECISION', flush=True)
    except BaseException as error:
        status('REPAIR_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
