"""Nine matched arms separating native ST choice from continuous receiver features."""

import argparse
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

from train_milestone import choose, write_csv
from wetok_comm.bit_support import optimizer_digest
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.interface_study import interface_calibrate, interface_definitions, interface_losses, interface_output, load_interface_study, make_interface_network
from wetok_comm.model import communicate
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, monitoring_indices, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until-additional', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    study, base, parent = load_interface_study()
    definitions = interface_definitions(study)
    until = arguments.until_additional
    if until <= 0 or until % 500:
        raise ValueError('a milestone must coincide with a complete paired calibration boundary')
    if not arguments.execute:
        print(json.dumps({'plan_only': True, 'arms': definitions, 'additional_milestone': until,
                          'optimizer': study['training']['optimizer']}, indent=2))
        return
    active = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU is occupied; do not stop or interfere with another process')
    configure_torch()
    profile_path = interface_output(study, 'profile') / 'profile.json'
    profile = json.loads(profile_path.read_text())
    if profile['status'] != 'INTERFACE_GRADIENT_PROFILE_PASS' or profile['optimizer_updates'] != 0:
        raise RuntimeError('real interface/gradient checks have not passed')
    verify_sources(profile['source_hashes'])
    output = interface_output(study, 'training')
    if arguments.resume:
        metadata = json.loads((output / 'metadata.json').read_text())
        verify_sources(metadata['source_hashes'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        metadata = {'created_local': now(), 'definitions': definitions, 'profile_sha256': sha256(profile_path),
                    'source_hashes': snapshot(output, [Path(__file__), EXPERIMENT / 'configs/interface_study.yaml',
                        EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/interface_protocol.md',
                        EXPERIMENT / 'scripts/train_milestone.py', *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])}
        write_json(output / 'metadata.json', metadata)
    if metadata['profile_sha256'] != sha256(profile_path):
        raise RuntimeError('engineering profile changed after initialization')
    completed, started = 0, time.time()
    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'additional_updates_per_arm': completed, 'global_data_step': study['parent_step'] + completed,
            'milestone': until, 'research_goal_complete': False, **extra})
    try:
        status('LOADING_INTERFACE_TRAINING')
        device = torch.device('cuda:0')
        decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
        if frozen != profile['frozen']:
            raise RuntimeError('frozen visuals differ from the engineering profile')
        networks, optimizers = {}, {}
        for name, definition in definitions.items():
            state = torch.load(parent / definition['variant'] / 'checkpoints/step_0002000.pt', map_location='cpu', weights_only=True)['model']
            networks[name] = make_interface_network(base, definition, state, device)
            optimizers[name] = torch.optim.AdamW(networks[name].parameters(), lr=study['training']['learning_rate'],
                betas=tuple(base['training']['betas']), weight_decay=base['training']['weight_decay'])
            (output / name / 'checkpoints').mkdir(parents=True, exist_ok=True)
        initial_hashes = {name: module_sha256(network) for name, network in networks.items()}
        for variant in study['variants']:
            if len({initial_hashes[f'{interface}__{variant}'] for interface in study['interfaces']}) != 1:
                raise RuntimeError('interfaces have different starting parameters')
        if len({optimizer_digest(value.state_dict()) for value in optimizers.values()}) != 1:
            raise RuntimeError('fresh optimizer settings differ across arms')
        histories, selected = {name: [] for name in definitions}, {name: None for name in definitions}
        calibrated, summaries, elapsed_prior = [], [], 0.
        timings = {kind: {name: 0. for name in definitions} for kind in ('training', 'calibration')}
        if arguments.resume:
            saved = torch.load(output / 'resume.pt', map_location=device, weights_only=True)
            if saved['frozen'] != frozen or saved['completed_additional_updates'] >= until:
                raise RuntimeError('resume frozen state or new milestone is invalid')
            completed = saved['completed_additional_updates']
            for name in definitions:
                networks[name].load_state_dict(saved['models'][name], strict=True)
                optimizers[name].load_state_dict(saved['optimizers'][name])
            histories, selected, calibrated = saved['histories'], saved['selected'], saved['calibrated']
            summaries, timings, elapsed_prior = saved['summaries'], saved['timings'], saved['elapsed_seconds']
            del saved
        else:
            write_json(output / 'initialization.json', {'model_hashes': initial_hashes, 'all_optimizers_fresh_and_equal': True,
                'parent_step': study['parent_step'], 'inherited_Adam_moments': False, 'parameters_per_arm': 2895944})
        population, calibration = read_population(base, 'train'), read_population(base, 'calibration')
        monitor = monitoring_indices(calibration[2], base)

        def persist():
            for name in definitions:
                write_csv(output / name / 'training.csv', histories[name])
                write_json(output / name / 'selected.json', selected[name])
            write_csv(output / 'calibration_summary.csv', summaries)
            state = {'completed_additional_updates': completed, 'global_data_step': study['parent_step'] + completed,
                'models': {name: network.state_dict() for name, network in networks.items()},
                'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                'histories': histories, 'selected': selected, 'calibrated': calibrated, 'summaries': summaries,
                'timings': timings, 'frozen': frozen, 'elapsed_seconds': elapsed_prior + time.time() - started}
            temporary = output / 'resume.pending.pt'
            torch.save(state, temporary)
            temporary.replace(output / 'resume.pt')

        def calibrate_point():
            if completed in calibrated or completed % study['calibration']['monitor_every_updates']:
                return
            full = completed in study['calibration']['full_additional_steps'] or completed == until
            scope = 'full' if full else 'monitor'
            for name, definition in definitions.items():
                status('CALIBRATING_INTERFACES', arm=name, scope=scope)
                tick = time.perf_counter()
                rows = interface_calibrate(networks[name], calibration, decoder, perceptual, base, study['weights'], device,
                    None if full else monitor)
                write_csv(output / name / f'{scope}_{completed:07d}.csv', rows)
                metrics = [key for key in rows[0] if key not in ('image_id', 'snr_db', 'interface')]
                means = {key: float(np.mean([row[key] for row in rows])) for key in metrics}
                for snr in base['channel']['snrs_db']:
                    subset = [row for row in rows if row['snr_db'] == snr]
                    summaries.append({'additional_step': completed, 'arm': name, **definition, 'scope': scope,
                        'snr_db': snr, 'source_images': len(subset),
                        **{key: float(np.mean([row[key] for row in subset])) for key in metrics}})
                if full:
                    path = output / name / 'checkpoints' / f'additional_{completed:07d}.pt'
                    torch.save({'model': networks[name].state_dict(), **definition, 'additional_step': completed}, path)
                    candidate = {'step': completed, 'scope': 'full', 'source_images': 1000,
                        'checkpoint': str(path.relative_to(output)), 'checkpoint_sha256': sha256(path), **means}
                    selected[name] = choose(selected[name], candidate)
                timings['calibration'][name] += time.perf_counter() - tick
                print(f'{scope} {name} additional={completed} LPIPS={means["lpips"]:.6f} PSNR={means["psnr_db"]:.5f} BER={means["bit_error_rate"]:.6f}', flush=True)
            calibrated.append(completed)
            persist()

        calibrate_point()
        persist()
        status('TRAINING_INTERFACES')
        for batch in paired_batches(base, 20000, study['parent_step'] + completed, study['parent_step'] + until):
            if batch['step'] != study['parent_step'] + completed:
                raise RuntimeError('paired data position changed')
            inputs = batch_inputs(population, batch, device)
            for name, definition in definitions.items():
                network, optimizer = networks[name], optimizers[name]
                network.train().zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                totals = {key: 0. for key in ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate',
                    'feature_mse', 'feature_abs_mean', 'feature_saturation_fraction', 'logit_abs_mean')}
                power_error = 0.
                for index in range(4):
                    part = {key: value[index:index + 1] for key, value in inputs.items()}
                    result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
                    objective, components, diagnostics = interface_losses(result, part['source_fq'], part['images'], decoder, perceptual, study['weights'])
                    if not bool(torch.isfinite(objective)):
                        raise FloatingPointError('nonfinite interface objective')
                    (objective / 4).backward()
                    totals['loss'] += float(objective.detach()) / 4
                    for key, value in components.items():
                        totals[key] += float(value.detach().sum()) / 4
                    for key in ('bit_error_rate', 'feature_mse', 'feature_abs_mean', 'feature_saturation_fraction', 'logit_abs_mean'):
                        totals[key] += float(diagnostics[key].detach().sum()) / 4
                    power_error = max(power_error, float((result['transmitted'].detach().square().sum(-1).mean(-1) - 2).abs().max()))
                    del result, objective, components, diagnostics, part
                norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True))
                if power_error > 1e-5:
                    raise RuntimeError('physical energy constraint changed')
                optimizer.step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                timings['training'][name] += seconds
                histories[name].append({'additional_step': completed + 1, 'global_data_step': batch['step'] + 1,
                    'Adam_step': completed + 1, 'batch_sha256': batch['fingerprint'], 'gradient_norm': norm,
                    'step_seconds': seconds, 'power_max_error': power_error,
                    'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), **totals})
            completed += 1
            if completed == 1 or completed % study['training']['autosave_every'] == 0:
                persist()
                status('TRAINING_INTERFACES')
                print(f'paired interface updates {completed}/{until} ' + ' '.join(f'{name}={histories[name][-1]["loss"]:.6f}' for name in definitions), flush=True)
            calibrate_point()
        if completed != until or completed not in calibrated:
            raise RuntimeError('incomplete interface milestone')
        for name, optimizer in optimizers.items():
            if len(histories[name]) != completed or any(int(value['step']) != completed for value in optimizer.state.values()):
                raise RuntimeError('unequal retained history or Adam counters')
        if frozen != {'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('frozen visual parameters changed')
        verify_sources(metadata['source_hashes'])
        persist()
        milestones = output / 'milestones'
        milestones.mkdir(exist_ok=True)
        checkpoint = milestones / f'additional_{completed:07d}_optimizer.pt'
        os.link(output / 'resume.pt', checkpoint)
        write_json(milestones / f'additional_{completed:07d}.json', {'status': 'INTERFACE_MILESTONE_COMPLETE',
            'completed_local': now(), 'additional_updates_per_arm': completed, 'selected': selected,
            'definitions': definitions, 'frozen': frozen, 'source_hashes': metadata['source_hashes'], 'timings': timings,
            'optimizer_checkpoint_sha256': sha256(checkpoint), 'checkpoint': str(checkpoint),
            'observed_wall_GPU_hours': (elapsed_prior + time.time() - started) / 3600, 'research_goal_complete': False})
        status('INTERFACE_MILESTONE_READY_FOR_REVIEW_NOT_RESEARCH_COMPLETE')
    except BaseException as error:
        status('INTERFACE_FAILED_OR_INTERRUPTED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
