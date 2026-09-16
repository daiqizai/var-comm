"""Real image gradients and observed cost for Joint grid controls, with zero updates."""

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
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch
import yaml

from evaluate_interfaces import require_uncontended_gpu
from grid_controls.common import completed_joint_reference, initial_system, load_parent, output_path, settings
from innovation_comm.hardware import gpu_telemetry
from innovation_comm.runtime import actual_observation
from joint_sender.runtime import backward_paired_batch
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, paired_batches, read_population


def component_gradients(system, inputs, decoder, perceptual, weights):
    groups = {'encoder': [parameter for name, parameter in system.named_parameters() if name.startswith('encoder.')],
              'receiver_including_fusion': [parameter for name, parameter in system.named_parameters() if not name.startswith('encoder.')]}
    vectors, norms = {}, {}
    for term in weights:
        active = {name: value if name == term else 0. for name, value in weights.items()}
        backward_paired_batch(system, inputs, decoder, perceptual, active)
        vectors[term] = {name: torch.cat([(parameter.grad.detach() if parameter.grad is not None else
            torch.zeros_like(parameter)).flatten().cpu() for parameter in parameters]) for name, parameters in groups.items()}
        norms[term] = {name: float(vector.norm()) for name, vector in vectors[term].items()}
        if not all(np.isfinite(value) for value in norms[term].values()):
            raise FloatingPointError('nonfinite component gradients')
    cosines = []
    for position, first in enumerate(weights):
        for second in list(weights)[position + 1:]:
            for group in groups:
                denominator = norms[first][group] * norms[second][group]
                cosine = float(torch.dot(vectors[first][group], vectors[second][group]) / denominator) if denominator > 0 else None
                cosines.append({'first': first, 'second': second, 'parameter_group': group, 'cosine': cosine})
    return {'weighted_norms': norms, 'cosines': cosines, 'used_for_loss_selection': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: real image-only E/R gradients and observed cost; no optimizer updates or new development')
        return
    require_uncontended_gpu()
    configure_torch()
    config, original, reference, base, parent_record = settings()
    references = completed_joint_reference(config, original, reference)
    initialization_path = output_path(config, 'preparation') / 'initialization.json'
    initialization = json.loads(initialization_path.read_text())
    if initialization['status'] != 'CPU_GRID_INITIALIZATION_PASS_NOT_ACTIVATED' or initialization['optimizer_updates'] != 0:
        raise RuntimeError('CPU preparation is incomplete')
    verify_sources(initialization['source_hashes'])
    output = output_path(config, 'profile')
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), EXPERIMENT / 'scripts/train.py', EXPERIMENT / 'configs/study.yaml',
        EXPERIMENT / 'docs/protocol.md', EXPERIMENT / 'docs/activation_decision_20260914.md',
        *sorted((EXPERIMENT / 'src/grid_controls').glob('*.py')), JOINT / 'src/joint_sender/runtime.py',
        INNOVATION / 'src/innovation_comm/model.py', INNOVATION / 'src/innovation_comm/runtime.py',
        BASE / 'scripts/evaluate_interfaces.py']
    sources = snapshot(output, files)
    started = time.monotonic()

    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'optimizer_updates': 0, 'research_goal_complete': False, **extra})

    try:
        status('LOADING_GRID_IMAGE_GRADIENT_PROFILE')
        device = torch.device('cuda:0')
        parent = load_parent(reference, base, device)
        systems = {name: initial_system(parent, name, reference, device) for name in config['variants']}
        decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
        before = {name: module_sha256(system) for name, system in systems.items()}
        if before != {row['variant']: row['model_state_sha256'] for row in initialization['variants']}:
            raise RuntimeError('GPU initialization differs from the real CPU parent')
        population = read_population(base, 'train')
        batch = next(paired_batches(base, 20000, 7000, 7001))
        inputs = batch_inputs(population, batch, device)
        signal, observed = actual_observation(parent, inputs)
        tolerance = yaml.safe_load((INNOVATION / 'configs/evaluation.yaml').read_text())['parent_signal_replay_tolerance']
        hardware_before = gpu_telemetry()
        results = {}
        for name, system in systems.items():
            status('CHECKING_REAL_GRID_IMAGE_GRADIENTS', variant=name)
            with torch.no_grad():
                difference = float((system.transmit(inputs['source_fq'], inputs['snrs']) - signal).abs().max())
                if difference > tolerance:
                    raise RuntimeError('initial waveform exceeds inherited tolerance')
                control = initial_system(parent, name, reference, device, update_sender=False).eval()
                expected_image = decoder.decode(control.receive(observed[:1], inputs['snrs'][:1])['receiver_features'])
                actual_image = decoder.decode(system.eval().receive(observed[:1], inputs['snrs'][:1])['receiver_features'])
                pixel_error = float((expected_image - actual_image).abs().max())
                if pixel_error != 0:
                    raise RuntimeError('initial same-y image mapping changed with sender update policy')
                del control, expected_image, actual_image
            image_only = backward_paired_batch(system, inputs, decoder, perceptual,
                {'mse': 1., 'lpips': .01, 'bits': 0., 'state': 0.})
            if any(not np.isfinite(image_only[key]) or image_only[key] <= 0 for key in
                   ('encoder_gradient_norm', 'receiver_gradient_norm', 'waveform_gradient_norm')):
                raise RuntimeError('real image loss did not reach both E and R')
            if any(parameter.grad is not None for model in (decoder.codec, perceptual, parent) for parameter in model.parameters()):
                raise RuntimeError('a frozen model accumulated parameter gradients')
            components = component_gradients(system, inputs, decoder, perceptual, config['unchanged_loss'])
            for warmup in range(2):
                backward_paired_batch(system, inputs, decoder, perceptual, config['unchanged_loss'])
            durations, peaks = [], []
            for repeat in range(3):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                total = backward_paired_batch(system, inputs, decoder, perceptual, config['unchanged_loss'])
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - tick)
                peaks.append(torch.cuda.max_memory_allocated())
            results[name] = {'initial_waveform_max_error': difference, 'same_y_initial_pixel_max_error': pixel_error,
                'image_only': image_only, 'gradient_components': components, 'joint_loss': total,
                'batch4_forward_backward_seconds': float(np.mean(durations)), 'peak_allocated_bytes': max(peaks),
                'optimized_parameters': sum(parameter.numel() for parameter in system.parameters() if parameter.requires_grad)}
            system.zero_grad(set_to_none=True)
        after = {name: module_sha256(system) for name, system in systems.items()}
        if before != after or frozen != {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('zero-update grid profile changed parameters')
        require_uncontended_gpu()
        verify_sources(sources)
        write_json(output / 'profile.json', {'status': 'GRID_IMAGE_GRADIENT_PROFILE_PASS', 'completed_local': now(),
            'results': results, 'initialization_sha256': sha256(initialization_path), 'source_hashes': sources,
            'reference_hashes': references['hashes'], 'model_hashes_before': before, 'model_hashes_after': after,
            'frozen': frozen, 'data_batch_sha256': batch['fingerprint'], 'first_batch_global_index': batch['step'],
            'hardware_before': hardware_before, 'hardware_after': gpu_telemetry(), 'optimizer_updates': 0,
            'profile_wall_seconds': time.monotonic() - started, 'new_development_access': False,
            'forward_backward_hours_per1000_both_arms': sum(row['batch4_forward_backward_seconds'] for row in results.values()) * 1000 / 3600,
            'cost_excludes_Adam_data_checkpoint_and_calibration': True, 'research_goal_complete': False})
        status('GRID_IMAGE_GRADIENT_PROFILE_PASS_NOT_RESEARCH_COMPLETE')
        print(json.dumps(results, indent=2))
    except BaseException as error:
        status('GRID_PROFILE_FAILED_NO_TRAINING', error=repr(error))
        raise


if __name__ == '__main__':
    main()
