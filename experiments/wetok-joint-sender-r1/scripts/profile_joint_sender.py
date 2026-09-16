"""Real image-only E/R gradient and training-cost checks, after the owned GPU becomes free."""

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
import yaml

from evaluate_interfaces import require_uncontended_gpu
from innovation_comm.hardware import gpu_telemetry
from innovation_comm.inference import receive_for_image, verify_pruned_result
from innovation_comm.runtime import actual_observation
from joint_sender.common import initial_system, load_parent, output_path, settings
from joint_sender.runtime import backward_paired_batch
from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, paired_batches, read_population


def component_gradients(system, inputs, decoder, perceptual, weights):
    vectors, norms = {}, {}
    modules = {'encoder': system.encoder, 'receiver': system.receiver}
    for term in weights:
        active = {name: (value if name == term else 0.) for name, value in weights.items()}
        backward_paired_batch(system, inputs, decoder, perceptual, active)
        vectors[term] = {name: torch.cat([(parameter.grad.detach() if parameter.grad is not None else
            torch.zeros_like(parameter)).flatten().cpu() for parameter in module.parameters()]) for name, module in modules.items()}
        norms[term] = {name: float(value.norm()) for name, value in vectors[term].items()}
        if not all(np.isfinite(value) for value in norms[term].values()):
            raise FloatingPointError('nonfinite per-component gradient diagnostic')
    cosines = []
    for index, first in enumerate(weights):
        for second in list(weights)[index + 1:]:
            for group in modules:
                denominator = norms[first][group] * norms[second][group]
                cosine = float(torch.dot(vectors[first][group], vectors[second][group]) / denominator) if denominator > 0 else None
                cosines.append({'first': first, 'second': second, 'parameter_group': group, 'cosine': cosine})
    return {'weighted_term_gradient_norms': norms, 'pairwise_cosines': cosines,
            'scope': 'one_fixed_training_batch_monitoring_not_loss_selection_or_performance_evidence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: initial waveform, image-only E/R gradients and full-batch source graph cost; zero optimizer updates')
        return
    require_uncontended_gpu()
    configure_torch()
    config, reference, base, parent_milestone = settings()
    initialization_path = output_path(config, 'preparation') / 'initialization.json'
    initialization = json.loads(initialization_path.read_text())
    if initialization['status'] != 'JOINT_SENDER_INITIAL_WEIGHT_POLICY_CPU_CHECK_PASS' or initialization['optimizer_updates'] != 0:
        raise RuntimeError('CPU initial-weight/gradient-policy preparation has not passed')
    verify_sources(initialization['source_hashes'])
    tolerance_file = INNOVATION / 'configs/evaluation.yaml'
    tolerance = yaml.safe_load(tolerance_file.read_text())['parent_signal_replay_tolerance']
    if tolerance != 1e-6:
        raise RuntimeError('inherited initial-waveform tolerance changed')
    output = output_path(config, 'profile')
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__), EXPERIMENT / 'scripts/train.py', EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/protocol.md',
        EXPERIMENT / 'docs/numerical_gradient_note.md', tolerance_file, *sorted((EXPERIMENT / 'src/joint_sender').glob('*.py')),
        INNOVATION / 'src/innovation_comm/model.py', INNOVATION / 'src/innovation_comm/runtime.py',
        INNOVATION / 'src/innovation_comm/hardware.py', INNOVATION / 'src/innovation_comm/inference.py', BASE / 'scripts/evaluate_interfaces.py'])
    started = time.time()
    def status(value, **extra):
        write_json(output / 'status.json', {'status': value, 'pid': os.getpid(), 'local_time': now(),
            'optimizer_updates': 0, 'research_goal_complete': False, **extra})
    try:
        status('LOADING_JOINT_GRADIENT_PROFILE')
        device = torch.device('cuda:0')
        parent = load_parent(reference, base, device)
        systems = {variant: initial_system(parent, variant, reference, device) for variant in config['variants']}
        decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
        frozen = {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
        before = {name: module_sha256(system) for name, system in systems.items()}
        if before != {name: record['model_sha256'] for name, record in initialization['models'].items()}:
            raise RuntimeError('GPU profile weights are not the shared initial weights')
        population = read_population(base, 'train')
        batch = next(paired_batches(base, 20000, 7000, 7001))
        inputs = batch_inputs(population, batch, device)
        signal, observed = actual_observation(parent, inputs)
        hardware_before = gpu_telemetry()
        results = {}
        for variant, system in systems.items():
            status('CHECKING_JOINT_IMAGE_GRADIENTS', variant=variant)
            with torch.no_grad():
                candidate = system.transmit(inputs['source_fq'], inputs['snrs'])
                waveform_error = float((candidate - signal).abs().max())
                if waveform_error > tolerance:
                    raise RuntimeError(f'initial Joint waveform exceeds registered replay tolerance: {waveform_error}')
                control = initial_system(parent, variant, reference, device, update_sender=False).eval()
                control_image = decoder.decode(control.receive(observed[:1], inputs['snrs'][:1])['receiver_features'])
                joint_image = decoder.decode(system.eval().receive(observed[:1], inputs['snrs'][:1])['receiver_features'])
                pixel_error = float((control_image - joint_image).abs().max())
                if pixel_error != 0:
                    raise RuntimeError('on identical y the starting receivers are not the same image mapping')
                optimized = receive_for_image(control, observed[:1], inputs['snrs'][:1])
                prune_error = verify_pruned_result(control, optimized, observed[:1], inputs['snrs'][:1])
                del control, control_image, joint_image, candidate
            image_weights = {'mse': 1., 'lpips': .01, 'bits': 0., 'state': 0.}
            image_gradient = backward_paired_batch(system, inputs, decoder, perceptual, image_weights)
            if any(image_gradient[key] <= 0 or not np.isfinite(image_gradient[key]) for key in
                   ('encoder_gradient_norm', 'receiver_gradient_norm', 'waveform_gradient_norm')):
                raise RuntimeError('real image-only loss does not reach both communication sides')
            if any(value.grad is not None for value in decoder.codec.parameters()) or any(value.grad is not None for value in perceptual.parameters()):
                raise RuntimeError('profile updated the frozen image/metric model gradient boundary')
            gradient_components = component_gradients(system, inputs, decoder, perceptual, config['unchanged_loss'])
            for warmup in range(2):
                backward_paired_batch(system, inputs, decoder, perceptual, config['unchanged_loss'])
            durations, peaks = [], []
            for repeat in range(3):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                result = backward_paired_batch(system, inputs, decoder, perceptual, config['unchanged_loss'])
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - tick)
                peaks.append(torch.cuda.max_memory_allocated())
            results[variant] = {'waveform_max_error': waveform_error, 'same_y_receiver_pixel_max_error': pixel_error,
                'no_history_prune_feature_max_error': prune_error, 'image_only': image_gradient,
                'gradient_components': gradient_components, 'joint_loss': result, 'batch4_forward_backward_seconds': float(np.mean(durations)),
                'peak_allocated_bytes': max(peaks), 'communication_trainable': sum(value.numel() for value in system.parameters() if value.requires_grad)}
            system.zero_grad(set_to_none=True)
        after = {name: module_sha256(system) for name, system in systems.items()}
        if before != after or frozen != {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
            raise RuntimeError('zero-update Joint profile changed model state')
        require_uncontended_gpu()
        hardware_after = gpu_telemetry()
        verify_sources(source_hashes)
        status('JOINT_IMAGE_GRADIENT_PROFILE_PASS_NOT_RESEARCH_COMPLETE')
        write_json(output / 'profile.json', {'status': 'JOINT_IMAGE_GRADIENT_PROFILE_PASS', 'completed_local': now(),
            'results': results, 'initialization_sha256': sha256(initialization_path), 'source_hashes': source_hashes,
            'model_hashes_before': before, 'model_hashes_after': after, 'frozen': frozen,
            'data_batch_sha256': batch['fingerprint'], 'first_batch_global_index': batch['step'],
            'completed_global_updates_before_trial': 7000,
            'estimated_three_arm_forward_backward_GPU_hours_per1000': sum(row['batch4_forward_backward_seconds'] for row in results.values()) * 1000 / 3600,
            'cost_scope': 'forward_backward_only_excludes_Adam_checkpoint_data_loading_and_calibration',
            'hardware_before': hardware_before, 'hardware_after': hardware_after,
            'profile_wall_seconds': time.time() - started, 'optimizer_updates': 0, 'new_development_access': False,
            'final_control_qualification_required_before_training': True, 'research_goal_complete': False})
        print(json.dumps(results, indent=2))
    except BaseException as error:
        status('JOINT_PROFILE_FAILED_NO_TRAINING', error=repr(error))
        raise


if __name__ == '__main__':
    main()
