"""Zero-update real GPU verification of R3 functions, image gradients and throughput."""

import argparse
import json
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT.parent / 'wetok-comm-v2-20260912/scripts')]

import torch

from evaluate_interfaces import require_uncontended_gpu
from innovation_comm.hardware import gpu_telemetry
from joint_sender.runtime import backward_paired_batch, observation
from vector_control.common import check_initial_models, initial_system, load_parent, output_path, qualified_reference, settings
from wetok_comm.common import artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.native import FrozenWeTok
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, r2, original, grid, reference, base, parent_record = settings()
    if not arguments.execute:
        print('PLAN ONLY: real GPU zero-update initial/image-gradient profile; no training or development inference')
        return
    require_uncontended_gpu()
    configure_torch()
    qualification_path = output_path(config, 'preparation') / 'initialization.json'
    qualification = json.loads(qualification_path.read_text())
    verify_sources(qualification['source_hashes'])
    qualified = qualified_reference(config, r2, grid)
    if qualification['status'] != 'R3_ORIGINAL_INITIALIZATION_AND_COMPLETE_HISTORY_PASS' or qualification['reference_bindings'] != qualified['bindings']:
        raise RuntimeError('CPU initialization/history qualification is missing or stale')
    device = torch.device('cuda:0')
    parent = load_parent(reference, base, device)
    model = initial_system(parent, reference, device).eval()
    control = check_initial_models(parent, model, reference, qualified, config)
    decoder, perceptual = FrozenWeTok(device, 'decoder'), load_lpips(device)
    frozen = {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}
    if frozen != qualified['frozen']:
        raise RuntimeError('GPU profile changed frozen visual/metric parameters')
    batch = next(paired_batches(base, 20000, 7000, 7001))
    inputs = batch_inputs(read_population(base, 'train'), batch, device)
    with torch.no_grad():
        signal, received = observation(model, inputs)
        old_signal, old_received = observation(control, inputs)
        actual, expected = model.receive(received, inputs['snrs']), control.receive(received, inputs['snrs'])
        torch.testing.assert_close(signal, old_signal, atol=0, rtol=0)
        torch.testing.assert_close(received, old_received, atol=0, rtol=0)
        torch.testing.assert_close(actual['receiver_features'], expected['receiver_features'], atol=0, rtol=0)
        actual_image = decoder.decode(actual['receiver_features'])
        expected_image = decoder.decode(expected['receiver_features'])
        torch.testing.assert_close(actual_image, expected_image, atol=0, rtol=0)
    model_before = module_sha256(model)
    hardware = [gpu_telemetry()]
    results = {}
    for label, weights in (('image_only', {**config['unchanged_loss'], 'bits': 0., 'state': 0.}), ('registered_loss', config['unchanged_loss'])):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = backward_paired_batch(model, inputs, decoder, perceptual, weights)
        torch.cuda.synchronize()
        results[label] = {**metrics, 'batch_seconds_without_optimizer_update': time.perf_counter() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        if any(metrics[key] <= 0 for key in ('encoder_gradient_norm', 'receiver_gradient_norm', 'waveform_gradient_norm')):
            raise RuntimeError('real image loss did not reach trainable communication parameters')
        if metrics['paired_standard_noise_sha256'] != qualified['trace'][0]['paired_standard_noise_sha256']:
            raise RuntimeError('GPU profile did not use the original paired first noise')
        if label == 'registered_loss' and any(abs(metrics[key] - qualified['trace'][0][key]) > 1e-6
            for key in ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate')):
            raise RuntimeError('initial real losses differ from the original residual first update')
    require_uncontended_gpu()
    hardware.append(gpu_telemetry())
    if model_before != module_sha256(model) or frozen != {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)}:
        raise RuntimeError('zero-update profile changed model parameters')
    output = output_path(config, 'profile')
    output.mkdir(parents=True, exist_ok=False)
    hashes = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/protocol.md',
        *sorted((EXPERIMENT / 'src/vector_control').glob('*.py'))])
    write_json(output / 'hardware.json', hardware)
    record = {'status': 'R3_REAL_INITIAL_FUNCTION_AND_IMAGE_GRADIENT_PROFILE_PASS', 'local_time': now(),
        'qualification_sha256': sha256(qualification_path), 'reference_bindings': qualified['bindings'],
        'initial_model_sha256': model_before, 'frozen': frozen, 'initial_pixel_max_error': 0., 'initial_waveform_max_error': 0.,
        'optimizer_updates': 0, 'results': results, 'source_hashes': hashes, 'new_development_inference': False,
        'research_goal_complete': False, 'output_hashes': artifact_hashes(output)}
    write_json(output / 'profile.json', record)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
