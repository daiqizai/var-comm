"""Real frozen-Decoder gradient checks and no-update training profiling."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np
import torch

from wetok_comm.common import WORKSPACE, artifact_hashes, configure_torch, now, output_path, settings, sha256, snapshot, verify_sources, write_json
from wetok_comm.model import communicate
from wetok_comm.native import FrozenWeTok
from wetok_comm.objective import losses
from wetok_comm.training import batch_inputs, load_lpips, module_sha256, new_network, paired_batches, read_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/study.yaml')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config = settings(arguments.config)
    if not arguments.execute:
        print('PLAN ONLY: frozen Decoder gradient/identity/throughput checks; no optimizer steps')
        return
    configure_torch()
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < 12000:
        raise RuntimeError('GPU not available for the registered fp32 gradient check')
    output = output_path(config, 'profile')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), arguments.config, EXPERIMENT / 'docs/protocol.md',
        *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])
    write_json(output / 'status.json', {'status': 'LOADING_FROZEN_DECODER', 'pid': os.getpid(), 'local_time': now(), 'optimizer_updates': 0})
    started = time.perf_counter()
    try:
        device = torch.device('cuda:0')
        population = read_population(config, 'train')
        native = FrozenWeTok(device, 'decoder')
        perceptual = load_lpips(device)
        frozen_before = {'codec': module_sha256(native.codec), 'lpips': module_sha256(perceptual)}
        batch = next(paired_batches(config, 20000, 0, 1))
        probe = batch_inputs(population, batch, device)
        latent = probe['source_fq'][:1].detach().clone().requires_grad_(True)
        image = native.decode(latent)
        (image - probe['images'][:1]).square().mean().backward()
        input_gradient = float(latent.grad.norm())
        if not np.isfinite(input_gradient) or input_gradient <= 0 or any(value.grad is not None for value in native.codec.parameters()):
            raise RuntimeError('frozen Decoder does not preserve a valid input gradient')
        del image, latent
        results = {}
        initial_hashes = {}
        for arm in config['arms']:
            network = new_network(config, arm, device)
            initial_hashes[arm] = module_sha256(network)
            current = {key: value[:1] for key, value in probe.items()}
            network.train()
            result = communicate(network, current['source_fq'], current['snrs'], current['noise'])
            objective, components, diagnostic = losses(result, current['source_fq'], current['images'], native, perceptual,
                                                       config['training']['joint_weights'])
            reference_image = result['image'].detach().clone()
            image_objective = (components['mse'] + .01 * components['lpips']).mean()
            image_gradients = {}
            for name, module in (('encoder', network.encoder), ('receiver', network.receiver)):
                gradients = torch.autograd.grad(image_objective, tuple(module.parameters()), retain_graph=True, allow_unused=True)
                norm = sum(float(value.square().sum()) for value in gradients if value is not None) ** .5
                if not np.isfinite(norm) or norm <= 0:
                    raise RuntimeError(f'{arm} image gradient missing at {name}')
                image_gradients[name] = norm
            del result, objective, components, diagnostic, image_objective, gradients
            with torch.no_grad():
                evaluation = communicate(network.eval(), current['source_fq'], current['snrs'], current['noise'])
                evaluated_image = native.decode(evaluation['native_fq'])
                error = float((evaluated_image - reference_image).abs().max())
            if error != 0:
                raise RuntimeError(f'{arm} training hard output does not match inference: {error}')
            del evaluation, evaluated_image, reference_image
            timings = {}
            for phase in ('representation', 'joint'):
                weights = config['training'][phase + '_weights']
                measured, peaks = [], []
                calls = {'decoder': 0}
                def record_call(module, inputs, outputs):
                    calls['decoder'] += 1
                hook = native.codec.decoder.register_forward_hook(record_call)
                try:
                    for repeat in range(4):
                        network.train().zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        tick = time.perf_counter()
                        for offset in range(4):
                            part = {key: value[offset:offset + 1] for key, value in probe.items()}
                            result = communicate(network, part['source_fq'], part['snrs'], part['noise'])
                            objective, components, diagnostic = losses(result, part['source_fq'], part['images'], native, perceptual, weights)
                            (objective / 4).backward()
                            if float((result['transmitted'].detach().square().sum(-1).mean(-1) - 2).abs().max()) > 1e-5:
                                raise RuntimeError('physical energy budget violated')
                            del result, objective, components, diagnostic, part
                        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                        torch.cuda.synchronize()
                        if repeat:
                            measured.append(time.perf_counter() - tick)
                            peaks.append(torch.cuda.max_memory_allocated())
                finally:
                    hook.remove()
                if phase == 'representation' and calls['decoder']:
                    raise RuntimeError('representation-only stage invoked the image Decoder')
                timings[phase] = {'effective_batch_seconds': float(np.median(measured)), 'peak_allocated_bytes': max(peaks),
                                  'effective_batch_size': 4, 'microbatch': 1, 'decoder_forward_calls': calls['decoder']}
            network.zero_grad(set_to_none=True)
            if module_sha256(network) != initial_hashes[arm]:
                raise RuntimeError('no-update profile changed communication parameters')
            results[arm] = {'parameters': sum(value.numel() for value in network.parameters()), 'timings': timings,
                            'hard_forward_max_image_error': error, 'image_gradients': image_gradients}
            print(json.dumps({'arm': arm, **results[arm]}), flush=True)
            del network
            torch.cuda.empty_cache()
        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError('comparison arms do not have identical initial parameters')
        after = {'codec': module_sha256(native.codec), 'lpips': module_sha256(perceptual)}
        if after != frozen_before or any(value.grad is not None or value.requires_grad for value in native.codec.parameters()):
            raise RuntimeError('frozen visual parameters changed')
        verify_sources(sources)
        first = config['training']['first_milestone_updates']
        pre = config['training']['representation_updates']
        estimate = sum(pre * row['timings']['representation']['effective_batch_seconds'] + (first - pre) * row['timings']['joint']['effective_batch_seconds'] for row in results.values())
        write_json(output / 'status.json', {'status': 'NATIVE_GRADIENT_PROFILE_PASS', 'local_time': now(), 'optimizer_updates': 0})
        write_json(output / 'profile.json', {'status': 'NATIVE_GRADIENT_PROFILE_PASS', 'completed_local': now(), 'source_hashes': sources,
            'native_decoder_input_gradient': input_gradient, 'frozen_before': frozen_before, 'frozen_after': after,
            'initial_model_hashes': initial_hashes, 'arms': results, 'optimizer_updates': 0,
            'first_milestone_core_training_hours': estimate / 3600, 'estimate_excludes': ['optimizer.step', 'calibration', 'checkpoint and IO', 'final development evaluation'],
            'observed_profile_gpu_hours': (time.perf_counter() - started) / 3600, 'output_hashes': artifact_hashes(output)})
        print('NATIVE_GRADIENT_PROFILE_PASS', flush=True)
    except BaseException as error:
        write_json(output / 'status.json', {'status': 'PROFILE_FAILED', 'local_time': now(), 'optimizer_updates': 0, 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
