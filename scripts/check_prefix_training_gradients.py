#!/usr/bin/env python3
"""Normal engineering checks: hard-forward equality, real image gradients and frozen weights."""

import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

from var_comm.learned_prefix import empty_cache
from var_comm.next_scale_prior import state_sha256
from var_comm.prefix_learning_support import build_codec, communication_forward, image_losses, load_frozen_training_models
from var_comm.prefix_training_data import HeaderProtocol, read_image_population
from var_comm.progressive import complete_image, split_prefix
from var_comm.study import artifact_hashes, create_output, snapshot, verify_artifacts, write_json


def main():
    config = yaml.safe_load((ROOT / 'configs/learned_prefix_jscc.yaml').read_text())
    verify_artifacts(ROOT / config['data_cache'], 'completion.json')
    output = create_output(ROOT / config['outputs']['selfcheck'])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda:0')
    vae, var, perceptual = load_frozen_training_models(config, device)
    backbones = {'vae': vae, 'var': var, 'lpips': perceptual}
    before = {name: state_sha256(model) for name, model in backbones.items()}
    images, labels, _ids, _bindings = read_image_population('calibration', verify=False)
    with np.load(ROOT / config['data_cache'] / 'calibration.npz', allow_pickle=False) as cache:
        tokens = torch.tensor(cache['tokens'][:2, 0].astype(np.int64), device=device)
    targets = images[:2].to(device).float() / 255
    snrs = torch.tensor([4.0, 7.0], device=device)
    random = torch.Generator().manual_seed(2026090716)
    noise = torch.randn((2, 3060, 2), generator=random)
    decoded, usable, _false = HeaderProtocol().decode(labels[:2].numpy(), snrs.cpu().numpy(), noise[:, :68].numpy())
    received_labels = torch.tensor(decoded, dtype=torch.long, device=device)
    valid = torch.tensor(usable, device=device)
    data_noise = noise[:, 68:].to(device)
    results, parameters = [], []
    try:
        for variant in config['variants']:
            torch.manual_seed(config['training']['initialization_seed'])
            codec = build_codec(config, vae, variant, device)
            parameters.append(sum(parameter.numel() for parameter in codec.parameters()))
            initial_encoder = codec.encoder.mixing.detach().clone()
            initial_decoder = codec.reader.backprojection.detach().clone()
            optimizer = torch.optim.AdamW(codec.parameters(), lr=0.0001)
            for _step in range(2):
                codec.train()
                prediction = communication_forward(codec, tokens, received_labels, snrs, data_noise, valid, vae, var)
                loss, _components = image_losses(prediction, targets, tokens, valid, perceptual, config['training'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                del prediction, loss
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            prediction = communication_forward(codec, tokens, received_labels, snrs, data_noise, valid, vae, var)
            loss, components = image_losses(prediction, targets, tokens, valid, perceptual, config['training'])
            selected_parameters = [codec.encoder.mixing, codec.reader.backprojection]
            image_gradients = {}
            for name in ('mse', 'lpips'):
                gradients = torch.autograd.grad(components[name].mean(), selected_parameters, retain_graph=True)
                norms = [float(gradient.norm()) for gradient in gradients]
                assert all(np.isfinite(norms)) and min(norms) > 0
                image_gradients[name] = norms
            weights = torch.linspace(-1, 1, prediction['suffix_embeddings'][-1].numel(), device=device).reshape_as(prediction['suffix_embeddings'][-1])
            suffix_probe = (prediction['suffix_embeddings'][-1] * weights).sum()
            prefix_gradient = torch.autograd.grad(suffix_probe, prediction['prefix_embeddings'][0], retain_graph=True)[0]
            assert prefix_gradient.isfinite().all() and prefix_gradient.norm() > 0
            hard_image = prediction['image'].detach().clone()
            hard_indices = prediction['indices'].detach().clone()
            power_error = float((prediction['symbols'].square().sum(-1).mean(-1) - 2).abs().max())
            assert power_error < 1e-5
            loss.backward()
            assert codec.encoder.mixing.grad.isfinite().all() and codec.reader.backprojection.grad.isfinite().all()
            elapsed = time.perf_counter() - tick
            peak = torch.cuda.max_memory_allocated(device)
            del prediction, components, loss, prefix_gradient
            codec.eval()
            with torch.no_grad():
                evaluated = communication_forward(codec, tokens, received_labels, snrs, data_noise, valid, vae, var)
            assert torch.equal(evaluated['indices'], hard_indices)
            pixel_error = float((evaluated['image'] - hard_image).abs().max())
            assert pixel_error < 1e-6
            official_errors = []
            for index in range(2):
                prefix = split_prefix(hard_indices[index].cpu().numpy(), 8)
                official = complete_image(vae, var, prefix, int(received_labels[index]), device)
                official_errors.append(float(np.max(np.abs(official - evaluated['image'][index].cpu().numpy()))))
            assert max(official_errors) < 1e-4
            assert not torch.equal(initial_encoder, codec.encoder.mixing) and not torch.equal(initial_decoder, codec.reader.backprojection)
            if variant == 'next_scale':
                codec.train()
                mixed = communication_forward(codec, tokens, received_labels, snrs, data_noise, valid, vae, var,
                                               teacher_tokens=tokens, teacher_mask=torch.tensor([True, False], device=device))
                mixed_loss, _components = image_losses(mixed, targets, tokens, valid, perceptual, config['training'])
                mixed_loss.backward()
                del mixed, mixed_loss, _components
            assert empty_cache(var)
            results.append({'variant': variant, 'trainable_parameters': parameters[-1], 'image_only_gradients_encoder_decoder': image_gradients,
                            'VAR_suffix_to_prefix_gradient_nonzero': True, 'train_eval_max_pixel_difference': pixel_error,
                            'official_receiver_max_pixel_difference': max(official_errors), 'power_max_error': power_error,
                            'batch2_multi_backward_seconds': elapsed, 'batch2_peak_allocated_bytes': peak})
            del codec, optimizer, evaluated
            torch.cuda.empty_cache()
        assert parameters[0] == parameters[1]
        assert before == {name: state_sha256(model) for name, model in backbones.items()}
        assert all(not parameter.requires_grad and parameter.grad is None for model in backbones.values() for parameter in model.parameters())
        sources = snapshot(output, [Path(__file__), ROOT / 'src/var_comm/learned_prefix.py', ROOT / 'src/var_comm/prefix_learning_support.py'])
        write_json(output / 'selfcheck.json', {'status': 'IMAGE_GRADIENT_AND_FROZEN_MODEL_CHECK_PASS', 'results': results,
                                              'frozen_state_sha256': before, 'DINO_loaded_or_optimized': False,
                                              'source_hashes': sources, 'output_hashes': artifact_hashes(output)})
        print(results, flush=True)
    except Exception as error:
        write_json(output / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
