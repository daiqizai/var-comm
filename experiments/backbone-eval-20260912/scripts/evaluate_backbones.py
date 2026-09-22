#!/usr/bin/env python3
"""No-training, image-paired source representation evaluation.

Run one adapter per subprocess: old VAR and XQ upstream both use ``models``.
The external wait_and_run supervisor owns GPU availability/exclusivity checks.
Model inference and metric inference are separate phases, so reported codec
memory excludes LPIPS/DINO resident weights. Images are evaluated as exact
float tensors, not PNG round trips. All outputs remain in this experiment.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PROJECT = EXPERIMENT.parents[1]
HISTORICAL = Path('/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc')
VAR_ROOT = Path('/home/liulu/projects/VAR-MAP-GATE0')
MANIFEST_SHA = '33d2a4f13eb97bb1d47fca25df04d9ca4b7ebf0fdcefe7d9497c5a6e4830c243'
OLD_PATCHES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
METRICS = ('psnr_db', 'ssim', 'lpips_alex', 'dino_cosine')
DEFAULTS = {
    'historical_source': str(HISTORICAL),
    'manifest': str(VAR_ROOT / 'results/vae_reconstruction/imagenet100_m6_to_m10_rate_sweep/selected_samples.json'),
    'manifest_sha256': MANIFEST_SHA,
    'old_source': str(VAR_ROOT / 'third_party/VAR'),
    'old_vae_checkpoint': str(VAR_ROOT / 'checkpoints/vae_ch160v4096z32.pth'),
    'old_vae_checkpoint_sha256': '7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c',
    'old_var_checkpoint': str(VAR_ROOT / 'checkpoints/var_d16.pth'),
    'old_var_checkpoint_sha256': '4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb',
    'fidelity_checkpoint': str(HISTORICAL / 'outputs/train/EXP-VAR-DECODER-ONLY-M89-001/checkpoints/best.pt'),
    'fidelity_checkpoint_sha256': '7b79b4b96f0b13ba5f77cf3cd4038ac1f0fe216e2b516e6a3a3ba57fb20d2802',
    'dino_source': str(VAR_ROOT / 'third_party/dinov2'),
    'dino_checkpoint': '/home/liulu/projects/CAP-VPR/artifacts/checkpoints/dinov2/dinov2_vits14_pretrain.pth',
    'dino_checkpoint_sha256': 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9',
    'output_root': str(EXPERIMENT / 'outputs'),
    'device': 'cuda:0', 'cpu_threads': 2, 'warmup_images': 1,
    'save_float_outputs': True,
    'xq_sampling': {'enabled': True, 'seeds': [0, 1, 2], 'cfg': 3.25},
}


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def json_write(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def resolve_config(path: Path) -> dict:
    data = copy.deepcopy(DEFAULTS)
    override = json.loads(path.read_text())
    # Flat paths are the public adapter API. Accept a paths wrapper for handoffs.
    data.update(override.get('paths', {}))
    data.update({key: value for key, value in override.items() if key != 'paths'})
    if 'old_var_source' in data:
        data['old_source'] = data['old_var_source']
    if isinstance(data.get('xq_sampling'), bool):
        data['xq_sampling'] = {'enabled': data['xq_sampling'],
            'seeds': data.get('xq_sampling_seeds', [0, 1, 2]),
            'cfg': data.get('xq_sampling_cfg', 3.25)}
    data['config_path'] = str(path.resolve())
    data['config_sha256'] = digest_file(path)
    return data


def conditions(model: str, config: dict) -> list[dict]:
    final_scale = None if model == 'wetok' else 10
    result = [{'name': 'full', 'kind': 'full', 'prefix_scales': final_scale,
               'mode': None, 'seed': None, 'cfg': None, 'primary': True}]
    if model not in ('old-official', 'xq'):
        return result
    for prefix in (8, 9):
        result.append({'name': f'p{prefix}_direct', 'kind': 'direct', 'prefix_scales': prefix,
                       'mode': None, 'seed': None, 'cfg': None, 'primary': True})
        result.append({'name': f'p{prefix}_argmax', 'kind': 'completion', 'prefix_scales': prefix,
                       'mode': 'argmax', 'seed': 0, 'cfg': 1.0, 'primary': True})
        sensitivity = config.get('xq_sampling', {})
        if model == 'xq' and sensitivity.get('enabled', True):
            for seed in sensitivity.get('seeds', [0, 1, 2]):
                result.append({'name': f'p{prefix}_sample_s{int(seed)}', 'kind': 'completion',
                    'prefix_scales': prefix, 'mode': 'sample', 'seed': int(seed),
                    'cfg': float(sensitivity.get('cfg', 3.25)), 'primary': False})
    return result


def truncate_tokens(tokens: Any, prefix: int, model: str) -> Any:
    """Receiver API boundary: there is no true suffix in the returned object."""
    if model == 'xq':
        if len(tokens) != 2 or any(len(branch) < prefix for branch in tokens):
            raise ValueError('expected XQ two-branch, scale-major token lists')
        return [list(branch[:prefix]) for branch in tokens]
    if model == 'old-official':
        if len(tokens) < prefix:
            raise ValueError('old VAR prefix longer than available token list')
        return list(tokens[:prefix])
    raise ValueError(f'completion is not authorized for {model}')


def expected_raw_bits(model: str, prefix: int | None) -> int:
    if model == 'wetok':
        return 8192
    if model == 'xq':
        return {8: 2424, 9: 3960, 10: 6864}[prefix]
    return {8: 3060, 9: 5088, 10: 8160}[prefix]


def check_token_metadata(meta: dict, model: str, prefix: int | None) -> dict:
    meta = dict(meta)
    actual = int(meta['raw_bits'])
    expected = expected_raw_bits(model, prefix)
    if actual != expected:
        raise RuntimeError(f'{model} prefix={prefix}: actual raw_bits={actual}, expected={expected}')
    if int(meta.get('index_count', meta.get('num_indices', 0))) <= 0:
        raise RuntimeError('token metadata must disclose positive actual index_count')
    if not meta.get('shapes'):
        raise RuntimeError('token metadata must disclose actual index shapes')
    return meta


def tensor_digest(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode())
    hasher.update(json.dumps(list(value.shape)).encode())
    hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def load_historical(config: dict):
    source = PROJECT
    sys.path.insert(0, str(source / 'src'))
    from cadsd_jscc import var_decoder_tuning as utilities
    path = source / 'scripts/evaluate_var_decoder_only.py'
    spec = importlib.util.spec_from_file_location('historical_frozen_metric_implementation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return utilities, module


@contextmanager
def preserve_nn_reset_parameters():
    """Old upstream builder globally disables initialization; never leak that."""
    import torch.nn as nn
    classes = (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm,
               nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
    saved = {cls: cls.reset_parameters for cls in classes}
    try:
        yield
    finally:
        for cls, method in saved.items():
            cls.reset_parameters = method


class OldAdapter:
    def __init__(self, config: dict, device, *, fidelity: bool, utilities):
        import torch
        self.torch = torch
        self.u = utilities
        self.var = None
        source = Path(config['old_source'])
        vae_path = utilities.require_sha(Path(config['old_vae_checkpoint']), config['old_vae_checkpoint_sha256'])
        if fidelity:
            self.vae = utilities.build_vae_only(source, vae_path, device)
            path = utilities.require_sha(Path(config['fidelity_checkpoint']), config['fidelity_checkpoint_sha256'])
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload.get('official_vae_checkpoint_sha256') != config['old_vae_checkpoint_sha256']:
                raise RuntimeError('fidelity decoder belongs to another official VAE')
            self.vae.decoder.load_state_dict(payload['decoder'], strict=True)
            del payload
        else:
            var_path = utilities.require_sha(Path(config['old_var_checkpoint']), config['old_var_checkpoint_sha256'])
            with preserve_nn_reset_parameters():
                self.vae, self.var = utilities.build_vae_var(source, vae_path, var_path, device)
        self.metadata = {'family': 'original_VAR', 'decoder': 'fidelity' if fidelity else 'official',
            'patch_nums': list(OLD_PATCHES), 'codebook_size': 4096, 'product_quant': 1,
            'token_layout': 'scale_major', 'precision': 'float32',
            'completion': 'class_conditional_closed_loop_argmax_no_cfg_mixing',
            'vae_checkpoint_sha256': config['old_vae_checkpoint_sha256'],
            'decoder_checkpoint_sha256': config['fidelity_checkpoint_sha256'] if fidelity else config['old_vae_checkpoint_sha256'],
            'var_checkpoint_sha256': None if fidelity else config['old_var_checkpoint_sha256'],
            'fidelity_selection_gate_passed': False if fidelity else None,
            'upstream_global_reset_parameters_restored': True}
        self.last_audit = {}

    def encode(self, x):
        latent = self.vae.quant_conv(self.vae.encoder(x))
        return self.vae.quantize.f_to_idxBl_or_fhat(latent, to_fhat=False, v_patch_nums=OLD_PATCHES)

    def _image(self, fhat):
        return self.vae.decoder(self.vae.post_quant_conv(fhat.float())).clamp(-1, 1).add(1).mul(.5)

    def decode(self, tokens, prefix_scales=10):
        if len(tokens) != prefix_scales:
            raise ValueError('direct decoder receives exactly its transmitted prefix, not the true suffix')
        if prefix_scales == 10:
            fhat = self.u.source_tokens_to_fhat(self.vae, tokens)
        else:
            first = self.vae.quantize.embedding(tokens[0])
            fhat = first.new_zeros(tokens[0].shape[0], self.vae.Cvae, 16, 16)
            # Exactly the historical residual accumulation, retaining total SN=10.
            for si, indices in enumerate(tokens):
                patch = OLD_PATCHES[si]
                embedding = self.vae.quantize.embedding(indices).transpose(1, 2).reshape(indices.shape[0], self.vae.Cvae, patch, patch)
                fhat, _ = self.vae.quantize.get_next_autoregressive_input(si, 10, fhat, embedding)
        return self._image(fhat)

    def token_metadata(self, tokens, prefix_scales=10):
        selected = tokens[:prefix_scales]
        for tokens_at_scale, patch in zip(selected, OLD_PATCHES):
            if list(tokens_at_scale.shape) != [1, patch * patch]:
                raise RuntimeError('old token index shape changed (batch size must be one)')
            if tokens_at_scale.min() < 0 or tokens_at_scale.max() >= 4096:
                raise RuntimeError('old token index outside codebook')
        count = sum(t.numel() for t in selected)
        return {'index_count': count, 'raw_bits': count * 12,
                'shapes': [list(t.shape) for t in selected], 'bits_per_index': 12,
                'product_quant': 1, 'prefix_scales': prefix_scales}

    def complete(self, tokens, labels, prefix_scales, mode='argmax', seed=0, cfg=1.0):
        """Historical loop, but physically accepts only the transmitted prefix."""
        if self.var is None or mode != 'argmax' or cfg != 1.0 or len(tokens) != prefix_scales:
            raise ValueError('old completion supports only class-conditional argmax with exactly its true prefix')
        torch, var, vae = self.torch, self.var, self.vae
        condition = var.class_emb(labels)
        positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
        next_map = (condition[:, None].expand(-1, var.first_l, -1) + var.pos_start.expand(labels.shape[0], var.first_l, -1)
                    + positions[:, :var.first_l])
        fhat = condition.new_zeros(labels.shape[0], var.Cvae, OLD_PATCHES[-1], OLD_PATCHES[-1])
        length, preserved, chosen_shapes = 0, True, []
        for block in var.blocks:
            block.attn.kv_caching(True)
        try:
            for si, patch in enumerate(OLD_PATCHES):
                length += patch * patch
                hidden = next_map
                conditioned = var.shared_ada_lin(condition)
                for block in var.blocks:
                    hidden = block(x=hidden, cond_BD=conditioned, attn_bias=None)
                if si < prefix_scales:
                    chosen = tokens[si]
                    preserved = preserved and torch.equal(chosen, tokens[si])
                else:
                    chosen = var.get_logits(hidden, condition).float().argmax(dim=-1)
                chosen_shapes.append(list(chosen.shape))
                embedding = vae.quantize.embedding(chosen).transpose(1, 2).reshape(labels.shape[0], var.Cvae, patch, patch)
                fhat, next_map = vae.quantize.get_next_autoregressive_input(si, len(OLD_PATCHES), fhat, embedding)
                if si != len(OLD_PATCHES) - 1:
                    next_map = next_map.reshape(labels.shape[0], var.Cvae, -1).transpose(1, 2)
                    next_map = var.word_embed(next_map) + positions[:, length:length + OLD_PATCHES[si + 1] ** 2]
        finally:
            for block in var.blocks:
                block.attn.kv_caching(False)
        self.last_audit = {'prefix_preserved': preserved, 'true_suffix_passed': False,
            'received_scales': len(tokens), 'generated_scales': 10 - prefix_scales,
            'output_shapes': chosen_shapes, 'class_condition': 'pre_shared_ground_truth',
            'cfg_convention': 'one_class_conditional_branch_no_cfg_mixing'}
        return self._image(fhat)


def make_adapter(model: str, config: dict, device, utilities):
    if model in ('old-official', 'old-fidelity'):
        return OldAdapter(config, device, fidelity=model == 'old-fidelity', utilities=utilities)
    module = __import__('xq_adapter' if model == 'xq' else 'wetok_adapter')
    return module.build_adapter(config, device)


def measure(function, device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base_alloc = torch.cuda.memory_allocated(device)
        base_reserved = torch.cuda.memory_reserved(device)
    else:
        base_alloc = base_reserved = 0
    start = time.perf_counter()
    result = function()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    return result, {'seconds': elapsed, 'baseline_allocated_bytes': base_alloc,
        'baseline_reserved_bytes': base_reserved, 'peak_allocated_bytes': peak_alloc,
        'peak_reserved_bytes': peak_reserved, 'incremental_peak_allocated_bytes': peak_alloc - base_alloc}


def execute_case(adapter, tokens, labels, case: dict, model: str):
    if case['kind'] != 'completion':
        receiver_tokens = truncate_tokens(tokens, case['prefix_scales'], model) if case['kind'] == 'direct' else tokens
        image = adapter.decode(receiver_tokens, prefix_scales=case['prefix_scales'])
        return image, {}
    received = truncate_tokens(tokens, case['prefix_scales'], model)
    image = adapter.complete(received, labels, case['prefix_scales'],
        mode=case['mode'], seed=case['seed'], cfg=case['cfg'])
    audit = dict(adapter.last_audit)
    if audit.get('prefix_preserved') is not True:
        raise RuntimeError('adapter did not establish exact transmitted-prefix preservation')
    audit['true_suffix_passed'] = False
    audit['receiver_api_received_scales'] = [len(branch) for branch in received] if model == 'xq' else len(received)
    return image, audit


def check_image(image):
    import torch
    if tuple(image.shape) != (1, 3, 256, 256) or not torch.isfinite(image).all():
        raise RuntimeError('decoder must return finite [1,3,256,256] RGB')
    if float(image.min()) < -1e-6 or float(image.max()) > 1 + 1e-6:
        raise RuntimeError('adapter RGB output outside [0,1]')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/eval.local.json')
    parser.add_argument('--model', required=True, choices=['old-official', 'old-fidelity', 'xq', 'wetok'])
    parser.add_argument('--smoke', action='store_true', help='first two manifest images, separate output tree')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    config = resolve_config(args.config)
    output = args.output_dir or Path(config['output_root']) / ('smoke' if args.smoke else 'full') / args.model
    output = output.expanduser().resolve()
    if not (output.is_relative_to(EXPERIMENT.resolve()) or output.is_relative_to((PROJECT / 'outputs').resolve())):
        raise RuntimeError('new outputs must remain in VAR_COMM, never a historical project')
    building = output.with_name(output.name + '.partial')
    if output.exists() or building.exists():
        raise FileExistsError(f'refusing to overwrite {output} or its partial output')
    building.mkdir(parents=True)
    floats_dir = building / 'float_images'
    floats_dir.mkdir()
    started = time.time()
    record = {'status': 'INITIALIZING', 'model': args.model, 'smoke': args.smoke,
        'config': config, 'started_unix': started, 'pid': os.getpid(),
        'training': False, 'source_only': True, 'channel_uses': None,
        'data_role': 'reused_development_not_holdout', 'class_side_information':
        'ground_truth_ImageNet_class_pre_shared_for_completion_only; raw_bits_excludes_class; optional_10bit_label_reported',
        'timing_protocol': 'batch1_cuda_synchronized_adapter_execution_including_inmemory_prefix_validation; no_file_IO_or_metrics; encode_once_per_source_shared_across_conditions',
        'memory_protocol': 'per_op_absolute_peak_allocated_and_reserved_and_increment; paired_generator_stays_resident_for_old_official_and_xq_including_full_arm; metric_models_loaded_only_after_codec_deleted',
        'float_images_retained': bool(config['save_float_outputs']),
        'script_sha256': digest_file(Path(__file__))}
    json_write(building / 'metadata.json', record)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(int(config['cpu_threads']))
        torch.set_grad_enabled(False)
        random.seed(20260912); np.random.seed(20260912); torch.manual_seed(20260912)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(config['device'])
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('formal/smoke model inference requires the externally authorized CUDA device')
        torch.cuda.set_device(device)
        memory_fraction = float(config.get('max_gpu_memory_fraction', .70))
        if not (0 < memory_fraction <= .70):
            raise ValueError('evaluation CUDA allocation cap must be within (0,0.70]')
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        record['max_gpu_memory_fraction'] = memory_fraction
        manifest_path = Path(config['manifest'])
        if digest_file(manifest_path) != config['manifest_sha256']:
            raise RuntimeError('frozen 100-image manifest changed')
        samples = json.loads(manifest_path.read_text())
        if len(samples) != 100 or len({sample['image_id'] for sample in samples}) != 100:
            raise RuntimeError('development population must be the original 100 unique images')
        if args.smoke:
            samples = samples[:2]
        record.update({'manifest_sha256': digest_file(manifest_path), 'images': len(samples),
            'torch_version': torch.__version__, 'gpu_name': torch.cuda.get_device_name(device),
            'batch_size': 1, 'tf32': False, 'autocast': False, 'cudnn_deterministic': True})
        utilities, metric_module = load_historical(config)
        historical_root = Path(config['historical_source'])
        record['historical_metric_script_sha256'] = digest_file(historical_root / 'scripts/evaluate_var_decoder_only.py')
        record['historical_utility_sha256'] = digest_file(historical_root / 'src/cadsd_jscc/var_decoder_tuning.py')
        cases = conditions(args.model, config)
        record['conditions'] = cases
        adapter, model_load_measure = measure(lambda: make_adapter(args.model, config, device, utilities), device)
        record['model_load_seconds'] = model_load_measure['seconds']
        record['model_load_measure'] = model_load_measure
        record['adapter_metadata'] = adapter.metadata
        adapter_file = HERE / ('xq_adapter.py' if args.model == 'xq' else 'wetok_adapter.py')
        if adapter_file.exists():
            record['adapter_sha256'] = digest_file(adapter_file)
        record['status'] = 'MODEL_INFERENCE'
        json_write(building / 'metadata.json', record)
        with torch.inference_mode():
            # Explicit warmup is excluded from all reported inference measurements.
            for _ in range(int(config['warmup_images'])):
                source = utilities.preprocess_path(Path(samples[0]['path']))[None].to(device)
                labels = torch.tensor([int(samples[0]['class_index'])], device=device)
                warm_tokens = adapter.encode(source)
                for case in cases:
                    warm_image, _ = execute_case(adapter, warm_tokens, labels, case, args.model)
                    check_image(warm_image)
                del warm_image, warm_tokens, source, labels
            json_write(building / 'status.json', {'stage': 'MODEL_INFERENCE', 'completed_images': 0})
            for image_index, sample in enumerate(samples):
                source_path = Path(sample['path'])
                source = utilities.preprocess_path(source_path)[None].to(device)
                labels = torch.tensor([int(sample['class_index'])], device=device)
                tokens, encode_measure = measure(lambda: adapter.encode(source), device)
                source_cpu = source.add(1).mul(.5).cpu()
                image_record = {'image_index': image_index, 'image_id': sample['image_id'],
                    'source_path': str(source_path), 'source_file_sha256': digest_file(source_path),
                    'source_tensor_sha256': tensor_digest(source_cpu), 'class_index': int(sample['class_index']),
                    'synset': sample.get('synset'), 'encode_measure': encode_measure,
                    'conditions': {}, 'model': args.model}
                images = {}
                for case in cases:
                    (image, audit), op_measure = measure(lambda: execute_case(adapter, tokens, labels, case, args.model), device)
                    check_image(image)
                    meta = check_token_metadata(adapter.token_metadata(tokens, prefix_scales=case['prefix_scales']), args.model, case['prefix_scales'])
                    cpu = image.float().cpu()
                    images[case['name']] = cpu
                    image_record['conditions'][case['name']] = {**case, 'token_metadata': meta,
                        'inference_measure': op_measure, 'prefix_audit': audit,
                        'image_tensor_sha256': tensor_digest(cpu),
                        'class_side_bits_if_sent': 10 if case['kind'] == 'completion' else 0,
                        'raw_bits_plus_optional_class': int(meta['raw_bits']) + (10 if case['kind'] == 'completion' else 0)}
                    del image
                if args.smoke and hasattr(adapter, 'smoke_checks'):
                    image_record['upstream_smoke_checks'] = adapter.smoke_checks(source, tokens, labels)
                if args.smoke and args.model in ('old-official', 'xq'):
                    # Identity when all received scales are true; also verify the
                    # primary output is independent of arbitrary offline suffix.
                    all_received = truncate_tokens(tokens, 10, args.model)
                    full = adapter.complete(all_received, labels, 10, mode='argmax', seed=0, cfg=1.0)
                    gap = float((full.cpu() - images['full']).abs().max())
                    if gap > 1e-5 or adapter.last_audit.get('prefix_preserved') is not True:
                        raise RuntimeError(f'full-token completion identity failed: max abs {gap}')
                    mutant = copy.deepcopy(tokens)
                    if args.model == 'xq':
                        for branch in mutant:
                            branch[9].zero_()
                    else:
                        mutant[9].zero_()
                    received = truncate_tokens(mutant, 9, args.model)
                    replay = adapter.complete(received, labels, 9, mode='argmax', seed=0, cfg=1.0)
                    suffix_gap = float((replay.cpu() - images['p9_argmax']).abs().max())
                    if suffix_gap != 0:
                        raise RuntimeError(f'withheld-suffix invariance failed: {suffix_gap}')
                    image_record['smoke_invariants'] = {'full_token_identity_max_abs': gap,
                        'withheld_suffix_mutation_max_abs': suffix_gap, 'true_suffix_not_passed': True}
                    del full, replay, mutant, received, all_received
                torch.save({'source_01': source_cpu, 'images': images}, floats_dir / f'{image_index:03d}.pt')
                json_write(floats_dir / f'{image_index:03d}.json', image_record)
                print(json.dumps({'stage': 'MODEL_INFERENCE', 'model': args.model,
                    'completed': image_index + 1, 'images': len(samples)}), flush=True)
                json_write(building / 'status.json', {'stage': 'MODEL_INFERENCE', 'completed_images': image_index + 1})
                del source, source_cpu, tokens, labels, images, cpu
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        record['status'] = 'METRIC_INFERENCE'
        record['allocated_before_metric_load_bytes'] = torch.cuda.memory_allocated(device)
        json_write(building / 'metadata.json', record)
        dino_path = utilities.require_sha(Path(config['dino_checkpoint']), config['dino_checkpoint_sha256'])
        dino = utilities.load_dino(Path(config['dino_source']), dino_path, device)
        lpips = metric_module.make_lpips(device)
        metrics_rows = []
        with (building / 'per_image.jsonl').open('w') as jsonl, torch.inference_mode():
            for image_index, sample in enumerate(samples):
                tensor_path = floats_dir / f'{image_index:03d}.pt'
                payload = torch.load(tensor_path, map_location='cpu', weights_only=True)
                image_record = json.loads((floats_dir / f'{image_index:03d}.json').read_text())
                source = payload['source_01'].to(device)
                figure_root = building / 'images' / f'{image_index:03d}'
                figure_root.mkdir(parents=True)
                metric_module.save_image(source[0], figure_root / 'source.png')
                for case in cases:
                    image = payload['images'][case['name']].to(device)
                    computed, metrics_measure = measure(lambda: metric_module.calculate_metrics(source, image[:, None],
                        lpips_model=lpips, dino_model=dino), device)
                    row = {key: value for key, value in image_record.items() if key != 'conditions'}
                    row.update(image_record['conditions'][case['name']])
                    row.update({metric: float(computed[metric][0, 0]) for metric in METRICS})
                    if not all(np.isfinite(row[metric]) for metric in METRICS):
                        raise RuntimeError('nonfinite image metric')
                    row['raw_bits'] = int(row['token_metadata']['raw_bits'])
                    row['metric_measure'] = metrics_measure
                    row['png_relative_path'] = str(Path('images') / f'{image_index:03d}' / (case['name'] + '.png'))
                    metric_module.save_image(image[0], building / row['png_relative_path'])
                    jsonl.write(json.dumps(row, ensure_ascii=False) + '\n')
                    metrics_rows.append(row)
                    del image
                jsonl.flush()
                if not config['save_float_outputs']:
                    tensor_path.unlink()  # Only this run's freshly-created spool, never an input.
                print(json.dumps({'stage': 'METRIC_INFERENCE', 'model': args.model,
                    'completed': image_index + 1, 'images': len(samples)}), flush=True)
                json_write(building / 'status.json', {'stage': 'METRIC_INFERENCE', 'completed_images': image_index + 1})
                del payload, source
        record.update({'status': 'COMPLETE', 'rows': len(metrics_rows), 'completed_unix': time.time(),
            'elapsed_wall_seconds': time.time() - started, 'per_image_jsonl_sha256': digest_file(building / 'per_image.jsonl')})
        json_write(building / 'metadata.json', record)
        json_write(building / 'completion.json', {'status': 'COMPLETE', 'model': args.model,
            'smoke': args.smoke, 'images': len(samples), 'rows': len(metrics_rows),
            'training_updates': 0, 'source_only': True, 'config_sha256': config['config_sha256'],
            'manifest_sha256': config['manifest_sha256'], 'per_image_jsonl_sha256': record['per_image_jsonl_sha256']})
        json_write(building / 'status.json', {'stage': 'COMPLETE', 'completed_images': len(samples)})
        building.rename(output)
        print(json.dumps({'status': 'COMPLETE', 'output': str(output), 'rows': len(metrics_rows)}), flush=True)
    except BaseException as error:
        record.update({'status': 'FAILED', 'error': repr(error), 'failed_unix': time.time()})
        json_write(building / 'metadata.json', record)
        raise


if __name__ == '__main__':
    main()
