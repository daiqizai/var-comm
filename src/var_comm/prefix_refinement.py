"""Own-history prefix-state supervision without changing the frozen codec architecture."""

import copy
import hashlib

import numpy as np
import torch
from torch.nn import functional

from .learned_prefix import FrozenVARStream, PREFIX_LENGTHS, empty_cache, straight_through_embedding
from .next_scale_prior import PATCH_NUMS


def optimizer_fingerprint(state):
    digest = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    visit(state)
    return digest.hexdigest()


def restore_optimizer(codec, saved, learning_rate):
    group = saved['param_groups'][0]
    optimizer = torch.optim.AdamW(codec.parameters(), lr=learning_rate, betas=tuple(group['betas']),
                                 eps=group['eps'], weight_decay=group['weight_decay'], amsgrad=group['amsgrad'])
    optimizer.load_state_dict(copy.deepcopy(saved))
    for restored in optimizer.param_groups:
        restored['lr'] = learning_rate
    return optimizer


def stage_weights(config, branch, completed_updates):
    if branch == 'continuation':
        return 'continuation', config['training']['continuation_weights']
    if branch != 'two_stage':
        raise ValueError('unknown continuation branch')
    if completed_updates < config['training']['prefix_only_updates']:
        return 'prefix', config['training']['prefix_weights']
    return 'joint', config['training']['joint_weights']


@torch.no_grad()
def target_prefix_states(tokens, codebook, vae):
    embeddings = functional.embedding(tokens, codebook).split(PREFIX_LENGTHS, dim=1)
    cumulative = codebook.new_zeros(len(tokens), 32, 16, 16)
    states = []
    for scale, values in enumerate(embeddings):
        size = PATCH_NUMS[scale]
        values = values.transpose(1, 2).reshape(len(tokens), 32, size, size)
        values = functional.interpolate(values, size=(16, 16), mode='bicubic')
        cumulative = cumulative + vae.quantize.quant_resi[scale / 9](values)
        states.append(cumulative)
    return torch.stack(states, dim=1)


def receive_prefix(codec, received, labels, snrs, vae, var, render_image=True):
    if codec.variant != 'next_scale':
        raise ValueError('refinement only accepts the frozen next-scale architecture')
    guessed, memory, condition = codec.reader.prepare(received, labels, snrs)
    stream = FrozenVARStream(vae, var, labels)
    predicted, indices_by_scale, logits_by_scale, states, gates = [], [], [], [], []
    offset = 0
    try:
        for length in PREFIX_LENGTHS:
            prior_logits = stream.predict()
            state = stream.state_features() / codec.embedding_std
            logits, gate = codec.reader.read(guessed, memory, condition, offset, offset + length,
                                            prior_logits, state, codec.normalized_codebook)
            embeddings, indices = straight_through_embedding(logits, codec.codebook)
            predicted.append(embeddings)
            logits_by_scale.append(logits)
            indices_by_scale.append(indices)
            gates.append(gate)
            stream.advance(embeddings)
            states.append(stream.fhat)
            offset += length
        result = {'logits': torch.cat(logits_by_scale, dim=1), 'indices': torch.cat(indices_by_scale, dim=1),
                  'prefix_embeddings': predicted, 'prefix_states': torch.stack(states, dim=1),
                  'gates': torch.cat(gates, dim=1), 'teacher_samples': 0, 'rendered_scales': 8}
        if render_image:
            suffix = []
            for scale in (8, 9):
                embeddings, indices = straight_through_embedding(stream.predict(), codec.codebook)
                suffix.append(embeddings)
                stream.advance(embeddings)
            result.update(image=stream.image(), suffix_embeddings=suffix, rendered_scales=10)
        return result
    finally:
        stream.close()


def refinement_forward(codec, tokens, labels, snrs, noise, valid, vae, var, render_image=True):
    symbols = codec.transmit(tokens)
    received = symbols + noise * torch.pow(10.0, snrs / 10.0).rsqrt()[:, None, None]
    result = receive_prefix(codec, received, labels, snrs, vae, var, render_image)
    if render_image:
        result['image'] = torch.where(valid[:, None, None, None], result['image'], result['image'].new_full((), 0.5))
    result.update(symbols=symbols, received=received)
    return result


def prefix_errors(result, tokens, targets, embedding_std):
    cross_entropy = functional.cross_entropy(result['logits'].transpose(1, 2), tokens, reduction='none')
    errors = (result['indices'] != tokens).float()
    state = ((result['prefix_states'] - targets) / embedding_std.reshape(1, 1, 32, 1, 1)).square().flatten(2).mean(2)
    ce_by_scale = torch.stack([values.mean(1) for values in cross_entropy.split(PREFIX_LENGTHS, dim=1)], dim=1)
    ter_by_scale = torch.stack([values.mean(1) for values in errors.split(PREFIX_LENGTHS, dim=1)], dim=1)
    return {'token_ce_raw': cross_entropy.mean(1), 'state_raw': state.mean(1), 'token_error_rate': errors.mean(1),
            'ce_by_scale': ce_by_scale, 'ter_by_scale': ter_by_scale, 'state_by_scale': state}


def refinement_losses(result, images, tokens, targets, valid, perceptual, embedding_std, weights):
    diagnostics = prefix_errors(result, tokens, targets, embedding_std)
    components = {'token_ce': diagnostics['token_ce_raw'] * valid, 'state': diagnostics['state_raw'] * valid}
    if 'image' in result:
        components['mse'] = (result['image'] - images).square().flatten(1).mean(1)
        components['lpips'] = perceptual(result['image'] * 2 - 1, images * 2 - 1).flatten()
    else:
        if weights['mse'] or weights['lpips']:
            raise ValueError('image losses require the real hard image forward')
        components['mse'] = images.new_zeros(len(images))
        components['lpips'] = images.new_zeros(len(images))
    loss = sum(weights[name] * components[name] for name in ('mse', 'lpips', 'token_ce', 'state') if weights[name]).mean()
    return loss, components, diagnostics


def gradient_monitor(codec, tokens, labels, snrs, noise, valid, images, vae, var, perceptual, weights):
    targets = target_prefix_states(tokens, codec.codebook, vae)
    result = refinement_forward(codec, tokens, labels, snrs, noise, valid, vae, var)
    full_weights = {'mse': 1.0, 'lpips': 0.01, 'token_ce': 0.01, 'state': 0.01}
    loss, components, diagnostics = refinement_losses(result, images, tokens, targets, valid, perceptual, codec.embedding_std, full_weights)
    groups = {'encoder': list(codec.encoder.parameters()), 'reader': list(codec.reader.parameters())}
    parameters = [parameter for group in groups.values() for parameter in group]
    observations = []
    for name in ('mse', 'lpips', 'token_ce', 'state'):
        gradients = torch.autograd.grad(components[name].mean(), parameters, retain_graph=True, allow_unused=True)
        position = 0
        for group, members in groups.items():
            selected = gradients[position:position + len(members)]
            squared_norm = sum(float(gradient.detach().double().square().sum()) for gradient in selected if gradient is not None)
            norm = squared_norm ** 0.5
            if not np.isfinite(norm):
                raise FloatingPointError(f'nonfinite {name} gradient to {group}')
            observations.append({'component': name, 'parameter_group': group, 'unweighted_norm': norm,
                                 'weight': weights[name], 'weighted_norm': norm * weights[name],
                                 'parameters_without_gradient': sum(gradient is None for gradient in selected)})
            position += len(members)
    if not empty_cache(var):
        raise RuntimeError('gradient monitor contaminated shared VAR state')
    return observations


def selection_summary(rows, base_psnr, tolerance):
    snrs = sorted({float(row['snr_db']) for row in rows})
    means = {str(snr): float(np.mean([float(row['psnr_db']) for row in rows if float(row['snr_db']) == snr])) for snr in snrs}
    admissible = base_psnr is None or all(means[snr] >= base_psnr[snr] - tolerance for snr in means)
    return {'mean_lpips': float(np.mean([float(row['lpips']) for row in rows])), 'psnr_by_snr': means, 'admissible': admissible}


def engineering_check(codec, tokens, labels, snrs, noise, valid, images, vae, var, perceptual, base_config):
    from .prefix_learning_support import communication_forward, image_losses

    codec.train()
    targets = target_prefix_states(tokens, codec.codebook, vae)
    official = targets.new_zeros(len(tokens), 32, 16, 16)
    maximum_target_error = 0.0
    with torch.no_grad():
        embeddings = functional.embedding(tokens, codec.codebook).split(PREFIX_LENGTHS, dim=1)
        for scale, values in enumerate(embeddings):
            size = PATCH_NUMS[scale]
            values = values.transpose(1, 2).reshape(len(tokens), 32, size, size)
            official, unused = vae.quantize.get_next_autoregressive_input(scale, 10, official, values)
            maximum_target_error = max(maximum_target_error, float((official - targets[:, scale]).abs().max()))
    if maximum_target_error > 1e-6:
        raise RuntimeError('state targets disagree with the original ten-scale quantizer mapping')
    calls = {'var_scales': 0, 'image_decoder': 0}

    def var_hook(module, inputs, output):
        calls['var_scales'] += 1

    def image_hook(module, inputs, output):
        calls['image_decoder'] += 1

    hooks = [var.blocks[0].register_forward_hook(var_hook), vae.decoder.register_forward_hook(image_hook)]
    try:
        prefix = refinement_forward(codec, tokens, labels, snrs, noise, valid, vae, var, render_image=False)
    finally:
        for hook in hooks:
            hook.remove()
    if calls != {'var_scales': 8, 'image_decoder': 0} or 'image' in prefix:
        raise RuntimeError('prefix-only phase accidentally generated suffix or RGB')
    prefix_loss, components, diagnostics = refinement_losses(prefix, images, tokens, targets, valid, perceptual,
                                                            codec.embedding_std, {'mse': 0, 'lpips': 0, 'token_ce': 1, 'state': 0.1})
    state_gradients = torch.autograd.grad(components['state'].mean(), [codec.encoder.mixing, codec.reader.backprojection])
    state_norms = [float(gradient.norm()) for gradient in state_gradients]
    if not all(np.isfinite(state_norms)) or min(state_norms) <= 0:
        raise RuntimeError('state loss does not reach both communication modules')
    prefix_indices = prefix['indices'].detach().clone()
    del prefix, prefix_loss, components, diagnostics, state_gradients
    original = communication_forward(codec, tokens, labels, snrs, noise, valid, vae, var)
    old_loss, components = image_losses(original, images, tokens, valid, perceptual, base_config['training'])
    old_image, old_indices = original['image'].detach().clone(), original['indices'].detach().clone()
    parameters = [codec.encoder.mixing, codec.reader.backprojection]
    old_gradients = torch.autograd.grad(old_loss, parameters)
    old_value = float(old_loss.detach())
    del original, old_loss, components
    revised = refinement_forward(codec, tokens, labels, snrs, noise, valid, vae, var)
    loss, components, diagnostics = refinement_losses(revised, images, tokens, targets, valid, perceptual,
                                                     codec.embedding_std, {'mse': 1, 'lpips': 0.01, 'token_ce': 0.001, 'state': 0})
    gradients = torch.autograd.grad(loss, parameters)
    image_error = float((revised['image'] - old_image).abs().max())
    if image_error != 0 or not torch.equal(revised['indices'], old_indices) or not torch.equal(prefix_indices, old_indices):
        raise RuntimeError('new state tracking changed the original hard forward')
    torch.testing.assert_close(loss.detach(), loss.new_tensor(old_value), rtol=1e-6, atol=1e-8)
    for actual, expected in zip(gradients, old_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    if any(parameter.grad is not None for parameter in codec.parameters()) or not empty_cache(var):
        raise RuntimeError('engineering check mutated gradient or receiver state')
    return {'status': 'ENGINEERING_CHECK_PASS', 'device': str(tokens.device), 'optimizer_updates': 0,
            'original_hard_image_max_error': image_error, 'original_loss_error': abs(float(loss.detach()) - old_value),
            'maximum_comparison_gradient_error': max(float((actual - expected).abs().max()) for actual, expected in zip(gradients, old_gradients)),
            'state_target_official_max_error': maximum_target_error, 'state_gradients_encoder_reader': state_norms,
            'prefix_only_forward_calls': calls, 'no_teacher': True}
