"""Candidate receiver acceleration; never patches or launches a frozen study.

Pass the original residual_receiver module explicitly. All model/prior/operator
calls remain in that module. The coordinate sweep keeps raster Gauss-Seidel
updates and FP32 GEMV order, while retaining token indices on the device.
Deployment requires a separate real-weight parity/timing qualification.
"""
from __future__ import annotations

import math


def coordinate_sweep(operator, tokens, error, variance, *, log_probs=None,
                     static_log_prior=None, lam=0., _torch=None):
    """One sequential sweep; no device scalar is read inside the position loop.

    tokens is one source [1, positions], error is [observation coordinates].
    log_probs is the original VAR [1, positions, vocabulary] output, or
    static_log_prior is the original per-scale [vocabulary] table. A caller
    supplies at most one. Returns tokens, updated error, and original diagnostics.
    _torch permits CPU interface fixtures; formal inference passes rx.torch.
    """
    if _torch is None:
        import torch as _torch
    torch = _torch
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
        raise ValueError('One nonempty source token row required')
    if error.ndim != 1 or operator.columns.shape[0] != tokens.shape[1]:
        raise ValueError('Coordinate operator/token/error shapes differ')
    if not math.isfinite(float(variance)) or variance <= 0:
        raise ValueError('Positive working variance required')
    if not math.isfinite(float(lam)) or lam < 0:
        raise ValueError('Nonnegative finite lambda required')
    if log_probs is not None and static_log_prior is not None:
        raise ValueError('Choose VAR or static prior, never both')
    E = operator.q.embedding.weight.detach()
    increments, flags = [], []
    for position in range(tokens.shape[1]):
        old = tokens[:, position]  # Device long[1], never int(old).
        previous = E.index_select(0, old).squeeze(0)
        G = operator.columns[position]
        b = G.T @ error + operator.gram[position] @ previous
        scores = (E @ b - .5 * operator.quadratic[position]) / float(variance)
        prior = log_probs[0, position] if log_probs is not None else static_log_prior
        if prior is not None:
            scores = scores + float(lam) * prior
        new = scores.argmax().reshape(1)
        increments.append((scores.index_select(0, new) - scores.index_select(0, old)).squeeze(0))
        changed = new != old
        flags.append(changed.squeeze(0))
        delta = G @ (E.index_select(0, new).squeeze(0) - previous)
        # Select the original error when the token is unchanged. Subtracting
        # a computed zero unconditionally can alter signed-zero bit patterns.
        error = torch.where(changed, error - delta, error)
        tokens[:, position].copy_(new)
    # Match the original Python float accumulation order; a GPU sum or a
    # tree reduction would change rounding. Only two transfers per scale.
    objective_increase = 0.
    for increment in torch.stack(increments).cpu().tolist():
        objective_increase += float(increment)
    changes = sum(bool(flag) for flag in torch.stack(flags).cpu().tolist())
    return dict(tokens=tokens, error=error, token_changes=changes,
                surrogate_objective_increase=objective_increase)


def infer(rx, vae, var, prefix, observation, projection, stats, *,
          noise_variance=0., prior='VAR', lam=1., static=None, operators=None,
          check=lambda: None):
    """Drop-in candidate with the original module as its first argument.

    Signature and returned fields after rx match residual_receiver.infer().
    No truth, target, source identity, feedback, or additional selection input.
    This function does not monkeypatch rx.infer or change CUDA/precision flags.
    """
    with rx.torch.no_grad():
        return _infer(rx, vae, var, prefix, observation, projection, stats,
            noise_variance=noise_variance, prior=prior, lam=lam, static=static,
            operators=operators, check=check)


def _infer(rx, vae, var, prefix, observation, projection, stats, *,
           noise_variance, prior, lam, static, operators, check):
    torch = rx.torch
    if tuple(vae.quantize.v_patch_nums) != rx.PATCH_NUMS or vae.quantize.using_znorm:
        raise ValueError('Original Euclidean official ten-scale quantizer required')
    if prior not in ('VAR', 'STATIC', 'LIKELIHOOD', 'UNGUIDED'):
        raise ValueError('Unknown registered residual receiver')
    if len(prefix) != rx.PREFIX_M or not math.isfinite(lam) or lam < 0:
        raise ValueError('Fixed m4 prefix and nonnegative finite lambda required')
    if not math.isfinite(float(noise_variance)) or noise_variance < 0:
        raise ValueError('Noise variance must be finite and nonnegative')
    q = vae.quantize
    if observation.shape != (1, projection.dimension) or observation.dtype != torch.float32:
        raise ValueError('Registered one-source FP32 observation required')
    if not torch.isfinite(observation).all():
        raise ValueError('Finite observation required')
    op = operators or rx.Operators(q, projection, check=check)
    residual = observation.clone()
    fhat = q.embedding.weight.new_zeros(1, q.embedding.embedding_dim, 16, 16)
    pr = rx.Prior(var, 1) if prior in ('VAR', 'UNGUIDED') else None
    chosen, sweep = [], []
    try:
        for k, pn in enumerate(rx.PATCH_NUMS):
            check()
            lp = pr.log_probs(k) if pr is not None else None
            if k < rx.PREFIX_M:
                tokens = prefix[k]
            elif prior == 'UNGUIDED':
                tokens = lp.argmax(-1)
            else:
                mean = stats['mean'][k:k + 1].to(observation)
                variance = float(stats['scalar_variance'][k]) + float(noise_variance)
                tokens = rx.initial_tokens(q, projection, residual, mean, k)
                so = op.get(k)
                error = (residual - mean - so.measured(tokens))[0]
                # The static values are unchanged; transfer once per scale
                # instead of repeating the same .to(observation) per position.
                static_lp = static[k].to(observation) if prior == 'STATIC' else None
                out = coordinate_sweep(so, tokens, error, variance,
                    log_probs=lp if prior == 'VAR' else None,
                    static_log_prior=static_lp, lam=lam, _torch=torch)
                tokens, error = out['tokens'], out['error']
                sweep.append(dict(scale_index=k, pn=pn,
                    token_changes=out['token_changes'],
                    surrogate_objective_increase=out['surrogate_objective_increase'],
                    working_variance=variance, passes=1, order='raster'))
                exact = (residual - mean - so.measured(tokens))[0]
                if not torch.allclose(error, exact, atol=2e-4, rtol=2e-4):
                    raise RuntimeError('Phi affine-difference consistency failed')
            chosen.append(tokens)
            fhat, nextmap = q.get_next_autoregressive_input(k, len(rx.PATCH_NUMS), fhat,
                rx.embedded(q, tokens, k))
            if k >= rx.PREFIX_M:
                residual -= projection.forward(rx.contribution(q, tokens, k))
            if pr is not None:
                pr.advance(nextmap, k)
    finally:
        if pr is not None:
            pr.close()
    return dict(fhat=fhat, tokens=chosen, diagnostics=sweep,
                inference='one-sweep scale-wise Gaussian plug-in/composite guidance')
