"""Frozen residual observation guidance with an explicitly approximate model.

The physical observation is linear in F, not in individual token indices.
This module preserves projected bicubic/Phi coupling. A Gaussian mean/scalar
variance for the unobserved future residual is a calibration plug-in model;
reusing the observation at subsequent scales is composite guidance, not an
exact joint posterior. No clean target enters infer().
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as fn

PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
PROJECTIONS = ((8, 32), (8, 8), (6, 8), (4, 32), (4, 16), (4, 8))
PREFIX_M = 4
VARIANCE_FLOOR = 1e-6


def tensor_sha(value):
    a = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def embedded(q, tokens, k):
    pn = PATCH_NUMS[k]
    return q.embedding(tokens).transpose(1, 2).reshape(len(tokens), q.embedding.embedding_dim, pn, pn)


def contribution_map(q, h, k):
    if k != len(PATCH_NUMS) - 1:
        h = fn.interpolate(h, size=(16, 16), mode='bicubic').contiguous()
    return q.quant_resi[k / (len(PATCH_NUMS) - 1)](h)


def contribution(q, tokens, k):
    return contribution_map(q, embedded(q, tokens, k), k)


def prefix_latent(q, prefix):
    if not prefix or len(prefix) > len(PATCH_NUMS):
        raise ValueError('A nonempty official prefix is required')
    out = q.embedding.weight.new_zeros(len(prefix[0]), q.embedding.embedding_dim, 16, 16)
    for k, tokens in enumerate(prefix):
        if tokens.shape != (len(out), PATCH_NUMS[k] ** 2):
            raise ValueError('Prefix shape disagrees with official scale schedule')
        out, _ = q.get_next_autoregressive_input(k, len(PATCH_NUMS), out, embedded(q, tokens, k))
    return out


@torch.no_grad()
def fit_channel_pca(residual):
    """Shared channel axes; covariance centered across sources/spatial positions.

    The transmitted measurement itself is uncentered, so its operator remains
    linear. Original 16x16 calibration residuals fit ONE basis for all grids.
    """
    if residual.ndim != 4 or tuple(residual.shape[1:]) != (32, 16, 16):
        raise ValueError('PCA requires original calibration residual32x16x16')
    x = residual.double().permute(0, 2, 3, 1).reshape(-1, 32)
    mean = x.mean(0)
    x = x - mean
    covariance = x.T @ x / len(x)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True, stable=True)
    values, vectors = values[order], vectors[:, order]
    # Register actual axes; this sign rule makes separately loaded copies stable.
    for j in range(32):
        i = int(vectors[:, j].abs().argmax())
        if vectors[i, j] < 0:
            vectors[:, j].neg_()
    axes = vectors.T.float().contiguous()
    return dict(axes=axes, mean=mean.float(), eigenvalues=values.float(),
                covariance=covariance.float(), samples=len(x),
                axis_sha256=tensor_sha(axes), neural_training_updates=0)


class Projection:
    """area_g followed by shared per-position PCA; no spatial PCA mixing."""
    def __init__(self, g, channels, axes, *, feature_channels=32):
        if (g, channels) not in PROJECTIONS and feature_channels == 32:
            raise ValueError('Unregistered residual projection')
        if axes.shape != (feature_channels, feature_channels):
            raise ValueError('Complete registered PCA axes are required')
        if not torch.allclose(axes @ axes.T, torch.eye(feature_channels, device=axes.device), atol=2e-5, rtol=2e-5):
            raise ValueError('Channel axes must be orthonormal')
        self.g, self.channels, self.feature_channels = int(g), int(channels), int(feature_channels)
        self.axes = axes[:channels].contiguous()
        self.dimension = g * g * channels
        self.name = f'g{g}_c{channels}'
        # The spatial right inverse is exact for the registered area operator,
        # including nondivisible g=6. Bicubic is NOT its inverse.
        eye = torch.eye(256, dtype=torch.float64, device=axes.device).reshape(256, 1, 16, 16)
        area = fn.interpolate(eye, size=(g, g), mode='area').flatten(1).T
        self.spatial_right_inverse = (area.T @ torch.linalg.inv(area @ area.T)).float()

    def forward(self, latent):
        if latent.ndim != 4 or tuple(latent.shape[1:]) != (self.feature_channels, 16, 16):
            raise ValueError('Projection expects original feature coordinates')
        low = fn.interpolate(latent, size=(self.g, self.g), mode='area')
        return torch.einsum('oc,bcij->boij', self.axes.to(low), low).flatten(1)

    def lift(self, observation):
        if observation.ndim != 2 or observation.shape[1] != self.dimension:
            raise ValueError('Observation dimension disagrees with projection')
        low = observation.reshape(len(observation), self.channels, self.g * self.g)
        channels = torch.einsum('co,boj->bcj', self.axes.T.to(low), low)
        high = channels @ self.spatial_right_inverse.T.to(low)
        return high.reshape(len(observation), self.feature_channels, 16, 16)

    def identity(self):
        return dict(name=self.name, g=self.g, channels=self.channels,
                    real_coordinates=self.dimension, complex_measurement_uses=self.dimension // 2,
                    channel_axes_sha256=tensor_sha(self.axes), spatial='original area16_to_g',
                    centered_measurement=False, channel_pca_shared_across_spatial_positions=True,
                    lift='area spatial right inverse and channel transpose')


@torch.no_grad()
def fit_structural_stats(q, F, tokens, projection, *, check=lambda: None, batch=10):
    """Fit remaining F error after each TRUE current scale, calibration only."""
    if len(tokens) != len(PATCH_NUMS) or len(F) != len(tokens[0]):
        raise ValueError('Complete calibration tokens must align with F')
    device = q.embedding.weight.device
    sums = [torch.zeros(projection.dimension, dtype=torch.float64, device=device) for _ in PATCH_NUMS]
    squares = [torch.zeros_like(x) for x in sums]
    for start in range(0, len(F), batch):
        check()
        residual = F[start:start + batch].to(device).clone()
        for k in range(len(PATCH_NUMS)):
            residual -= contribution(q, tokens[k][start:start + batch].to(device), k)
            measured = projection.forward(residual).double()
            sums[k] += measured.sum(0)
            squares[k] += measured.square().sum(0)
    means = [x / len(F) for x in sums]
    diag = [(x / len(F) - m.square()).clamp_min(0) for x, m in zip(squares, means)]
    scalar = [max(float(x.mean()), VARIANCE_FLOOR) for x in diag]
    return dict(mean=torch.stack(means).float().cpu(),
                scalar_variance=torch.tensor(scalar, dtype=torch.float32),
                diagonal_variance_diagnostic=torch.stack(diag).float().cpu(),
                variance_floor=VARIANCE_FLOOR, calibration_sources=len(F),
                scope='true-prefix remaining residual; receiver prefix errors not separately modeled',
                model='Gaussian scalar-isotropic structural approximation; per-coordinate mean',
                projection=projection.identity())


class ScaleOperator:
    """Affine projected contribution, represented by bias and exact differences."""
    @torch.no_grad()
    def __init__(self, q, projection, k, *, check=lambda: None):
        self.k, self.q, self.projection = k, q, projection
        self.pn = PATCH_NUMS[k]
        channels = q.embedding.embedding_dim
        zero = q.embedding.weight.new_zeros(1, channels, self.pn, self.pn)
        self.bias = projection.forward(contribution_map(q, zero, k))[0]
        columns = []
        for position in range(self.pn ** 2):
            check()
            basis = zero.expand(channels, -1, -1, -1).clone()
            basis[torch.arange(channels, device=basis.device), torch.arange(channels, device=basis.device),
                  position // self.pn, position % self.pn] = 1
            mapped = projection.forward(contribution_map(q, basis, k)) - self.bias[None]
            columns.append(mapped.T.contiguous())
        self.columns = torch.stack(columns)  # position, observation, embedding channel
        self.gram = self.columns.transpose(1, 2) @ self.columns
        E = q.embedding.weight.detach()
        self.quadratic = torch.einsum('vc,pcd,vd->pv', E, self.gram, E).contiguous()

    def measured(self, tokens):
        return self.projection.forward(contribution(self.q, tokens, self.k))


class Operators:
    def __init__(self, q, projection, *, check=lambda: None):
        self.q, self.projection, self.check = q, projection, check
        self.cache = {}

    def get(self, k):
        if k not in self.cache:
            self.cache[k] = ScaleOperator(self.q, self.projection, k, check=self.check)
        return self.cache[k]


class Prior:
    """Frozen unconditional original VAR cached forward, no CFG or truncation."""
    def __init__(self, var, batch):
        if var.training or float(var.cond_drop_rate) != 0:
            raise ValueError('Frozen eval VAR with condition dropout disabled required')
        self.var, self.batch, self.offset = var, batch, 0
        labels = torch.full((batch,), 1000, dtype=torch.long, device=next(var.parameters()).device)
        self.cond = var.class_emb(labels)
        self.condition = var.shared_ada_lin(self.cond)
        self.positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
        self.x = (self.cond[:, None].expand(-1, var.first_l, -1)
                  + var.pos_start.expand(batch, var.first_l, -1) + self.positions[:, :var.first_l])
        for block in var.blocks:
            block.attn.kv_caching(True)

    def log_probs(self, k):
        if self.x.shape[1] != PATCH_NUMS[k] ** 2:
            raise ValueError('Prior input length mismatch')
        h = self.x
        for block in self.var.blocks:
            h = block(x=h, cond_BD=self.condition, attn_bias=None)
        result = fn.log_softmax(self.var.get_logits(h, self.cond).float(), -1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite full-vocabulary prior')
        return result

    def advance(self, nextmap, k):
        self.offset += PATCH_NUMS[k] ** 2
        if k < len(PATCH_NUMS) - 1:
            x = nextmap.reshape(self.batch, self.var.Cvae, -1).transpose(1, 2)
            self.x = self.var.word_embed(x) + self.positions[:, self.offset:self.offset + PATCH_NUMS[k + 1] ** 2]

    def close(self):
        for block in self.var.blocks:
            block.attn.kv_caching(False)


def candidate_scores(operator, position, current, error, variance, log_prior=None, lam=0.):
    """Whole-observation conditional score; other same-scale tokens fixed.

    Terms constant across candidate values are dropped. Gaussian score is exact
    for THIS fixed surrogate conditional objective, not the actual posterior.
    """
    if not math.isfinite(float(variance)) or variance <= 0:
        raise ValueError('Positive working variance required')
    E = operator.q.embedding.weight.detach()
    G = operator.columns[position]
    b = G.T @ error + operator.gram[position] @ E[current]
    score = (E @ b - .5 * operator.quadratic[position]) / float(variance)
    if log_prior is not None:
        score = score + float(lam) * log_prior
    return score


def initial_tokens(q, projection, observation, mean, k):
    x = projection.lift(observation - mean)
    pn = PATCH_NUMS[k]
    if k != len(PATCH_NUMS) - 1:
        x = fn.interpolate(x, size=(pn, pn), mode='area')
    x = x.permute(0, 2, 3, 1).reshape(-1, q.embedding.embedding_dim)
    E = q.embedding.weight.detach()
    d = x.square().sum(1, keepdim=True) + E.square().sum(1) - 2 * x @ E.T
    return d.argmin(-1).reshape(len(observation), pn * pn)


@torch.no_grad()
def infer(vae, var, prefix, observation, projection, stats, *, noise_variance=0.,
          prior='VAR', lam=1., static=None, operators=None, check=lambda: None):
    """Receiver inputs only. No F, true missing tokens or source identity.

    One deterministic raster conditional-coordinate sweep per missing scale.
    Likelihood/static controls use the SAME observation-only initialization.
    """
    if tuple(vae.quantize.v_patch_nums) != PATCH_NUMS or vae.quantize.using_znorm:
        raise ValueError('Original Euclidean official ten-scale quantizer required')
    if prior not in ('VAR', 'STATIC', 'LIKELIHOOD', 'UNGUIDED'):
        raise ValueError('Unknown registered residual receiver')
    if len(prefix) != PREFIX_M or not math.isfinite(lam) or lam < 0:
        raise ValueError('Fixed m4 prefix and nonnegative finite lambda required')
    if not math.isfinite(float(noise_variance)) or noise_variance < 0:
        raise ValueError('Noise variance must be finite and nonnegative')
    q = vae.quantize
    if observation.shape != (1, projection.dimension) or observation.dtype != torch.float32:
        raise ValueError('Registered one-source FP32 observation required')
    if not torch.isfinite(observation).all():
        raise ValueError('Finite observation required')
    op = operators or Operators(q, projection, check=check)
    residual = observation.clone()
    fhat = q.embedding.weight.new_zeros(1, q.embedding.embedding_dim, 16, 16)
    pr = Prior(var, 1) if prior in ('VAR', 'UNGUIDED') else None
    chosen, sweep = [], []
    try:
        for k, pn in enumerate(PATCH_NUMS):
            check()
            lp = pr.log_probs(k) if pr is not None else None
            if k < PREFIX_M:
                tokens = prefix[k]
            elif prior == 'UNGUIDED':
                tokens = lp.argmax(-1)
            else:
                mean = stats['mean'][k:k + 1].to(observation)
                variance = float(stats['scalar_variance'][k]) + float(noise_variance)
                tokens = initial_tokens(q, projection, residual, mean, k)
                so = op.get(k)
                error = (residual - mean - so.measured(tokens))[0]
                objective_increase, changed = 0., 0
                for position in range(pn * pn):
                    old = int(tokens[0, position])
                    prior_lp = (lp[0, position] if prior == 'VAR' else
                                static[k].to(observation) if prior == 'STATIC' else None)
                    scores = candidate_scores(so, position, old, error, variance, prior_lp, lam)
                    new = int(scores.argmax())
                    objective_increase += float(scores[new] - scores[old])
                    if new != old:
                        error -= so.columns[position] @ (q.embedding.weight[new] - q.embedding.weight[old])
                        tokens[0, position] = new
                        changed += 1
                sweep.append(dict(scale_index=k, pn=pn, token_changes=changed,
                                  surrogate_objective_increase=objective_increase,
                                  working_variance=variance, passes=1, order='raster'))
                # Check our coupled affine differences against the real forward.
                exact = (residual - mean - so.measured(tokens))[0]
                if not torch.allclose(error, exact, atol=2e-4, rtol=2e-4):
                    raise RuntimeError('Phi affine-difference consistency failed')
            chosen.append(tokens)
            fhat, nextmap = q.get_next_autoregressive_input(k, len(PATCH_NUMS), fhat, embedded(q, tokens, k))
            if k >= PREFIX_M:
                residual -= projection.forward(contribution(q, tokens, k))
            if pr is not None:
                pr.advance(nextmap, k)
    finally:
        if pr is not None:
            pr.close()
    return dict(fhat=fhat, tokens=chosen, diagnostics=sweep,
                inference='one-sweep scale-wise Gaussian plug-in/composite guidance')


def direct_correction(base, prefix_map, observation, projection, alpha=1.):
    if not 0 <= alpha <= 1:
        raise ValueError('Registered direct alpha must be within[0,1]')
    discrepancy = observation - projection.forward(base - prefix_map)
    return base + float(alpha) * projection.lift(discrepancy)


def resource_record(N, g, channels, phy='QPSK'):
    if (g, channels) not in PROJECTIONS or N not in (512, 1024) or phy not in ('QPSK', '16QAM'):
        raise ValueError('Unregistered budget/projection/PHY')
    measurement = g * g * channels // 2
    digital = N - 68 - 32 - measurement - 1
    width = 2 if phy == 'QPSK' else 4
    rate = 382 / (width * digital) if digital > 0 else None
    feasible = digital > 0 and rate <= .9
    return dict(N=N, g=g, channels=channels, N_header=68, N_gain=32,
                N_measurement=measurement, N_padding=1, N_digital=max(digital, 0),
                source_bits=360, prefix_crc_bits=16, prefix_tail_bits=6,
                gain_payload_bits=32, gain_crc_bits=16, gain_tail_bits=6,
                effective_prefix_rate=rate, gain_rate=54 / 64, phy_family=phy,
                feasible=feasible, status='FEASIBLE' if feasible else 'NOT_FEASIBLE',
                resource_scope='ideal correct digital prefix and gain in oracle; paid symbols still counted')


def analog_transmit(coordinates):
    """Paid float32 gain plus ignored padding makes a strict energy waveform.

    The caller separately transmits all32 gain bits with CRC/tail in32 QPSK
    uses. Padding is NEVER used to estimate gain at RX.
    """
    z = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    if len(z) % 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError('Finite even-length projected real coordinates required')
    measurement = len(z) // 2
    energy = float(z @ z)
    ideal = math.sqrt((2 * measurement) / energy) if energy > 0 else 1.
    gain = np.float32(ideal)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Gain outside registered positive finite float32 domain')
    if float(gain) > ideal:
        gain = np.nextafter(gain, np.float32(0))
    symbols = (z * float(gain)).reshape(-1, 2)
    remaining = 2 * (measurement + 1) - float(np.square(symbols).sum())
    if remaining < -1e-7:
        raise RuntimeError('Down-rounded gain exceeded energy budget')
    padding = np.array([[math.sqrt(max(remaining, 0.)), 0.]])
    wave = np.concatenate((symbols, padding))
    if not np.isclose(float(np.square(wave).sum()), 2 * (measurement + 1), atol=1e-8, rtol=1e-12):
        raise RuntimeError('Analog energy normalization failed')
    return wave, float(gain)

