"""Frozen, full-scale receiver corrections for Step 2 A.

Only Z, the codebook, fixed variance/static tables and receiver-selected prefix
enter closed-loop decisions. clean/truth are optional offline scoring branches.
No signal handlers, launchers, model loading or training run on import.
"""
import math

import torch
from torch.nn import functional as fn

PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
VARIANCE_FLOOR = 1e-12


def _validate(vae, Z, prior, variance, static, lam, variance_floor):
    if prior not in ('A1', 'A2', 'V'):
        raise ValueError('prior must be A1, A2 or V')
    if not math.isfinite(float(lam)) or lam < 0:
        raise ValueError('lambda must be finite and nonnegative')
    if not math.isfinite(float(variance_floor)) or variance_floor <= 0:
        raise ValueError('variance_floor must be positive and finite')
    if tuple(vae.quantize.v_patch_nums) != PATCH_NUMS:
        raise ValueError('the complete official ten-scale schedule is required')
    if vae.quantize.using_znorm:
        raise ValueError('Euclidean scoring requires the original codebook coordinates')
    if Z.ndim != 4 or tuple(Z.shape[1:]) != (32, 16, 16):
        raise ValueError('Z must have shape B,32,16,16')
    if Z.dtype != torch.float32:
        raise ValueError('the registered official quantizer uses FP32 coordinates')
    if len(variance) != len(PATCH_NUMS):
        raise ValueError('one fixed scalar variance per scale is required')
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in variance):
        raise ValueError('variance must be finite and nonnegative')
    if prior == 'A2' and lam != 0:
        if static is None or len(static) != len(PATCH_NUMS):
            raise ValueError('static log probabilities are required for every scale')
        if any(x.ndim != 1 or x.numel() != 4096 for x in static):
            raise ValueError('static prior must give all 4096 codeword log probabilities')


def _truth_shapes(truth, Z):
    if len(truth) != len(PATCH_NUMS):
        raise ValueError('truth must contain every official scale')
    for pn, tokens in zip(PATCH_NUMS, truth):
        if tokens.shape != (Z.shape[0], pn * pn) or tokens.dtype != torch.long:
            raise ValueError('truth shape/dtype differs from the frozen source tokens')
        if tokens.device != Z.device:
            raise ValueError('truth and Z must use the same device')


def distance(q, residual, k):
    """Official FP32 Euclidean distance, preserving nearest-token tie ordering."""
    pn = PATCH_NUMS[k]
    z = (fn.interpolate(residual, size=(pn, pn), mode='area')
         if k < len(PATCH_NUMS) - 1 else residual)
    z = z.permute(0, 2, 3, 1).reshape(-1, 32)
    codebook = q.embedding.weight.data
    d = z.square().sum(1, keepdim=True) + codebook.square().sum(1)
    d.addmm_(z, codebook.T, alpha=-2, beta=1)
    return d.reshape(residual.shape[0], pn * pn, codebook.shape[0])


def embedded(q, tokens, k):
    pn = PATCH_NUMS[k]
    return q.embedding(tokens.reshape(-1, pn, pn)).permute(0, 3, 1, 2).contiguous()


def contribution(q, tokens, k):
    h = embedded(q, tokens, k)
    if k < len(PATCH_NUMS) - 1:
        h = fn.interpolate(h, size=(16, 16), mode='bicubic').contiguous()
    return q.quant_resi[k / (len(PATCH_NUMS) - 1)](h)


class Prior:
    """Official unconditional cached forward; batch form of Step 1 probe.Prior."""
    def __init__(self, var, batch_size):
        if var is None or var.training or float(var.cond_drop_rate) != 0:
            raise ValueError('VAR must be frozen eval with condition dropout disabled')
        self.var = var
        self.batch_size = batch_size
        label = torch.full((batch_size,), 1000, dtype=torch.long,
                           device=next(var.parameters()).device)
        self.cond = var.class_emb(label)
        self.condition = var.shared_ada_lin(self.cond)
        self.positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
        self.x = (self.cond[:, None].expand(-1, var.first_l, -1)
                  + var.pos_start.expand(batch_size, var.first_l, -1)
                  + self.positions[:, :var.first_l])
        self.offset = 0
        for block in var.blocks:
            block.attn.kv_caching(True)

    def logits(self, k):
        if self.x.shape[1] != PATCH_NUMS[k] ** 2:
            raise ValueError('VAR prefix length disagrees with the registered scale')
        h = self.x
        for block in self.var.blocks:
            h = block(x=h, cond_BD=self.condition, attn_bias=None)
        logits = self.var.get_logits(h, self.cond).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite conditional VAR logits')
        return logits

    def advance(self, nextmap, k):
        self.offset += PATCH_NUMS[k] ** 2
        if k < len(PATCH_NUMS) - 1:
            x = nextmap.reshape(self.batch_size, 32, -1).transpose(1, 2)
            next_length = PATCH_NUMS[k + 1] ** 2
            self.x = (self.var.word_embed(x)
                      + self.positions[:, self.offset:self.offset + next_length])

    def close(self):
        for block in self.var.blocks:
            block.attn.kv_caching(False)


def _metrics(score, idx, original=None, path=None):
    """Accuracy targets are deliberately kept separate; arrays retain batch IDs."""
    row = dict(token_count=int(idx.shape[1]))
    if original is not None:
        values = idx.eq(original).double().mean(-1)
        row.update(acc=float(values.mean()), acc_per_image=values.cpu().tolist())
    if path is not None:
        values = idx.eq(path).double().mean(-1)
        row.update(path_accuracy=float(values.mean()),
                   path_accuracy_per_image=values.cpu().tolist())
    if score is not None:
        post = fn.log_softmax(score, dim=-1)
        entropy = -(post.exp().double() * post.double()).sum(-1).mean(-1)
        row.update(entropy=float(entropy.mean()), entropy_per_image=entropy.cpu().tolist())
        if original is not None:
            lp = post.gather(-1, original[..., None]).squeeze(-1).double().mean(-1)
            row.update(logp_true=float(lp.mean()), logp_true_per_image=lp.cpu().tolist())
    return row


def _score(d, prior_lp, variance, lam, variance_floor):
    # The zero-prior branch chooses tokens using distance directly, so it equals
    # A1 even at zero variance. A protected finite score is used only for reports.
    value = max(float(variance), float(variance_floor))
    score = -d / (2 * value)
    if prior_lp is not None:
        score = score + float(lam) * prior_lp
        idx = score.argmax(-1)
    else:
        idx = d.argmin(-1)
    return score, idx


@torch.no_grad()
def infer(vae, var, Z, prior, variance, static=None, lam=1., clean=None,
          truth=None, details=True, variance_floor=VARIANCE_FLOOR,
          return_logits=False):
    """Closed loop. clean/truth affect metrics only and never choose a token.

    Returned fhat is the unfused Fq_post in original codebook coordinates. The
    caller chooses a frozen alpha or BYPASS. Batch metrics contain per-image
    arrays; their scalar versions average images for convenience.
    """
    _validate(vae, Z, prior, variance, static, lam, variance_floor)
    if truth is not None:
        _truth_shapes(truth, Z)
    if clean is not None and (clean.shape != Z.shape or clean.device != Z.device):
        raise ValueError('offline clean scoring input must match Z')
    q = vae.quantize
    rest, fhat = Z.clone(), torch.zeros_like(Z)
    clean_rest = clean.clone() if clean is not None and details else None
    pr = Prior(var, Z.shape[0]) if prior == 'V' and lam != 0 else None
    tokens, metrics, logits = [], [], []
    try:
        for k in range(len(PATCH_NUMS)):
            d = distance(q, rest, k)
            lp = None
            if pr is not None:
                lg = pr.logits(k)
                lp = fn.log_softmax(lg, dim=-1)
                if return_logits:
                    logits.append(lg)
            elif prior == 'A2' and lam != 0:
                lp = static[k].to(device=Z.device, dtype=torch.float32)[None, None]
            score, idx = _score(d, lp, variance[k], lam, variance_floor)
            tokens.append(idx)
            if details:
                path = distance(q, clean_rest, k).argmin(-1) if clean_rest is not None else None
                metrics.append(_metrics(score, idx, truth[k] if truth is not None else None, path))
            # Both residuals subtract the same receiver-selected contribution.
            # clean_rest never changes any decision, accumulated map or prefix.
            h = contribution(q, idx, k)
            rest.sub_(h)
            if clean_rest is not None:
                clean_rest.sub_(h)
            fhat, nxt = q.get_next_autoregressive_input(
                k, len(PATCH_NUMS), fhat, embedded(q, idx, k))
            if pr is not None:
                pr.advance(nxt, k)
    finally:
        if pr is not None:
            pr.close()
    errors = ((clean.double() - fhat.double()).square().flatten(1).sum(1)
              if clean is not None else None)
    return dict(fhat=fhat, tokens=tokens, metrics=metrics, logits=logits,
                latent_error=float(errors.sum()) if errors is not None else None,
                latent_error_per_image=errors.cpu().tolist() if errors is not None else None)


@torch.no_grad()
def tf_metrics(vae, var, Z, truth, prior, variance, static=None, lam=1.,
               details=True, canonical_lp=None, variance_floor=VARIANCE_FLOOR):
    """Offline true-prefix scores, never a deployable method or policy output.

    canonical_lp may contain the ten full-vocabulary log-probability maps from
    an independently verified official true-prefix forward, to reuse across
    lambda candidates. Otherwise the same cached forward uses truth prefixes.
    """
    _validate(vae, Z, prior, variance, static, lam, variance_floor)
    _truth_shapes(truth, Z)
    if canonical_lp is not None and len(canonical_lp) != len(PATCH_NUMS):
        raise ValueError('canonical_lp must include all scales')
    q = vae.quantize
    rest, fhat = Z.clone(), torch.zeros_like(Z)
    pr = (Prior(var, Z.shape[0])
          if prior == 'V' and lam != 0 and canonical_lp is None else None)
    records = []
    try:
        for k in range(len(PATCH_NUMS)):
            lp = None
            if prior == 'V' and lam != 0:
                lp = (canonical_lp[k].to(Z) if canonical_lp is not None
                      else fn.log_softmax(pr.logits(k), dim=-1))
            elif prior == 'A2' and lam != 0:
                lp = static[k].to(Z)[None, None]
            score, idx = _score(distance(q, rest, k), lp, variance[k], lam, variance_floor)
            records.append(_metrics(score if details else None, idx, truth[k]))
            rest.sub_(contribution(q, truth[k], k))
            fhat, nxt = q.get_next_autoregressive_input(
                k, len(PATCH_NUMS), fhat, embedded(q, truth[k], k))
            if pr is not None:
                pr.advance(nxt, k)
    finally:
        if pr is not None:
            pr.close()
    return records


def bypass(Z):
    """The actual B2 endpoint: no quantization, standardization or arithmetic."""
    return Z


def fusion(Z, Fq, alpha):
    """Frozen channel shrinkage in original coordinates; common alpha is legal."""
    alpha = torch.as_tensor(alpha, device=Z.device, dtype=Z.dtype)
    if alpha.ndim == 1 and alpha.numel() == Z.shape[1]:
        alpha = alpha.reshape(1, Z.shape[1], 1, 1)
    if not torch.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError('fixed fusion alpha must be finite within [0,1]')
    if Fq.shape != Z.shape:
        raise ValueError('candidate and B2 feature shapes must agree')
    return Z + alpha * (Fq - Z)


def fusion_alpha(tau, rho):
    """Registered tau/(tau+rho), with weight zero when both variances are zero."""
    tau, rho = torch.broadcast_tensors(torch.as_tensor(tau), torch.as_tensor(rho))
    if (not torch.isfinite(tau).all() or not torch.isfinite(rho).all()
            or (tau < 0).any() or (rho < 0).any()):
        raise ValueError('fusion variances must be finite and nonnegative')
    total = tau + rho
    safe = torch.where(total > 0, total, torch.ones_like(total))
    return torch.where(total > 0, tau / safe, torch.zeros_like(total))
