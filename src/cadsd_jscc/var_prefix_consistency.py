"""Receiver-only quantization-cell constraints for a fixed, received VAR prefix."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as functional


@dataclass(frozen=True)
class CorrectionSnapshot:
    latent: torch.Tensor
    steps: int
    elapsed_seconds: float
    objective: float
    anchor_loss: float
    consistency_loss: float


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PrefixConstraints:
    """Keep received residual contributions fixed, even after a cell mismatch.

    Only received tokens and shared quantizer parameters enter this object.
    Scale numbering always refers to the complete pretrained tokenizer.
    """

    def __init__(self, quantizer: torch.nn.Module, received: Sequence[torch.Tensor]):
        self.patch_nums = tuple(int(value) for value in quantizer.v_patch_nums)
        if len(self.patch_nums) < 2 or not 1 <= len(received) <= len(self.patch_nums):
            raise ValueError("received prefix must lie within the full scale schedule")
        self.codebook = quantizer.embedding.weight.detach().float().clone()
        if self.codebook.shape[0] < 2 or not torch.isfinite(self.codebook).all():
            raise ValueError("codebook must contain at least two finite entries")
        self.using_znorm = bool(quantizer.using_znorm)
        self.coordinate_scale = float(self.codebook.square().mean().clamp_min(1e-12))
        self.distance_scale = self.coordinate_scale * self.codebook.shape[1]
        self.received = tuple(tokens.detach().clone() for tokens in received)
        self.batch_size = self.received[0].shape[0]
        self.prior_contributions: list[torch.Tensor] = []
        cumulative = self.codebook.new_zeros(
            self.batch_size, self.codebook.shape[1], self.patch_nums[-1], self.patch_nums[-1]
        )
        with torch.no_grad():
            for scale_index, tokens in enumerate(self.received):
                patch_num = self.patch_nums[scale_index]
                if tokens.shape != (self.batch_size, patch_num ** 2):
                    raise ValueError("received token shape disagrees with the full schedule")
                if tokens.dtype != torch.long or tokens.device != self.codebook.device:
                    raise ValueError("received tokens must be long tensors on the model device")
                if (tokens < 0).any() or (tokens >= len(self.codebook)).any():
                    raise ValueError("received codebook index out of range")
                self.prior_contributions.append(cumulative.clone())
                embedded = functional.embedding(tokens, self.codebook).transpose(1, 2)
                embedded = embedded.reshape(self.batch_size, -1, patch_num, patch_num)
                if scale_index != len(self.patch_nums) - 1:
                    embedded = functional.interpolate(
                        embedded, size=self.patch_nums[-1], mode="bicubic", align_corners=False
                    )
                contribution = quantizer.quant_resi[scale_index / (len(self.patch_nums) - 1)](embedded)
                cumulative = cumulative + contribution
        self.prefix_latent = cumulative.detach()

    def _selected_scales(self, last_only: bool) -> tuple[int, ...]:
        return (len(self.received) - 1,) if last_only else tuple(range(len(self.received)))

    def margins(self, latent: torch.Tensor, *, last_only: bool = False) -> list[torch.Tensor]:
        if latent.shape != self.prefix_latent.shape or latent.dtype != torch.float32:
            raise ValueError("candidate must be FP32 with the original final latent shape")
        scale_indices = self._selected_scales(last_only)
        vectors, targets, lengths = [], [], []
        for scale_index in scale_indices:
            residual = latent - self.prior_contributions[scale_index]
            if scale_index != len(self.patch_nums) - 1:
                residual = functional.interpolate(
                    residual, size=self.patch_nums[scale_index], mode="area"
                )
            flattened = residual.permute(0, 2, 3, 1).reshape(-1, latent.shape[1])
            vectors.append(flattened)
            targets.append(self.received[scale_index].reshape(-1))
            lengths.append(len(flattened))
        vectors_flat = torch.cat(vectors)
        targets_flat = torch.cat(targets)
        if self.using_znorm:
            vectors_flat = functional.normalize(vectors_flat, dim=-1)
            codebook = functional.normalize(self.codebook, dim=-1)
            with torch.no_grad():
                scores = vectors_flat @ codebook.T
                scores.scatter_(1, targets_flat[:, None], -float("inf"))
                rivals = scores.argmax(dim=-1)
            difference = codebook[rivals] - codebook[targets_flat]
            margins = (vectors_flat * difference).sum(dim=-1)
        else:
            codebook = self.codebook
            with torch.no_grad():
                distances = vectors_flat.square().sum(1, keepdim=True) + codebook.square().sum(1)
                distances.addmm_(vectors_flat, codebook.T, beta=1, alpha=-2)
                distances.scatter_(1, targets_flat[:, None], float("inf"))
                rivals = distances.argmin(dim=-1)
            target_embedding, rival_embedding = codebook[targets_flat], codebook[rivals]
            margins = (
                2 * (vectors_flat * (rival_embedding - target_embedding)).sum(-1)
                + target_embedding.square().sum(-1) - rival_embedding.square().sum(-1)
            ) / self.distance_scale
        return [part.reshape(self.batch_size, -1) for part in margins.split(lengths)]

    def loss(self, latent: torch.Tensor, *, last_only: bool = False) -> torch.Tensor:
        return torch.stack([
            margin.relu().mean(dim=1) for margin in self.margins(latent, last_only=last_only)
        ]).mean(dim=0)

    @torch.no_grad()
    def statistics(self, latent: torch.Tensor, *, tolerance: float = 1e-5) -> list[dict]:
        rows = []
        for scale_number, margin in enumerate(self.margins(latent), start=1):
            for batch_index in range(self.batch_size):
                values = margin[batch_index]
                rows.append({
                    "batch_index": batch_index,
                    "scale": scale_number,
                    "positions": values.numel(),
                    "violation_rate": float((values > tolerance).float().mean()),
                    "positive_margin_mean": float(values.relu().mean()),
                    "positive_margin_max": float(values.relu().max()),
                })
        return rows


def correct_latent(
    initial: torch.Tensor,
    constraints: PrefixConstraints,
    *,
    weight: float,
    checkpoints: Sequence[int],
    learning_rate: float,
    last_only: bool = False,
) -> list[CorrectionSnapshot]:
    """Optimize only Z; neither a source image nor its unsent suffix is accepted."""

    checkpoints = tuple(int(step) for step in checkpoints)
    if not checkpoints or sorted(set(checkpoints)) != list(checkpoints) or checkpoints[0] < 1:
        raise ValueError("checkpoints must be positive, increasing and unique")
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("weight and learning rate must be finite and nonnegative/positive")
    initial = initial.detach().float()
    latent = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=learning_rate, foreach=False)
    snapshots = []
    synchronize(latent.device)
    start = time.perf_counter()
    evaluation_seconds = 0.0
    with torch.enable_grad(), torch.autocast(device_type=latent.device.type, enabled=False):
        for step in range(1, checkpoints[-1] + 1):
            optimizer.zero_grad(set_to_none=True)
            consistency = constraints.loss(latent, last_only=last_only)
            anchor = (latent - initial).square().flatten(1).mean(1) / constraints.coordinate_scale
            objective = (anchor + weight * consistency).sum()
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite receiver objective")
            objective.backward()
            if not torch.isfinite(latent.grad).all():
                raise FloatingPointError("nonfinite receiver gradient")
            optimizer.step()
            if step in checkpoints:
                synchronize(latent.device)
                elapsed = time.perf_counter() - start - evaluation_seconds
                evaluation_start = time.perf_counter()
                with torch.no_grad():
                    consistency = constraints.loss(latent, last_only=last_only).mean()
                    anchor = (latent - initial).square().mean() / constraints.coordinate_scale
                    objective = anchor + weight * consistency
                    if not torch.isfinite(objective):
                        raise FloatingPointError("nonfinite corrected latent")
                    snapshots.append(CorrectionSnapshot(
                        latent.detach().clone(), step, elapsed, float(objective),
                        float(anchor), float(consistency),
                    ))
                synchronize(latent.device)
                evaluation_seconds += time.perf_counter() - evaluation_start
    return snapshots


@torch.no_grad()
def complete_received_prefix(
    var: torch.nn.Module,
    vae: torch.nn.Module,
    received: Sequence[torch.Tensor],
    labels: torch.Tensor,
    *,
    seed: int | None = None,
    top_k: int = 900,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Preserve the historical no-CFG argmax stack, or explicitly sample a suffix."""

    patch_nums = tuple(vae.quantize.v_patch_nums)
    if not 1 <= len(received) < len(patch_nums):
        raise ValueError("completion needs a proper received prefix, not a full oracle sequence")
    if seed is not None and (top_k < 1 or not 0 < top_p <= 1):
        raise ValueError("invalid sampling settings")
    generator = torch.Generator(device=labels.device).manual_seed(seed) if seed is not None else None
    condition = var.class_emb(labels)
    level_positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_token_map = (
        condition.unsqueeze(1).expand(-1, var.first_l, -1)
        + var.pos_start.expand(len(labels), var.first_l, -1)
        + level_positions[:, :var.first_l]
    )
    latent = condition.new_zeros(len(labels), var.Cvae, patch_nums[-1], patch_nums[-1])
    current_length = 0
    for block in var.blocks:
        block.attn.kv_caching(True)
    try:
        for scale_index, patch_num in enumerate(patch_nums):
            current_length += patch_num ** 2
            conditioned = var.shared_ada_lin(condition)
            hidden = next_token_map
            for block in var.blocks:
                hidden = block(x=hidden, cond_BD=conditioned, attn_bias=None)
            if scale_index < len(received):
                chosen = received[scale_index]
            else:
                logits = var.get_logits(hidden, condition).float()
                if generator is None:
                    chosen = logits.argmax(dim=-1)
                else:
                    from models.helpers import sample_with_top_k_top_p_

                    chosen = sample_with_top_k_top_p_(
                        logits, top_k=top_k, top_p=top_p, rng=generator, num_samples=1
                    )[:, :, 0]
            embedded = vae.quantize.embedding(chosen).transpose(1, 2)
            embedded = embedded.reshape(len(labels), var.Cvae, patch_num, patch_num)
            latent, next_token_map = vae.quantize.get_next_autoregressive_input(
                scale_index, len(patch_nums), latent, embedded
            )
            if scale_index != len(patch_nums) - 1:
                next_token_map = next_token_map.reshape(len(labels), var.Cvae, -1).transpose(1, 2)
                next_length = patch_nums[scale_index + 1] ** 2
                next_token_map = var.word_embed(next_token_map) + level_positions[
                    :, current_length:current_length + next_length
                ]
    finally:
        for block in var.blocks:
            block.attn.kv_caching(False)
    return latent
