"""Full, receiver-side next-scale probabilities and separate scoring utilities."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as functional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms


PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
VOCAB_SIZE = 4096
METRICS = (
    "nll_bits_per_token", "ideal_cross_entropy_bits_per_scale",
    "predictive_entropy_bits_per_token", "top1_match_rate",
    "fraction_gt_probability_below_uniform", "bit_marginal_nll_bits_per_token",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preprocess(path: Path) -> tuple[torch.Tensor, str]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = 256 / min(width, height)
        image = transforms.resize(
            image, [round(height * scale), round(width * scale)],
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
        image = transforms.center_crop(image, [256, 256])
        pixels = transforms.pil_to_tensor(image)
    content_sha = hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest()
    return pixels.float().div(127.5).sub(1), content_sha


def load_models(paths: dict, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    source = str(Path(paths["var_source"]).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from models import build_vae_var

    initializers = {module: module.reset_parameters for module in (
        torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm,
        torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
    )}
    try:
        vae, var = build_vae_var(
            device=device, patch_nums=PATCH_NUMS, V=VOCAB_SIZE, Cvae=32, ch=160,
            share_quant_resi=4, num_classes=1000, depth=16, attn_l2_norm=True,
            flash_if_available=False, fused_if_available=False,
        )
    finally:
        for module, initializer in initializers.items():
            module.reset_parameters = initializer
    vae.load_state_dict(torch.load(paths["vae_checkpoint"], map_location="cpu", weights_only=True), strict=True)
    var.load_state_dict(torch.load(paths["var_checkpoint"], map_location="cpu", weights_only=True), strict=True)
    var.cond_drop_rate = 0.0
    return vae.eval().requires_grad_(False), var.eval().requires_grad_(False)


@torch.no_grad()
def next_scale_log_probs(
    var: torch.nn.Module,
    vae: torch.nn.Module,
    received_prefix: Sequence[torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return q_k using r_1:k-1 only, with neither CFG nor sampling truncation."""

    schedule = tuple(vae.quantize.v_patch_nums)
    if schedule != PATCH_NUMS or not 1 <= len(received_prefix) < len(schedule):
        raise ValueError("a proper prefix of the complete official schedule is required")
    if var.training or vae.training or float(var.cond_drop_rate) != 0:
        raise ValueError("models must be frozen eval models with no condition dropout")
    for scale_index, tokens in enumerate(received_prefix):
        if tokens.shape != (len(labels), schedule[scale_index] ** 2) or tokens.dtype != torch.long:
            raise ValueError("received token shape/dtype does not match the prefix")
        if tokens.device != labels.device or (tokens < 0).any() or (tokens >= VOCAB_SIZE).any():
            raise ValueError("received token values or device are invalid")
    target_index = len(received_prefix)
    condition = var.class_emb(labels)
    positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_input = (
        condition[:, None].expand(-1, var.first_l, -1)
        + var.pos_start.expand(len(labels), var.first_l, -1)
        + positions[:, :var.first_l]
    )
    cumulative = condition.new_zeros(len(labels), var.Cvae, schedule[-1], schedule[-1])
    position_offset = 0
    for block in var.blocks:
        block.attn.kv_caching(True)
    try:
        for scale_index in range(target_index + 1):
            patch_num = schedule[scale_index]
            position_offset += patch_num ** 2
            hidden = next_input
            conditioned = var.shared_ada_lin(condition)
            for block in var.blocks:
                hidden = block(x=hidden, cond_BD=conditioned, attn_bias=None)
            if scale_index == target_index:
                logits = var.get_logits(hidden, condition).float()
                probabilities = functional.log_softmax(logits, dim=-1)
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError("full next-scale log probabilities are not finite")
                return probabilities
            embedded = vae.quantize.embedding(received_prefix[scale_index]).transpose(1, 2)
            embedded = embedded.reshape(len(labels), var.Cvae, patch_num, patch_num)
            cumulative, next_input = vae.quantize.get_next_autoregressive_input(
                scale_index, len(schedule), cumulative, embedded
            )
            next_input = next_input.reshape(len(labels), var.Cvae, -1).transpose(1, 2)
            next_length = schedule[scale_index + 1] ** 2
            next_input = var.word_embed(next_input) + positions[:, position_offset:position_offset + next_length]
    finally:
        for block in var.blocks:
            block.attn.kv_caching(False)
    raise RuntimeError("next-scale loop did not reach its target")


def frequency_log_probs(counts: torch.Tensor, pseudocount: float) -> torch.Tensor:
    if counts.ndim != 1 or counts.numel() != VOCAB_SIZE or (counts < 0).any():
        raise ValueError("expected a nonnegative count for every codeword")
    if not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("positive finite smoothing is required")
    mass = counts.double() + pseudocount
    return (mass / mass.sum()).log()


@torch.no_grad()
def score_target_tokens(log_probs: torch.Tensor, targets: torch.Tensor) -> tuple[dict, torch.Tensor]:
    if log_probs.ndim != 3 or log_probs.shape[:2] != targets.shape or targets.dtype != torch.long:
        raise ValueError("expected [batch, positions, vocabulary] probabilities and matching long targets")
    vocabulary = log_probs.shape[-1]
    bits_per_token = int(math.log2(vocabulary))
    if vocabulary != 2 ** bits_per_token or vocabulary < 2:
        raise ValueError("a power-of-two vocabulary is required")
    if (targets < 0).any() or (targets >= vocabulary).any():
        raise ValueError("target index outside codebook")
    probabilities = log_probs.double()
    if not torch.isfinite(probabilities).all() or float(probabilities.logsumexp(-1).abs().max()) > 2e-5:
        raise ValueError("expected finite, normalized, untruncated log probabilities")
    target_log_probs = probabilities.gather(-1, targets[..., None]).squeeze(-1)
    token_nll = -target_log_probs / math.log(2)
    entropy = -(probabilities.exp() * probabilities).sum(-1) / math.log(2)
    marginal_nll = torch.zeros_like(target_log_probs)
    indices = torch.arange(vocabulary, device=probabilities.device)
    for shift in range(bits_per_token - 1, -1, -1):
        ones = ((indices >> shift) & 1).bool()
        log_one = probabilities[..., ones].logsumexp(-1)
        log_zero = probabilities[..., ~ones].logsumexp(-1)
        target_one = ((targets >> shift) & 1).bool()
        marginal_nll -= torch.where(target_one, log_one, log_zero) / math.log(2)
    summary = {
        "nll_bits_per_token": token_nll.mean(1),
        "ideal_cross_entropy_bits_per_scale": token_nll.sum(1),
        "predictive_entropy_bits_per_token": entropy.mean(1),
        "top1_match_rate": (probabilities.argmax(-1) == targets).double().mean(1),
        "fraction_gt_probability_below_uniform": (target_log_probs < -math.log(vocabulary) - 1e-7).double().mean(1),
        "bit_marginal_nll_bits_per_token": marginal_nll.mean(1),
    }
    return {name: values.cpu().numpy() for name, values in summary.items()}, token_nll.cpu()


def paired_mean_interval(differences: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or not len(differences) or not np.isfinite(differences).all():
        raise ValueError("expected one finite paired value per source image")
    indices = np.random.default_rng(seed).integers(0, len(differences), (resamples, len(differences)))
    lower, upper = np.quantile(differences[indices].mean(1), [0.025, 0.975])
    return float(differences.mean()), float(lower), float(upper)


def numerical_selfcheck() -> dict:
    targets = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    uniform = torch.full((1, 4, 4), -math.log(4), dtype=torch.float64)
    metrics, token_nll = score_target_tokens(uniform, targets)
    assert np.allclose(metrics["nll_bits_per_token"], 2)
    assert np.allclose(metrics["bit_marginal_nll_bits_per_token"], 2)
    assert np.allclose(token_nll.numpy(), 2)
    mass = torch.tensor([[[0.45, 0.05, 0.05, 0.45]]], dtype=torch.float64)
    correlated, _ = score_target_tokens(mass.log(), torch.zeros(1, 1, dtype=torch.long))
    assert np.allclose(correlated["nll_bits_per_token"], -math.log2(0.45))
    assert np.allclose(correlated["bit_marginal_nll_bits_per_token"], 2)
    frequencies = frequency_log_probs(torch.zeros(VOCAB_SIZE, dtype=torch.long), 0.5)
    assert torch.allclose(frequencies, torch.full_like(frequencies, -math.log(VOCAB_SIZE)))
    mean, lower, upper = paired_mean_interval(np.array([1., 1., 1.]), seed=17, resamples=100)
    assert mean == lower == upper == 1
    return {"uniform_logloss": "PASS", "joint_vs_bit_marginal": "PASS",
            "positive_frequency_smoothing": "PASS", "paired_image_bootstrap": "PASS"}
