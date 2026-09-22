from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from var_comm.progressive import receive_whole, transmit_whole
from var_comm.study import seeded_noise


class ContinuousDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = copy.deepcopy(vae.post_quant_conv)
        self.decoder = copy.deepcopy(vae.decoder)
        self.requires_grad_(True)

    def forward(self, latent):
        return self.decoder(self.post_quant_conv(latent)).clamp(-1, 1).add(1).mul(0.5)


@torch.no_grad()
def original_rgb(vae, latent):
    return vae.fhat_to_img(latent).clamp(-1, 1).add(1).mul(0.5)


@torch.no_grad()
def quantized_latent(vae, scales):
    sizes = tuple(vae.quantize.v_patch_nums)
    if len(scales) != len(sizes):
        raise ValueError("Fq must use all true scales")
    embeddings = [vae.quantize.embedding(indices).transpose(1, 2).reshape(
        len(indices), vae.Cvae, size, size) for size, indices in zip(sizes, scales)]
    return vae.quantize.embed_to_fhat(embeddings, all_to_max_scale=True, last_one=True)


@torch.no_grad()
def complete_latent(vae, var, prefix, label, device):
    if label is None:
        return None
    sizes = tuple(vae.quantize.v_patch_nums)
    labels = torch.tensor([label], device=device, dtype=torch.long)
    condition = var.class_emb(labels)
    positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_input = condition[:, None].expand(-1, var.first_l, -1) + var.pos_start + positions[:, :var.first_l]
    latent = condition.new_zeros(1, var.Cvae, sizes[-1], sizes[-1])
    offset = 0
    for block in var.blocks:
        block.attn.kv_caching(True)
    try:
        for scale_index, size in enumerate(sizes):
            offset += size ** 2
            hidden = next_input
            conditioned = var.shared_ada_lin(condition)
            for block in var.blocks:
                hidden = block(x=hidden, cond_BD=conditioned, attn_bias=None)
            if scale_index < len(prefix):
                chosen = torch.tensor(np.asarray(prefix[scale_index]), device=device, dtype=torch.long)[None]
            else:
                chosen = var.get_logits(hidden, condition).float().argmax(dim=-1)
            embedded = vae.quantize.embedding(chosen).transpose(1, 2).reshape(1, var.Cvae, size, size)
            latent, next_input = vae.quantize.get_next_autoregressive_input(scale_index, len(sizes), latent, embedded)
            if scale_index + 1 < len(sizes):
                next_input = var.word_embed(next_input.reshape(1, var.Cvae, -1).transpose(1, 2)) + positions[:, offset:offset + sizes[scale_index + 1] ** 2]
        return latent
    finally:
        for block in var.blocks:
            block.attn.kv_caching(False)


def normalize_enhancement(coordinates):
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("enhancement coordinates must be [batch,complex_uses,2]")
    if not torch.isfinite(coordinates).all():
        raise FloatingPointError("nonfinite enhancement coordinates")
    mean_square = coordinates.square().mean(dim=(1, 2), keepdim=True)
    if not torch.isfinite(mean_square).all():
        raise FloatingPointError("enhancement energy overflow or nonfinite value")
    active = mean_square > 1e-20
    normalized = coordinates / mean_square.clamp_min(1e-20).sqrt()
    if active.any() and not torch.isfinite(normalized[active.expand_as(normalized)]).all():
        raise FloatingPointError("nonfinite normalized enhancement")
    return torch.where(active, normalized, torch.ones_like(normalized))


def enhancement_noise(image_id, seed, uses):
    if uses not in (512, 1024):
        raise ValueError("only registered additional budgets are supported")
    identifier = "VAR-LATENT-ENHANCEMENT-v1/enhancement|" + str(image_id)
    return seeded_noise(identifier, seed, (1024, 2))[:uses].copy()


def append_observations(base_waveform, enhancement_waveform, image_id, seed, snr_db):
    base = np.asarray(base_waveform)
    addition = np.asarray(enhancement_waveform)
    if base.shape != (3060, 2) or addition.shape not in ((512, 2), (1024, 2)):
        raise ValueError("resource ledger mismatch")
    if not np.isfinite(base).all() or not np.isfinite(addition).all() or not np.isfinite(float(snr_db)):
        raise FloatingPointError("nonfinite waveform or SNR")
    deviation = np.sqrt(10 ** (float(snr_db) / 10))
    base_received = base + seeded_noise(image_id, seed, base.shape) / deviation
    added_received = addition + enhancement_noise(image_id, seed, len(addition)) / deviation
    return np.concatenate((base_received, added_received)), np.concatenate((base, addition))


def base_transmission(source, label, image_id, seed, snr_db):
    waveform = transmit_whole(source, label, 8)
    if not np.isfinite(waveform).all() or not np.isfinite(float(snr_db)):
        raise FloatingPointError("nonfinite base waveform or SNR")
    observation = waveform + seeded_noise(image_id, seed, waveform.shape) / np.sqrt(10 ** (float(snr_db) / 10))
    return waveform, observation, receive_whole(observation, snr_db)


def observed_status(reception):
    accepted = bool(reception["header"]["accepted"])
    body_ok = accepted and bool(reception["events"][0]["accepted"])
    return np.asarray([float(accepted), float(body_ok),
                       float(reception["header"]["mode"]) if accepted else 0.0], dtype=np.float32)
