"""Fixed-budget digital prefix and independently transmitted source residual."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from .next_scale_prior import PATCH_NUMS
from .progressive import decode_header, header_bits, split_prefix
from .scale_channel import (
    bits_to_indices, channel_evidence, convolutional_encode, crc_accepts,
    decode_map, encode_packet, indices_to_bits, rate_match_indices,
)


@dataclass(frozen=True)
class HybridBudget:
    mode: int = 7
    header_uses: int = 68
    digital_uses: int = 1882
    analog_uses: int = 1110

    def __post_init__(self):
        if (self.mode, self.header_uses, self.digital_uses, self.analog_uses) != (7, 68, 1882, 1110):
            raise ValueError("this registered experiment permits exactly one resource split")

    @property
    def digital_end(self):
        return self.header_uses + self.digital_uses

    @property
    def source_bits(self):
        return 12 * sum(size ** 2 for size in PATCH_NUMS[:self.mode])

    def ledger(self):
        return {"mode": self.mode, "source_bits": self.source_bits,
                "header_payload_bits": 12, "header_crc_bits": 16, "header_tail_bits": 6,
                "body_crc_bits": 16, "body_tail_bits": 6,
                "header_uses": self.header_uses, "digital_uses": self.digital_uses,
                "analog_uses": self.analog_uses, "total_complex_uses": 3060,
                "header_energy": 2 * self.header_uses, "digital_energy": 2 * self.digital_uses,
                "analog_energy": 2 * self.analog_uses, "total_energy": 6120,
                "body_information_code_rate": (self.source_bits + 22) / (2 * self.digital_uses)}


def encode_digital(tokens, label, budget=HybridBudget()):
    source = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if source.size * 12 != budget.source_bits or not 0 <= int(label) < 1000:
        raise ValueError("invalid registered prefix or class")
    header = encode_packet(header_bits(int(label), budget.mode), budget.header_uses)
    body = encode_packet(indices_to_bits(source), budget.digital_uses)
    return np.concatenate((header["symbols"], body["symbols"]))


def receive_digital(received, snr_db, budget=HybridBudget()):
    observed = np.asarray(received, dtype=np.float64)
    if observed.shape != (budget.digital_end, 2) or not np.isfinite(observed).all():
        raise ValueError("receiver expects exactly the finite digital observation, not source data")
    header = decode_header(observed[:budget.header_uses], float(snr_db))
    usable = bool(header["accepted"] and header["mode"] == budget.mode)
    result = {"header": header, "header_usable": usable, "label": header["label"] if usable else None,
              "prefix": [], "tokens": None, "crc_accepted": False, "received_distance": 0.0}
    if not usable:
        return result
    length = budget.source_bits + 22
    mapping = rate_match_indices(2 * length, 2 * budget.digital_uses)
    body = observed[budget.header_uses:]
    evidence = channel_evidence(body, mapping, length, float(snr_db))
    decoded, score = decode_map(evidence)
    tokens = bits_to_indices(decoded[:budget.source_bits])
    reencoded = (1.0 - 2.0 * convolutional_encode(decoded)[mapping].astype(np.float64)).reshape(-1, 2)
    distance = np.mean((body - reencoded) ** 2) * 10 ** (float(snr_db) / 10)
    result.update(tokens=tokens, prefix=split_prefix(tokens, budget.mode),
                  crc_accepted=bool(crc_accepts(decoded[:-6])), decoded_bits=decoded,
                  received_distance=float(distance), score=float(score))
    return result


def receiver_features(result, snr_db):
    return np.asarray([float(snr_db) / 20, float(result["crc_accepted"]),
                       np.log1p(result["received_distance"])], dtype=np.float32)


def normalize_analog(active):
    if active.ndim != 2 or active.shape[1] != 2220:
        raise ValueError("exactly 1110 I/Q pairs are required")
    power = active.float().square().mean(1, keepdim=True)
    if not torch.isfinite(power).all():
        raise FloatingPointError("nonfinite channel energy")
    normalized = active / power.clamp_min(1e-12).sqrt()
    fallback = torch.ones_like(normalized)
    return torch.where(power > 1e-12, normalized, fallback)


class ResidualLink(nn.Module):
    def __init__(self, specification, arm):
        super().__init__()
        if arm not in ("add", "snr_gain", "reliability_gain"):
            raise ValueError("unregistered fusion arm")
        legacy = Path(__file__).resolve().parents[1]
        source = legacy / 'cadsd_jscc/strong_jscc.py'
        if hashlib.sha256(source.read_bytes()).hexdigest() != specification["legacy_backbone_sha256"]:
            raise RuntimeError("read-only residual building blocks changed")
        if str(legacy) not in sys.path:
            sys.path.insert(0, str(legacy))
        from cadsd_jscc.strong_jscc import SNRConditionEmbedding, StrongJSCCEncoder, StrongJSCCDecoder

        self.arm = arm
        arguments = {name: tuple(specification[name]) if name.startswith("stage_") else specification[name]
                     for name in ("latent_channels", "stage_channels", "stage_blocks", "condition_dim")}
        self.condition = SNRConditionEmbedding(specification["condition_dim"])
        self.encoder = StrongJSCCEncoder(**arguments)
        self.decoder = StrongJSCCDecoder(**arguments)
        pairs = torch.div(torch.arange(1110) * 1280, 1110, rounding_mode="floor")
        indices = torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)
        self.register_buffer("active_indices", indices)
        if arm != "add":
            self.gain = nn.Sequential(nn.Linear(3, specification["gain_hidden"]), nn.SiLU(),
                                      nn.Linear(specification["gain_hidden"], 3))
            nn.init.zeros_(self.gain[-1].weight)
            nn.init.zeros_(self.gain[-1].bias)

    def encode(self, residual, snrs):
        latent = self.encoder(residual, self.condition(snrs)).flatten(1)
        return normalize_analog(latent.index_select(1, self.active_indices))

    def decode(self, observed, snrs):
        if observed.ndim != 2 or observed.shape[1] != 2220:
            raise ValueError("invalid received analog symbol count")
        full = observed.new_zeros(len(observed), 2560)
        full = full.index_copy(1, self.active_indices, observed)
        return 2 * self.decoder(full.reshape(-1, 10, 16, 16), self.condition(snrs)) - 1

    def fuse(self, base, correction, features, header_usable):
        if self.arm == "add":
            gain = correction.new_ones(len(correction), 3)
        else:
            actual = features
            if self.arm == "snr_gain":
                actual = torch.cat((features[:, :1], torch.zeros_like(features[:, 1:])), dim=1)
            gain = 2 * self.gain(actual).sigmoid()
        image = (base + gain[:, :, None, None] * correction).clamp(0, 1)
        image = torch.where(header_usable[:, None, None, None], image, torch.full_like(image, 0.5))
        return image, gain

    def forward(self, residual, base, snrs, standard_noise, features, header_usable):
        transmitted = self.encode(residual, snrs)
        if standard_noise.shape != transmitted.shape:
            raise ValueError("analog noise must cover only the registered active symbols")
        observed = transmitted + standard_noise / torch.pow(10.0, snrs[:, None] / 20)
        image, gain = self.fuse(base, self.decode(observed, snrs), features, header_usable)
        return image, transmitted, gain
