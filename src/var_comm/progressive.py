"""Receiver-accepted prefixes, actual headers, entropy packets and fixed budgets."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .entropy import arithmetic_decode, arithmetic_encode, probability_cdf
from .next_scale_prior import PATCH_NUMS, next_scale_log_probs
from .scale_channel import (
    bits_to_indices, channel_evidence, crc16, crc_accepts, decode_map,
    encode_packet, indices_to_bits, rate_match_indices,
)

RAW_BITS = (1092, 768, 1200, 2028)


def prefix_key(prefix, label):
    digest = hashlib.sha256(f"{int(label)}|{len(prefix)}|".encode())
    for scale in prefix:
        digest.update(np.asarray(scale, dtype="<u2").tobytes())
    return digest.hexdigest()


def split_prefix(tokens, scales):
    values = np.asarray(tokens, dtype=np.int64)
    lengths = [size ** 2 for size in PATCH_NUMS[:scales]]
    if len(values) != sum(lengths):
        raise ValueError("prefix token count does not match the scale schedule")
    return list(np.split(values, np.cumsum(lengths)[:-1]))


def allocation(base_uses, entropy=False):
    header_uses = 140 if entropy else 68
    remaining = 3060 - header_uses - int(base_uses)
    weights = np.array([790, 1222, 2050], dtype=np.int64)
    fine = remaining * weights // weights.sum()
    fine[-1] += remaining - int(fine.sum())
    result = [int(base_uses), *fine.tolist()]
    if min(result) <= 0 or sum(result) + header_uses != 3060:
        raise ValueError("invalid fixed-use allocation")
    return header_uses, result


class PriorCache:
    def __init__(self, vae, var, device, directory):
        self.vae, self.var, self.device = vae, var, device
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.tables, self.cdfs = {}, {}
        self.requests, self.compute_seconds = 0, 0.0

    def table(self, prefix, label):
        self.requests += 1
        key = prefix_key(prefix, label)
        if key not in self.tables:
            received = [torch.tensor(np.asarray(scale), device=self.device, dtype=torch.long)[None] for scale in prefix]
            labels = torch.tensor([label], device=self.device, dtype=torch.long)
            torch.cuda.synchronize(self.device)
            tick = time.perf_counter()
            value = next_scale_log_probs(self.var, self.vae, received, labels)[0].cpu().numpy()
            torch.cuda.synchronize(self.device)
            self.compute_seconds += time.perf_counter() - tick
            self.tables[key] = value
            np.savez(self.directory / f"{key}.npz", log_probs=value, prefix=np.concatenate(prefix).astype(np.uint16),
                     prefix_scales=np.asarray(len(prefix)), label=np.asarray(label))
        return self.tables[key], key

    def cdf(self, prefix, label):
        table, key = self.table(prefix, label)
        if key not in self.cdfs:
            self.cdfs[key] = probability_cdf(table)
        return self.cdfs[key], key


def header_bits(label, mode, lengths=None):
    fields = [indices_to_bits([label], 10), indices_to_bits([mode - 6], 2)]
    if lengths is not None:
        fields.extend(indices_to_bits([length], 12) for length in lengths)
    return np.concatenate(fields)


def decode_header(received, snr, entropy=False):
    payload_length, uses = (48, 140) if entropy else (12, 68)
    information_length = payload_length + 22
    mapping = rate_match_indices(2 * information_length, 2 * uses)
    evidence = channel_evidence(received[:uses], mapping, information_length, snr)
    decoded, _score = decode_map(evidence)
    accepted = crc_accepts(decoded[:-6])
    label = int(bits_to_indices(decoded[:10], 10)[0])
    mode = 6 + int(bits_to_indices(decoded[10:12], 2)[0])
    accepted = accepted and label < 1000 and mode in (7, 8, 9)
    lengths = bits_to_indices(decoded[12:48]).tolist() if entropy else []
    return {"accepted": bool(accepted), "label": label, "mode": mode, "lengths": lengths,
            "decoded_bits": decoded.tolist(), "complex_uses": uses}


def transmit_group(source, label, base_uses, priors, entropy=False, encoded_cache=None):
    header_uses, uses = allocation(base_uses, entropy)
    raw = [indices_to_bits(np.concatenate(source[:6]))] + [indices_to_bits(source[scale - 1]) for scale in (7, 8, 9)]
    wire_payloads, compressed_lengths = [], []
    for group, payload in enumerate(raw):
        coded_source = payload
        stored_length = 0
        if entropy and group > 0:
            prefix = source[:group + 5]
            cdf, key = priors.cdf(prefix, label)
            cache_key = (key, hashlib.sha256(payload.tobytes()).hexdigest())
            if encoded_cache is not None and cache_key in encoded_cache:
                compressed = encoded_cache[cache_key]
            else:
                compressed = arithmetic_encode(source[group + 5], cdf)
                if not np.array_equal(arithmetic_decode(compressed, cdf), source[group + 5]):
                    raise RuntimeError("sender arithmetic roundtrip failed")
                if encoded_cache is not None:
                    encoded_cache[cache_key] = compressed
            if len(compressed) < len(payload):
                coded_source, stored_length = compressed, len(compressed)
        if group > 0:
            compressed_lengths.append(stored_length)
        wire_payloads.append(np.concatenate((coded_source, indices_to_bits([crc16(payload)], 16))))
    header = encode_packet(header_bits(label, 9, compressed_lengths if entropy else None), header_uses)
    blocks = [encode_packet(payload, count, with_crc=False) for payload, count in zip(wire_payloads, uses)]
    waveform = np.concatenate([header["symbols"], *[block["symbols"] for block in blocks]])
    if waveform.shape != (3060, 2):
        raise RuntimeError("grouped transmission exceeds fixed budget")
    ledger = {"header_uses": header_uses, "block_uses": uses, "raw_token_bits": sum(RAW_BITS),
              "wire_source_bits": [len(payload) - 16 for payload in wire_payloads],
              "data_crc_bits": 64, "data_tail_bits": 24, "header_payload_bits": 48 if entropy else 12,
              "header_crc_bits": 16, "header_tail_bits": 6, "compressed_lengths": compressed_lengths,
              "total_complex_uses": 3060}
    return waveform, ledger


def receive_group(received, snr, base_uses, arm, priors, static):
    entropy = arm == "group_entropy"
    header_uses, uses = allocation(base_uses, entropy)
    header = decode_header(received, snr, entropy)
    result = {"header": header, "label": header["label"] if header["accepted"] else None,
              "prefix": [], "last_accepted_scale": 0, "events": []}
    if not header["accepted"] or header["mode"] != 9:
        result["label"] = None
        return result
    position = header_uses
    for group, count in enumerate(uses):
        compressed_length = header["lengths"][group - 1] if entropy and group > 0 else 0
        source_length = compressed_length or RAW_BITS[group]
        if source_length < 1 or source_length > RAW_BITS[group]:
            result["events"].append({"group": group, "accepted": False, "reason": "invalid_length_header"})
            break
        information_length = source_length + 22
        mapping = rate_match_indices(2 * information_length, 2 * count)
        observations = received[position:position + count]
        evidence = channel_evidence(observations, mapping, information_length, snr)
        table, prior_key = None, None
        if group > 0 and not entropy:
            scale = group + 6
            if arm == "group_var":
                table, prior_key = priors.table(result["prefix"], result["label"])
            elif arm == "group_static":
                table = np.broadcast_to(static[scale], (PATCH_NUMS[scale - 1] ** 2, 4096))
            elif arm == "group_ml":
                table = np.full((PATCH_NUMS[scale - 1] ** 2, 4096), -np.log(4096))
            else:
                raise ValueError("unknown matched receiver arm")
        decoded, score = decode_map(evidence, table)
        if compressed_length:
            cdf, prior_key = priors.cdf(result["prefix"], result["label"])
            recovered = arithmetic_decode(decoded[:source_length], cdf)
            payload = indices_to_bits(recovered)
        else:
            payload = decoded[:source_length]
            recovered = bits_to_indices(payload)
        received_crc = int(bits_to_indices(decoded[source_length:source_length + 16], 16)[0])
        accepts = crc16(payload) == received_crc
        event = {"group": group, "accepted": bool(accepts), "prior_key": prior_key,
                 "prior_prefix_scales": len(result["prefix"]), "source_wire_bits": source_length,
                 "complex_uses": count, "position": position, "score": score,
                 "received_sha256": hashlib.sha256(np.asarray(observations).tobytes()).hexdigest(),
                 "decoded_bits": decoded.tolist(), "recovered_tokens": recovered.tolist()}
        result["events"].append(event)
        position += count
        if not accepts:
            break
        if group == 0:
            result["prefix"] = split_prefix(recovered, 6)
        else:
            result["prefix"].append(recovered)
        result["last_accepted_scale"] = len(result["prefix"])
    return result


def transmit_whole(source, label, mode):
    header = encode_packet(header_bits(label, mode), 68)
    body = encode_packet(indices_to_bits(np.concatenate(source[:mode])), 2992)
    return np.concatenate((header["symbols"], body["symbols"]))


def receive_whole(received, snr):
    header = decode_header(received, snr)
    result = {"header": header, "label": header["label"] if header["accepted"] else None,
              "prefix": [], "last_accepted_scale": 0, "events": []}
    if not header["accepted"]:
        return result
    mode = header["mode"]
    source_length = 12 * sum(size ** 2 for size in PATCH_NUMS[:mode])
    information_length = source_length + 22
    mapping = rate_match_indices(2 * information_length, 5984)
    evidence = channel_evidence(received[68:], mapping, information_length, snr)
    decoded, score = decode_map(evidence)
    accepts = crc_accepts(decoded[:-6])
    result["prefix"] = split_prefix(bits_to_indices(decoded[:source_length]), mode)
    result["last_accepted_scale"] = mode if accepts else 0
    result["events"] = [{"accepted": bool(accepts), "decoded_bits": decoded.tolist(), "score": score,
                          "output_uses_unverified_ML_prefix_if_CRC_fails": True}]
    return result


@torch.no_grad()
def complete_image(vae, var, prefix, label, device):
    if label is None:
        return np.full((3, 256, 256), 0.5, dtype=np.float32)
    labels = torch.tensor([label], device=device, dtype=torch.long)
    condition = var.class_emb(labels)
    positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_input = condition[:, None].expand(-1, var.first_l, -1) + var.pos_start + positions[:, :var.first_l]
    latent = condition.new_zeros(1, var.Cvae, PATCH_NUMS[-1], PATCH_NUMS[-1])
    offset = 0
    for block in var.blocks:
        block.attn.kv_caching(True)
    try:
        for scale_index, size in enumerate(PATCH_NUMS):
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
            latent, next_input = vae.quantize.get_next_autoregressive_input(scale_index, len(PATCH_NUMS), latent, embedded)
            if scale_index + 1 < len(PATCH_NUMS):
                next_input = var.word_embed(next_input.reshape(1, var.Cvae, -1).transpose(1, 2)) + positions[:, offset:offset + PATCH_NUMS[scale_index + 1] ** 2]
        return vae.fhat_to_img(latent).clamp(-1, 1).add(1).mul(0.5)[0].cpu().numpy()
    finally:
        for block in var.blocks:
            block.attn.kv_caching(False)
