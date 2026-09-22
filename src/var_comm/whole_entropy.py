"""One continuous VAR-probability arithmetic payload and one terminated FEC data block."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .entropy import FULL, HALF, QUARTER, TOTAL, probability_cdf, validate_cdf, validate_bits
from .next_scale_prior import PATCH_NUMS
from .progressive import complete_image, split_prefix
from .scale_channel import binary, bits_to_indices, channel_evidence, crc_accepts, decode_map, encode_packet, indices_to_bits, rate_match_indices


HEADER_USES = 94
DATA_USES = 2966
MODES = (7, 8, 9)


class ArithmeticEncoder:
    def __init__(self):
        self.low, self.high, self.pending = 0, FULL - 1, 0
        self.bits = []

    def emit(self, bit):
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def encode(self, tokens, cdf):
        raw_values = np.asarray(tokens)
        if raw_values.ndim != 1:
            raise ValueError("invalid arithmetic token/table layout")
        if not np.issubdtype(raw_values.dtype, np.integer):
            try:
                numeric = raw_values.astype(np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("invalid arithmetic token/table layout") from error
            if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError("invalid arithmetic token/table layout")
            raw_values = numeric
        values = raw_values.astype(np.int64, copy=False)
        cdf = validate_cdf(cdf, rows=len(values))
        if np.any(values < 0) or np.any(values >= 4096):
            raise ValueError("invalid arithmetic token/table layout")
        for token, cumulative in zip(values, cdf):
            interval = self.high - self.low + 1
            self.high = self.low + interval * int(cumulative[token + 1]) // TOTAL - 1
            self.low += interval * int(cumulative[token]) // TOTAL
            while True:
                if self.high < HALF:
                    self.emit(0)
                elif self.low >= HALF:
                    self.emit(1)
                    self.low -= HALF
                    self.high -= HALF
                elif self.low >= QUARTER and self.high < 3 * QUARTER:
                    self.pending += 1
                    self.low -= QUARTER
                    self.high -= QUARTER
                else:
                    break
                self.low *= 2
                self.high = self.high * 2 + 1

    def finish(self):
        bit = 0 if self.low < QUARTER else 1
        return np.asarray(self.bits + [bit] + [1 - bit] * (self.pending + 1), dtype=np.uint8)


class ArithmeticDecoder:
    def __init__(self, bits):
        self.bits = validate_bits(bits, name="arithmetic bitstream")
        self.position = 0
        self.low, self.high, self.value = 0, FULL - 1, 0
        for offset in range(32):
            self.value = self.value * 2 + self.read_bit()

    def read_bit(self):
        value = int(self.bits[self.position]) if self.position < len(self.bits) else 0
        self.position += 1
        return value

    def decode(self, cdf):
        cdf = validate_cdf(cdf)
        tokens = []
        for cumulative in cdf:
            interval = self.high - self.low + 1
            scaled = ((self.value - self.low + 1) * TOTAL - 1) // interval
            token = int(np.searchsorted(cumulative, scaled, side="right") - 1)
            if not 0 <= token < 4096:
                raise ValueError("invalid arithmetic state")
            tokens.append(token)
            self.high = self.low + interval * int(cumulative[token + 1]) // TOTAL - 1
            self.low += interval * int(cumulative[token]) // TOTAL
            while True:
                if self.high < HALF:
                    pass
                elif self.low >= HALF:
                    self.value -= HALF
                    self.low -= HALF
                    self.high -= HALF
                elif self.low >= QUARTER and self.high < 3 * QUARTER:
                    self.value -= QUARTER
                    self.low -= QUARTER
                    self.high -= QUARTER
                else:
                    break
                self.low *= 2
                self.high = self.high * 2 + 1
                self.value = self.value * 2 + self.read_bit()
        return np.asarray(tokens, dtype=np.int64)


class VarScaleStream:
    def __init__(self, vae, var, label, device):
        if not 0 <= int(label) < 1000 or vae.training or var.training:
            raise ValueError("VAR source coder requires a valid received class and frozen eval models")
        self.vae, self.var, self.device, self.label = vae, var, device, int(label)

    @torch.no_grad()
    def __enter__(self):
        labels = torch.tensor([self.label], device=self.device, dtype=torch.long)
        self.condition = self.var.class_emb(labels)
        self.positions = self.var.lvl_embed(self.var.lvl_1L) + self.var.pos_1LC
        self.next_input = self.condition[:, None].expand(-1, self.var.first_l, -1) + self.var.pos_start + self.positions[:, :self.var.first_l]
        self.latent = self.condition.new_zeros(1, self.var.Cvae, PATCH_NUMS[-1], PATCH_NUMS[-1])
        self.scale, self.offset, self.hidden = 0, 0, None
        for block in self.var.blocks:
            block.attn.kv_caching(True)
        return self

    @torch.no_grad()
    def logits(self):
        if self.scale >= len(PATCH_NUMS):
            raise ValueError("all VAR scales have already been consumed")
        if self.hidden is None:
            hidden = self.next_input
            conditioned = self.var.shared_ada_lin(self.condition)
            for block in self.var.blocks:
                hidden = block(x=hidden, cond_BD=conditioned, attn_bias=None)
            self.hidden = hidden
        return self.var.get_logits(self.hidden, self.condition).float()

    @torch.no_grad()
    def log_probs(self):
        return torch.log_softmax(self.logits(), dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def advance(self, tokens):
        if self.hidden is None:
            raise ValueError("probabilities must be computed before advancing the source state")
        size = PATCH_NUMS[self.scale]
        values = np.asarray(tokens, dtype=np.int64)
        if values.shape != (size ** 2,) or np.any(values < 0) or np.any(values >= 4096):
            raise ValueError("invalid complete scale tokens")
        chosen = torch.as_tensor(values, device=self.device, dtype=torch.long)[None]
        embedded = self.vae.quantize.embedding(chosen).transpose(1, 2).reshape(1, self.var.Cvae, size, size)
        self.latent, next_input = self.vae.quantize.get_next_autoregressive_input(self.scale, len(PATCH_NUMS), self.latent, embedded)
        self.offset += size ** 2
        self.scale += 1
        if self.scale < len(PATCH_NUMS):
            count = PATCH_NUMS[self.scale] ** 2
            self.next_input = self.var.word_embed(next_input.reshape(1, self.var.Cvae, -1).transpose(1, 2)) + self.positions[:, self.offset:self.offset + count]
        self.hidden = None

    @torch.no_grad()
    def render_suffix(self, render=True):
        while self.scale < len(PATCH_NUMS):
            chosen = self.logits()[0].argmax(dim=-1).cpu().numpy()
            self.advance(chosen)
        if not render:
            return None
        return self.vae.fhat_to_img(self.latent).clamp(-1, 1).add(1).mul(.5)[0].cpu().numpy()

    def __exit__(self, error_type, error, trace):
        for block in self.var.blocks:
            block.attn.kv_caching(False)


def choose_payload(raw_bits, arithmetic_bits):
    raw_bits, arithmetic_bits = binary(raw_bits), binary(arithmetic_bits)
    if 2 <= len(arithmetic_bits) < len(raw_bits) and len(arithmetic_bits) < (1 << 13):
        return arithmetic_bits, len(arithmetic_bits)
    return raw_bits, 0


@torch.no_grad()
def encode_prefixes(vae, var, source, label, device, modes=MODES):
    if tuple(modes) != tuple(sorted(set(modes))) or any(mode not in MODES for mode in modes):
        raise ValueError("only the fixed prefix modes are allowed")
    encoder, outputs = ArithmeticEncoder(), {}
    with VarScaleStream(vae, var, label, device) as stream:
        for scale in range(max(modes)):
            table = probability_cdf(stream.log_probs())
            encoder.encode(source[scale], table)
            stream.advance(source[scale])
            mode = scale + 1
            if mode in modes:
                attempted = encoder.finish()
                raw = indices_to_bits(np.concatenate(source[:mode]))
                payload, length_field = choose_payload(raw, attempted)
                outputs[mode] = {"payload": payload.copy(), "length_field": length_field,
                    "raw_payload_bits": len(raw), "attempted_arithmetic_bits": len(attempted), "raw_fallback": length_field == 0}
    return outputs


def transmit(payload, label, mode):
    if mode not in MODES or not 0 <= int(label) < 1000:
        raise ValueError("invalid source packet mode or class")
    length = int(payload["length_field"])
    raw_length = 12 * sum(size ** 2 for size in PATCH_NUMS[:mode])
    data = binary(payload["payload"])
    if (length == 0 and len(data) != raw_length) or (length != 0 and not 2 <= length == len(data) < raw_length):
        raise ValueError("invalid declared arithmetic/raw payload length")
    header_bits = np.concatenate((indices_to_bits([label], 10), indices_to_bits([mode - 6], 2), indices_to_bits([length], 13)))
    header = encode_packet(header_bits, HEADER_USES)
    body = encode_packet(data, DATA_USES)
    signal = np.concatenate((header["symbols"], body["symbols"]))
    ledger = {"raw_payload_bits": raw_length, "attempted_arithmetic_bits": int(payload["attempted_arithmetic_bits"]),
              "actual_payload_bits": len(data), "arithmetic_length_field": length, "raw_fallback": length == 0,
              "header_payload_bits": 25, "header_crc_bits": 16, "header_tail_bits": 6, "header_coded_bits": 188,
              "data_crc_bits": 16, "data_tail_bits": 6, "data_mother_coded_bits": 2 * (len(data) + 22),
              "data_coded_bits": 5932, "data_information_rate": (len(data) + 22) / 5932,
              "header_uses": HEADER_USES, "data_uses": DATA_USES, "complex_uses": len(signal),
              "total_energy": float(np.square(signal).sum())}
    return signal, ledger


def decode_header(received, snr):
    mapping = rate_match_indices(94, 188)
    bits, score = decode_map(channel_evidence(received[:HEADER_USES], mapping, 47, snr))
    label = int(bits_to_indices(bits[:10], 10)[0])
    mode = 6 + int(bits_to_indices(bits[10:12], 2)[0])
    length = int(bits_to_indices(bits[12:25], 13)[0])
    checksum = bool(crc_accepts(bits[:-6]))
    raw_length = 12 * sum(size ** 2 for size in PATCH_NUMS[:mode]) if mode in MODES else 0
    legal = label < 1000 and mode in MODES and (length == 0 or 2 <= length < raw_length)
    return {"accepted": bool(checksum and legal), "phy_crc_accepted": checksum, "fields_legal": bool(legal),
            "label": label, "mode": mode, "length_field": length, "decoded_bits": bits, "score": score}


def decode_phy(received, snr):
    observations = np.asarray(received, dtype=np.float64)
    if observations.shape != (3060, 2) or not np.isfinite(observations).all():
        raise ValueError("whole entropy RX requires exactly the paid finite waveform")
    header = decode_header(observations, snr)
    if not header["accepted"]:
        return {"header": header, "label": None, "mode": None, "payload": None, "body_crc_accepted": False}
    mode, length = header["mode"], header["length_field"]
    payload_length = length or 12 * sum(size ** 2 for size in PATCH_NUMS[:mode])
    information = payload_length + 22
    mapping = rate_match_indices(2 * information, 2 * DATA_USES)
    decoded, score = decode_map(channel_evidence(observations[HEADER_USES:], mapping, information, snr))
    return {"header": header, "label": header["label"], "mode": mode, "payload": decoded[:payload_length].copy(),
            "body_crc_accepted": bool(crc_accepts(decoded[:-6])), "body_decoded_bits": decoded, "body_score": score}


def payload_key(phy):
    if phy["label"] is None:
        return "header_erasure"
    digest = hashlib.sha256(str((phy["label"], phy["mode"], phy["header"]["length_field"])).encode())
    digest.update(phy["payload"].tobytes())
    return digest.hexdigest()


@torch.no_grad()
def decode_source(phy, vae, var, device, render=True, return_latent=False):
    if phy["label"] is None:
        return {"prefix": [], "image": np.full((3, 256, 256), .5, dtype=np.float32) if render else None,
                "latent": None, "source_complete": False, "arithmetic_padding_reads": 0, "source_error": "header_failure"}
    mode, label = phy["mode"], phy["label"]
    if phy["header"]["length_field"] == 0:
        prefix = split_prefix(bits_to_indices(phy["payload"]), mode)
        image = complete_image(vae, var, prefix, label, device) if render else None
        return {"prefix": prefix, "image": image, "latent": None, "source_complete": True, "arithmetic_padding_reads": 0, "source_error": ""}
    decoder, prefix, error, image, latent = ArithmeticDecoder(phy["payload"]), [], "", None, None
    try:
        with VarScaleStream(vae, var, label, device) as stream:
            for scale in range(mode):
                tokens = decoder.decode(probability_cdf(stream.log_probs()))
                prefix.append(tokens)
                stream.advance(tokens)
            if render or return_latent:
                generated = stream.render_suffix(render=render)
                latent = stream.latent
                if render:
                    image = generated
    except ValueError as failure:
        error = str(failure)
        if render:
            image = complete_image(vae, var, prefix, label, device)
    return {"prefix": prefix, "image": image, "latent": latent if return_latent else None,
            "source_complete": len(prefix) == mode and not error,
            "arithmetic_padding_reads": max(0, decoder.position - len(decoder.bits)), "source_error": error}
