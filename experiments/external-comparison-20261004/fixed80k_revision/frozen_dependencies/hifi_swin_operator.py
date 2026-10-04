"""Paid-header SwinJSCC operator for the frozen author HiFi-DiffCom sampler.

This is a registered adaptation, not an upstream SwinJSCC/HiFi implementation.
It fixes the observed channel mask and float32 gain at the receiver.  Candidate
images pass through all encoder/decoder modulation derivatives; upstream's
training-only input detach operations are not appropriate for image gradients.
No transmitter state, source image, or candidate-selected mask enters this API.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Tuple

import torch


@dataclass(frozen=True)
class DecodedSwinContext:
    """One CRC-accepted paid header, plus public frame configuration."""

    total_n: int
    header_n: int
    rate: int
    indices: Tuple[int, ...]
    power: float
    snr_db: float
    crc_accepted: bool
    latent_tokens: int = 256
    latent_width: int = 320

    def __post_init__(self):
        if self.crc_accepted is not True:
            raise ValueError("HiFi must not run with an unaccepted header; use fixed gray")
        if not isinstance(self.indices, tuple):
            raise TypeError("decoded indices must be an immutable tuple")
        if not (math.isfinite(self.power) and self.power > 0):
            raise ValueError("decoded power must be finite and positive")
        if not math.isfinite(self.snr_db):
            raise ValueError("SNR must be finite")
        if self.latent_tokens <= 0 or self.latent_tokens % 2:
            raise ValueError("half-flat complex pairing requires an even token count")
        if not 0 < self.rate <= self.latent_width:
            raise ValueError("invalid active-channel count")
        if len(self.indices) != self.rate or tuple(sorted(set(self.indices))) != self.indices:
            raise ValueError("decoded channels must be sorted, distinct, and have length rate")
        if any(type(i) is not int or not 0 <= i < self.latent_width for i in self.indices):
            raise ValueError("decoded channel index out of range")
        if self.header_n <= 0 or self.data_n + self.header_n != self.total_n:
            raise ValueError("data and paid header must exactly exhaust the total budget")
        if 2 * self.header_n < self.header_information_bits:
            raise ValueError("paid QPSK header cannot fit its own information bits")

    @property
    def data_n(self):
        return self.latent_tokens * self.rate // 2

    @property
    def mask_bits(self):
        return (math.comb(self.latent_width, self.rate) - 1).bit_length()

    @property
    def header_information_bits(self):
        return self.mask_bits + 32 + 16 + 6

    def ledger(self):
        return {"N_total": self.total_n, "N_data": self.data_n,
                "N_header": self.header_n, "E_total": 2 * self.total_n,
                "E_data": 2 * self.data_n, "E_header": 2 * self.header_n,
                "mask_bits": self.mask_bits, "power_bits": 32,
                "crc_bits": 16, "tail_bits": 6,
                "header_information_bits": self.header_information_bits,
                "header_coded_bits": 2 * self.header_n,
                "project_noise_variance_per_real": 10 ** (-self.snr_db / 10),
                "author_noise_variance_per_real": .5 * 10 ** (-self.snr_db / 10)}


def strict_decoded_context(total_n: int, snr_db: float, decoded_header: Mapping,
                           *, crc_accepted: bool) -> DecodedSwinContext:
    """Build only from the decoder's returned header, never TX metadata.

    CRC acceptance is a separate required keyword, so accidentally passing a TX
    dictionary is insufficient.  The caller remains responsible for calling the
    actual header decoder; the numerical operator cannot authenticate a dict.
    """
    try:
        rate, header_n = {1024: (6, 256), 2048: (13, 384)}[int(total_n)]
    except KeyError as exc:
        raise ValueError("this registration only permits exact N1024/N2048") from exc
    if "rate" in decoded_header and int(decoded_header["rate"]) != rate:
        raise ValueError("decoded header rate conflicts with the public frame budget")
    if "width" in decoded_header and int(decoded_header["width"]) != 320:
        raise ValueError("decoded header width differs from the registered architecture")
    return DecodedSwinContext(int(total_n), header_n, rate,
                              tuple(int(i) for i in decoded_header["indices"]),
                              float(decoded_header["power"]), float(snr_db), crc_accepted)


def _modulate(features, scalar, sm_list, bm_list, sigmoid, layer_num):
    """Numerically native gating, including its full input Jacobian."""
    batch, length, _ = features.shape
    condition = features.new_tensor(float(scalar)).reshape(1, 1).expand(batch, -1)
    temporary = features
    for i in range(layer_num):
        temporary = sm_list[i](temporary)
        temporary = temporary * bm_list[i](condition).unsqueeze(1).expand(-1, length, -1)
    modulation = sigmoid(sm_list[-1](temporary))
    return features * modulation, modulation


def encode_before_mask(encoder, image, snr_db, rate):
    """SA+RA features before the discrete top-k, without input stop gradients."""
    features = encoder.patch_embed(image)
    for layer in encoder.layers:
        features = layer(features)
    features = encoder.norm(features)
    features, _ = _modulate(features, snr_db, encoder.sm_list1, encoder.bm_list1,
                            encoder.sigmoid1, encoder.layer_num)
    features, rate_modulation = _modulate(features, rate, encoder.sm_list, encoder.bm_list,
                                         encoder.sigmoid, encoder.layer_num)
    return features, rate_modulation


def decode_with_image_gradient(decoder, features, snr_db):
    """SA+RA decoder with full derivatives through its SNR modulation."""
    features, _ = _modulate(features, snr_db, decoder.sm_list, decoder.bm_list,
                            decoder.sigmoid, decoder.layer_num)
    for layer in decoder.layers:
        features = layer(features)
    batch, _, channels = features.shape
    return features.reshape(batch, decoder.H, decoder.W, channels).permute(0, 3, 1, 2)


class SwinHiFiOperator:
    """Author conditioner API: encode -> forward -> transpose -> decode.

    encode returns unnormalized selected I/Q values. forward divides by the
    *decoded fixed* sqrt(2*power). Thus the author sampler sees complex energy 1;
    measured project waveforms, with energy 2, are divided by sqrt(2) exactly once.
    transpose reverses only this normalization. All channel use outside data is
    in context.ledger(); header symbols never become fictional latent symbols.
    """

    def __init__(self, model, context: DecodedSwinContext):
        if model.training:
            raise ValueError("posterior operator requires an eval-mode Swin model")
        self.model = model
        self.context = context
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self._indices = torch.tensor(context.indices, device=self.device, dtype=torch.long)
        token_offsets = torch.arange(context.latent_tokens // 2, device=self.device)[:, None]
        self._active = (token_offsets * context.latent_width + self._indices[None]).reshape(-1)
        self._half = context.latent_tokens * context.latent_width // 2
        self._scale = torch.tensor(math.sqrt(2 * context.power), device=self.device, dtype=self.dtype)

    def encode(self, data):
        if data.ndim != 4 or data.shape[0] != 1:
            raise ValueError("the registered posterior handles one image at a time")
        features, _ = encode_before_mask(self.model.encoder, data, self.context.snr_db, self.context.rate)
        expected = (1, self.context.latent_tokens, self.context.latent_width)
        if tuple(features.shape) != expected:
            raise ValueError(f"encoder returned {tuple(features.shape)}, expected {expected}")
        flat = features.reshape(1, -1)
        # Canonical order matches native half-flat I/Q pairing, not spatial I/Q.
        selected = torch.stack((flat[:, self._active], flat[:, self._half + self._active]), dim=-1)
        return selected.reshape(1, 2 * self.context.data_n)

    def _validate_data(self, data):
        if tuple(data.shape) != (1, 2 * self.context.data_n):
            raise ValueError("posterior signal length does not match the received paid mask")

    def forward(self, s, cof=None):
        if cof is not None:
            raise ValueError("this registration supports AWGN only")
        self._validate_data(s)
        return s / self._scale

    def transpose(self, observed, cof=None):
        if cof is not None:
            raise ValueError("this registration supports AWGN only")
        self._validate_data(observed)
        return observed * self._scale

    def decode(self, s_hat):
        self._validate_data(s_hat)
        pairs = s_hat.reshape(1, self.context.data_n, 2)
        full = s_hat.new_zeros((1, 2 * self._half))
        # scatter is differentiable w.r.t. all selected received I/Q values.
        full = full.scatter(1, self._active[None], pairs[..., 0])
        full = full.scatter(1, (self._half + self._active)[None], pairs[..., 1])
        features = full.reshape(1, self.context.latent_tokens, self.context.latent_width)
        # Swin's registered receiver exposes clipped RGB. ADJSCC, the author's
        # original HiFi backbone, enforces this domain with its final sigmoid.
        # Match the paired Swin output for initialization and confirming loss.
        return decode_with_image_gradient(self.model.decoder, features, self.context.snr_db).clamp(0, 1)

    def measurement(self, observed_project):
        received = torch.as_tensor(observed_project, device=self.device, dtype=self.dtype)
        if tuple(received.shape) != (self.context.data_n, 2):
            raise ValueError("supply only the actual data waveform [N_data,2]")
        if not torch.isfinite(received).all():
            raise ValueError("received waveform contains nonfinite values")
        received = (received / math.sqrt(2)).reshape(1, -1).detach()
        # Only the fixed observed target is detached; candidate paths remain live.
        with torch.no_grad():
            latent = self.transpose(received)
            base = self.decode(latent)
        return {"x_mse": base.detach(), "ofdm_sig": received, "s_hat": latent.detach(),
                "cof_est": None, "cof_gt": None, "channel_usage": self.context.data_n,
                "resource_ledger": self.context.ledger()}


def qualify_native_parity(model, image, snr_db, rate):
    """Runtime prerequisite: exact numerical parity with the pinned GPU vendor.

    This does not pick a policy or evaluate reconstruction quality. It checks
    original mask-selected features and decoder outputs before any main frames.
    """
    if model.training:
        raise ValueError("qualification must use eval mode")
    with torch.no_grad():
        native, mask = model.encoder(image, float(snr_db), int(rate), model.model)
        unmasked, rate_modulation = encode_before_mask(model.encoder, image, snr_db, rate)
        candidate_mask = torch.zeros_like(mask[:, 0])
        indices = rate_modulation.sum(dim=1).sort(dim=1, descending=True).indices[:, :rate]
        candidate_mask.scatter_(1, indices, 1)
        candidate_mask = candidate_mask[:, None].expand_as(mask)
        if not torch.equal(mask, candidate_mask):
            raise AssertionError("posterior pre-mask path changed the author's selected channels")
        masked = unmasked * mask
        torch.testing.assert_close(masked, native, rtol=1e-6, atol=1e-7)
        native_image = model.decoder(native, float(snr_db), model.model)
        full_gradient_image = decode_with_image_gradient(model.decoder, native, snr_db)
        torch.testing.assert_close(full_gradient_image, native_image, rtol=1e-6, atol=1e-7)
        return {"encoder_max_abs": float((masked - native).abs().max()),
                "decoder_max_abs": float((full_gradient_image - native_image).abs().max()),
                "mask_exact": True, "input_detach_removed_only_for_posterior": True}
