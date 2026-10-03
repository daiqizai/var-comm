"""Official Swin SA+RA modules with explicit per-image channel accounting.

The upstream neural modules are unmodified. Training keeps the original input
detach in ModNets; posterior image gradients belong to the separate operator.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from torch import nn

MODEL = 'SwinJSCC_w/_SAandRA'
WIDTH, TOKENS = 320, 256


class _UniqueChannelSelect(torch.autograd.Function):
    """Gather unique channel indices with a deterministic, collision-free VJP.

    Torch 1.12 implements gather's generic backward via atomic scatter-add,
    even for unique indices. Here every destination is written exactly once,
    so plain scatter is the exact derivative and needs no atomic reduction.
    """
    @staticmethod
    def forward(ctx, features, indices):
        if indices.ndim != 2 or indices.shape[0] != features.shape[0]:
            raise RuntimeError('Unique channel selection shape differs')
        if indices.shape[1] > 1 and not bool((indices[:, 1:] > indices[:, :-1]).all()):
            raise RuntimeError('Channel selection requires sorted unique indices')
        ctx.save_for_backward(indices)
        ctx.feature_shape = tuple(features.shape)
        return features.gather(2, indices[:, None, :].expand(-1, features.shape[1], -1))

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        restored = gradient.new_zeros(ctx.feature_shape)
        restored.scatter_(2, indices[:, None, :].expand(-1, gradient.shape[1], -1), gradient)
        return restored, None


def build_official(vendor_root, device='cuda:0'):
    vendor_root = Path(vendor_root).resolve()
    old = sys.modules.get('net')
    if old is not None and any(not Path(p).resolve().is_relative_to(vendor_root) for p in old.__path__):
        raise RuntimeError('Another net package shadows the registered Swin vendor')
    sys.path.insert(0, str(vendor_root))
    from net.network import SwinJSCC
    common = dict(model=MODEL, img_size=(256, 256), C=None, window_size=8,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True)
    encoder = dict(common, patch_size=2, in_chans=3, embed_dims=[128, 192, 256, 320],
        depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10])
    decoder = dict(common, embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4])
    args = SimpleNamespace(model=MODEL, channel_type='awgn', multiple_snr=','.join(map(str, range(1, 14))),
        C='6,13', distortion_metric='MSE')
    config = SimpleNamespace(encoder_kwargs=encoder, decoder_kwargs=decoder, logger=None,
        device=torch.device(device), pass_channel=True, downsample=4, norm=False)
    native = SwinJSCC(args, config).to(device)
    native.encoder.update_resolution(256, 256)
    native.decoder.update_resolution(16, 16)
    native.H = native.W = 256
    return SwinCodec(native)


def pack_data(features, mask):
    """Return energy-two complex IQ, per-image power, and sorted active channels."""
    if features.shape != mask.shape or tuple(features.shape[1:]) != (TOKENS, WIDTH):
        raise RuntimeError('Swin feature layout changed')
    if not torch.equal(mask, mask[:, :1].expand_as(mask)):
        raise RuntimeError('Swin mask is not channel-wise')
    active = mask[:, 0].bool()
    counts = active.sum(dim=1)
    count = int(counts[0])
    if count < 1 or not torch.equal(counts, counts[:1].expand_as(counts)):
        raise RuntimeError('A packed batch needs one nonzero rate')
    indices = torch.arange(WIDTH, device=features.device).expand(len(features), -1)[active].reshape(len(features), count)
    power = features.square().sum(dim=(1, 2)) / (TOKENS * count)
    if not bool(torch.isfinite(power).all()) or not bool((power > 0).all()):
        raise RuntimeError('Swin TX power is invalid')
    selected = _UniqueChannelSelect.apply(features, indices).flatten(1)
    normalized = selected / power.sqrt()[:, None]
    # Native channel pairs first/second flattened halves, not adjacent values.
    half = normalized.shape[1] // 2
    iq = torch.stack((normalized[:, :half], normalized[:, half:]), dim=-1)
    return iq, power, indices


def unpack_data(observed, power, indices):
    """Use only RX-decoded context to restore the author's decoder coordinates."""
    batch, length, iq = observed.shape
    count = indices.shape[1]
    if iq != 2 or length != TOKENS * count // 2 or power.shape != (batch,):
        raise RuntimeError('Received Swin shape/context differs')
    if indices.shape[0] != batch or bool((indices < 0).any()) or bool((indices >= WIDTH).any()):
        raise RuntimeError('Received Swin mask range differs')
    if count > 1 and not bool((indices[:, 1:] > indices[:, :-1]).all()):
        raise RuntimeError('Received Swin channels must be sorted and unique')
    if not bool(torch.isfinite(power).all()) or not bool((power > 0).all()):
        raise RuntimeError('Received Swin power is invalid')
    selected = torch.cat((observed[:, :, 0], observed[:, :, 1]), dim=1).reshape(batch, TOKENS, count)
    restored = selected * power.sqrt()[:, None, None]
    full = observed.new_zeros((batch, TOKENS, WIDTH))
    return full.scatter(2, indices[:, None, :].expand(-1, TOKENS, -1), restored)


class SwinCodec(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def encode_features(self, image, snr, channels):
        if tuple(image.shape[1:]) != (3, 256, 256) or not 1 <= int(channels) <= WIDTH:
            raise ValueError('Registered 256 RGB and positive integer rate required')
        return self.native.encoder(image, float(snr), int(channels), MODEL)

    def encode_data(self, image, snr, channels):
        return pack_data(*self.encode_features(image, snr, channels))

    def receive_data(self, observed, snr, power, indices):
        return self.native.decoder(unpack_data(observed, power, indices), float(snr), MODEL).clamp(0, 1)

    def training_forward(self, image, snr, channels, standard_noise):
        signal, power, indices = self.encode_data(image, snr, channels)
        if standard_noise.shape != signal.shape:
            raise RuntimeError('Registered training noise shape differs')
        observed = signal + standard_noise / (10 ** (float(snr) / 20))
        return self.receive_data(observed, snr, power, indices)

