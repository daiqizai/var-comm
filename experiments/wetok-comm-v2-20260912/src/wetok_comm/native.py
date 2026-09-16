"""Native 4x8-bit LFQ bridge; frozen Decoder input gradients stay enabled."""

from pathlib import Path

import torch

from .common import REFERENCE, assets, load_module


def indices_to_features(indices):
    if indices.ndim != 4 or tuple(indices.shape[1:]) != (16, 16, 4):
        raise ValueError('native WeTok group grid must be Bx16x16x4')
    masks = 2 ** torch.arange(8, device=indices.device, dtype=torch.int64)
    bits = indices.to(torch.int64)[..., None].bitwise_and(masks).ne(0)
    return bits.flatten(-2).permute(0, 3, 1, 2).float().mul(2).sub(1).contiguous()


def features_to_indices(features):
    if features.ndim != 4 or tuple(features.shape[1:]) != (32, 16, 16):
        raise ValueError('native features must be Bx32x16x16')
    if not torch.all((features == -1) | (features == 1)):
        raise ValueError('only exact native signs can be serialized as group indices')
    bits = features.permute(0, 2, 3, 1).reshape(len(features), 16, 16, 4, 8).gt(0)
    masks = 2 ** torch.arange(8, device=features.device, dtype=torch.int64)
    return (bits.to(torch.int64) * masks).sum(-1).to(torch.uint8)


def hard_native_st(logits):
    hard = torch.where(logits > 0, torch.ones_like(logits), -torch.ones_like(logits))
    return hard + (logits - logits.detach()) if torch.is_grad_enabled() and logits.requires_grad else hard


class FrozenWeTok:
    def __init__(self, device, part='both'):
        if part not in ('encoder', 'decoder', 'both'):
            raise ValueError('unknown frozen visual part')
        module = load_module('wetok_communication_pinned_adapter', REFERENCE / 'scripts/wetok_adapter.py')
        self.adapter = module.WeTokAdapter(assets(), 'cpu')
        self.codec, self.quantizer = self.adapter.codec, self.adapter.quantize
        self.device = torch.device(device)
        if part in ('encoder', 'both'):
            self.codec.encoder.to(self.device)
            self.quantizer.to(self.device)
        if part in ('decoder', 'both'):
            self.codec.decoder.to(self.device)
        self.codec.eval().requires_grad_(False)
        self.metadata = {**self.adapter.metadata, 'communication_part': part, 'native_shape': [32, 16, 16],
                         'decoder_stochastic': False, 'source_bits': 8192,
                         'training_decode_uses_inference_wrapper': False}

    @torch.inference_mode()
    def encode(self, images01):
        if tuple(images01.shape[1:]) != (3, 256, 256):
            raise ValueError('source images must be Bx3x256x256')
        encoded = self.codec.encoder(images01.to(self.device).float().mul(2).sub(1))
        quantized, auxiliary, indices = self.quantizer(encoded)
        if tuple(encoded.shape[1:]) != (32, 16, 16) or indices.numel() != len(images01) * 1024:
            raise RuntimeError('official WeTok layout changed')
        indices = indices.reshape(len(images01), 16, 16, 4)
        if int(indices.min()) < 0 or int(indices.max()) > 255:
            raise RuntimeError('invalid 8-bit group code')
        official = self.quantizer.decode(indices).permute(0, 3, 1, 2).contiguous()
        native = indices_to_features(indices)
        if not torch.equal(official, native) or float((quantized - native).abs().max()) > 1e-5:
            raise RuntimeError('native bit order or quantization identity differs from the official code')
        return indices.to(torch.uint8)

    def decode(self, features):
        if tuple(features.shape[1:]) != (32, 16, 16):
            raise ValueError('Decoder input shape changed')
        return self.codec.decoder(features.float()).clamp(-1, 1).add(1).mul(.5)
