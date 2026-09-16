"""Frozen WeTok ImageNet/16 codec adapter: full-token reconstruction only.

The official model wrapper instantiates training losses and Lightning. This
inference-only adapter instead imports the *unmodified*, pinned official
Encoder, Decoder, and LFQ implementations, and strictly loads every inference
parameter from the checkpoint EMA namespace. No adversarial/perceptual loss,
optimizer, training harness, or generation model is instantiated.

Only integer code indices cross encode/decode. In particular, the encoder
latent and original image are NOT retained in the opaque token object.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml


SOURCE_COMMIT = "caa2ad7e709cdabe8432bead448f7514def13919"
HF_REVISION = "85fc6eb084d458b8d4fa3a32d541379e95b2bf87"
CHECKPOINT_BYTES = 3763537210
CHECKPOINT_SHA256 = "e40a0bcdefd8509b2201e93a9f5007d38c8fba9423c17a98b0e45f9e8763e696"
EXPECTED_DDCONFIG = {
    "double_z": False,
    "z_channels": 32,
    "resolution": 128,  # Unused spatial-size attribute, NOT the input crop size.
    "in_channels": 3,
    "out_ch": 3,
    "ch": 256,
    "ch_mult": [1, 1, 2, 2, 4],
    "num_res_blocks": 4,
}
CONFIG_SHA256 = "e82931ca9144cd2781e10def997cdb6995ac38ebbdd00b25beae08d4b8517fc5"
SOURCE_SHA256 = {
    "model": "daf6f2e66de765392045bad0d42dc049484778d38706eef5080b0761a14875ee",
    "quantizer": "b57a37a96c11ae44b58cd48d2946c9098b56534cda4be697b6500c48783edb6d",
}
SOURCE_FILES = {
    "model": "src/WeTok/modules/diffusionmodules/improved_model.py",
    "quantizer": "src/WeTok/modules/vqvae/lookup_free_quantize.py",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_ema_state(codec: nn.Module, state: dict[str, Any]):
    """Reproduce official LitEma.copy_to for ALL encoder/decoder parameters.

    LitEma stores each parameter as ``model_ema.`` + name with periods removed.
    Never fall back to random, partial, or non-EMA inference weights. Runtime
    buffers (if present) come from the ordinary model state as in ema_scope.
    """
    parameters = dict(codec.named_parameters())
    mapped: dict[str, torch.Tensor] = {}
    ema_keys = []
    for name, template in codec.state_dict().items():
        source_key = (
            "model_ema." + name.replace(".", "") if name in parameters else name
        )
        if source_key not in state:
            raise RuntimeError(f"WeTok required inference weight missing: {source_key}")
        tensor = state[source_key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"WeTok inference state is not a tensor: {source_key}")
        if tuple(tensor.shape) != tuple(template.shape):
            raise RuntimeError(
                f"WeTok shape mismatch {source_key}: {tuple(tensor.shape)} != "
                f"{tuple(template.shape)}"
            )
        mapped[name] = tensor
        if name in parameters:
            ema_keys.append(source_key)
    if not ema_keys or len(ema_keys) != len(parameters):
        raise RuntimeError("Incomplete WeTok EMA parameter mapping")
    return mapped, ema_keys


class WeTokAdapter:
    def __init__(self, config: dict[str, Any], device: str | torch.device):
        self.device = torch.device(device)
        self.source = Path(config["wetok_source"]).resolve()
        config_path = Path(config["wetok_config"]).resolve()
        checkpoint_path = Path(config["wetok_checkpoint"]).resolve()
        if (self.source / "PINNED_COMMIT").read_text().strip() != SOURCE_COMMIT:
            raise RuntimeError("Wrong WeTok source revision")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
            raise RuntimeError("WeTok checkpoint size differs from pinned public release")
        actual_sha = _sha256(checkpoint_path)
        if actual_sha != CHECKPOINT_SHA256:
            raise RuntimeError("WeTok checkpoint SHA256 differs from pinned public release")
        if _sha256(config_path) != CONFIG_SHA256:
            raise RuntimeError("WeTok config bytes differ from the pinned public config")
        cfg = yaml.safe_load(config_path.read_text())
        p = cfg["model"]["init_args"]
        if cfg["model"]["class_path"] != "src.WeTok.models.lfqgan.VQModel":
            raise ValueError("Unsupported WeTok model class")
        if p["ddconfig"] != EXPECTED_DDCONFIG:
            raise ValueError("Not the specified WeTok ImageNet/downsample16 architecture")
        for key, expected in {"n_embed": 256, "embed_dim": 32, "num_codebooks": 4,
                              "use_ema": True}.items():
            if p.get(key) != expected:
                raise ValueError(f"Wrong WeTok setting {key}={p.get(key)!r}")
        for key in ("gan_decoder", "use_GFQ", "token_factorization"):
            if p.get(key, False):
                raise ValueError(f"Unexpected setting for the specified release: {key}")
        self.num_codebooks = int(p["num_codebooks"])
        self.codebook_size = int(p["n_embed"])
        self.bits_per_index = int(math.log2(self.codebook_size))
        self.embed_dim = int(p["embed_dim"])
        module_paths = {k: self.source / v for k, v in SOURCE_FILES.items()}
        for key, path in module_paths.items():
            if _sha256(path) != SOURCE_SHA256[key]:
                raise RuntimeError(f"WeTok pinned source hash mismatch: {path}")
        model_code = _import_file("_wetok_pinned_image_codec", module_paths["model"])
        quant_code = _import_file("_wetok_pinned_lfq", module_paths["quantizer"])
        # Meta construction avoids allocating/random-initializing a second full
        # model in host memory while the (EMA+training) checkpoint is loaded.
        with torch.device("meta"):
            self.codec = nn.Module()
            self.codec.encoder = model_code.Encoder(**p["ddconfig"])
            self.codec.decoder = model_code.Decoder(**p["ddconfig"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True,
                                weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Expected the official Lightning checkpoint state_dict")
        selected, ema_keys = _select_ema_state(self.codec, checkpoint["state_dict"])
        self.codec.load_state_dict(selected, strict=True, assign=True)
        self.codec = self.codec.to(device=self.device, dtype=torch.float32).eval()
        self.codec.requires_grad_(False)
        self.quantize = quant_code.LFQ(
            dim=self.embed_dim, codebook_size=self.codebook_size,
            num_codebooks=self.num_codebooks,
            sample_minimization_weight=p["sample_minimization_weight"],
            batch_maximization_weight=p["batch_maximization_weight"],
        ).to(self.device).eval().requires_grad_(False)
        del selected, checkpoint
        self.metadata = {
            "model": "WeTok-ImageNet-downsample16",
            "purpose": "full_codec_reference_only_no_prefix_no_completion",
            "source_repository": "https://github.com/zhuangshaobin/WeTok",
            "source_commit": SOURCE_COMMIT,
            "source_file_sha256": {k: _sha256(v) for k, v in module_paths.items()},
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "checkpoint_sha256": actual_sha,
            "checkpoint_sha256_verified": True,
            "checkpoint_hf_repository": "GrayShine/WeTok",
            "checkpoint_hf_revision": HF_REVISION,
            "checkpoint_hf_path": "ImageNet/downsample16/WeTok.ckpt",
            "weight_variant": "EMA_encoder_and_decoder",
            "ema_parameter_tensor_count": len(ema_keys),
            "inference_parameter_count": sum(p.numel() for p in self.codec.parameters()),
            "codebook_size_per_group": self.codebook_size,
            "num_codebooks": self.num_codebooks,
            "bits_per_index": self.bits_per_index,
            "bits_per_spatial_position": self.bits_per_index * self.num_codebooks,
            "fixed_input_shape": [3, 256, 256],
            "logical_index_shape_per_image": [16, 16, 4],
            "raw_bits_per_image": 8192,
            "precision": "float32_autocast_disabled",
            "training_performed": False,
            "label_side_information_bits": 0,
            "entropy_coding": False,
            "discrete_roundtrip": "Only integer LFQ group indices retained; LFQ.decode reconstructs signs",
            "checkpoint_loading": "weights_only=True; mmap=True; strict EMA inference state",
            "excluded_checkpoint_namespaces": "training loss/discriminator/optimizer/ordinary non-EMA model weights",
        }

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> dict[str, Any]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError("This bounded WeTok profile only accepts Bx3x256x256")
        if x.device != self.device:
            x = x.to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            h = self.codec.encoder(x.float())
            if tuple(h.shape[1:]) != (32, 16, 16):
                raise RuntimeError(f"Unexpected actual WeTok encoder shape: {tuple(h.shape)}")
            quantized, _, indices = self.quantize(h)
            expected_grid = (h.shape[0], h.shape[2], h.shape[3], self.num_codebooks)
            if indices.ndim != 1 or indices.numel() != math.prod(expected_grid):
                raise RuntimeError(f"Unexpected actual WeTok index shape: {tuple(indices.shape)}")
            if indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("WeTok indices must be integer")
            codes = self.quantize.decode(indices.reshape(expected_grid)).permute(0, 3, 1, 2)
            roundtrip_error = float((codes - quantized).abs().max().item())
            # Official LFQ retains an STE expression in evaluation, possibly
            # introducing tiny fp32 cancellation versus the exact decoded signs.
            if not math.isfinite(roundtrip_error) or roundtrip_error > 1e-5:
                raise RuntimeError(f"WeTok discrete latent roundtrip mismatch: {roundtrip_error}")
            return {
                "indices": indices,
                "index_grid_shape": tuple(int(v) for v in expected_grid),
                "latent_shape": tuple(int(v) for v in h.shape),
                "official_quantized_vs_index_roundtrip_max_abs": roundtrip_error,
            }

    @torch.inference_mode()
    def decode(self, tokens: dict[str, Any], prefix_scales: int | None = None):
        if prefix_scales is not None:
            raise ValueError("WeTok is a full-codec reference; prefix decoding is not supported")
        grid = tuple(tokens["index_grid_shape"])
        indices = tokens["indices"]
        if grid[1:] != (16, 16, 4) or indices.numel() != math.prod(grid):
            raise ValueError("Invalid WeTok index layout")
        with torch.autocast(device_type=self.device.type, enabled=False):
            quantized = self.quantize.decode(indices.reshape(grid)).permute(0, 3, 1, 2).contiguous()
            reconstructed = self.codec.decoder(quantized)
            return reconstructed.clamp(-1, 1).add(1).mul(0.5)

    def token_metadata(self, tokens: dict[str, Any], prefix_scales: int | None = None):
        if prefix_scales is not None:
            raise ValueError("WeTok prefix budgets are outside this evaluation")
        indices = tokens["indices"]
        grid = tuple(tokens["index_grid_shape"])
        batch_size = grid[0]
        per_image_indices = indices.numel() // batch_size
        raw_bits = per_image_indices * self.bits_per_index
        if raw_bits != 8192:
            raise RuntimeError(f"Actual WeTok raw source budget is {raw_bits}, expected 8192")
        return {
            "raw_bits": raw_bits,
            "index_count": per_image_indices,
            "index_count_total": indices.numel(),
            "shapes": [list(indices.shape)],
            "official_flat_index_shape": list(indices.shape),
            "logical_index_shape": list(grid),
            "latent_shape": list(tokens["latent_shape"]),
            "spatial_positions_per_image": grid[1] * grid[2],
            "num_codebooks": self.num_codebooks,
            "codebook_size_per_group": self.codebook_size,
            "bits_per_index": self.bits_per_index,
            "index_dtype": str(indices.dtype),
            "index_min": int(indices.min().item()),
            "index_max": int(indices.max().item()),
            "official_quantized_vs_index_roundtrip_max_abs": tokens[
                "official_quantized_vs_index_roundtrip_max_abs"],
            "rate_interpretation": "fixed-length noiseless source bits; not channel uses",
        }


def build_adapter(config: dict[str, Any], device: str | torch.device):
    return WeTokAdapter(config, device)
