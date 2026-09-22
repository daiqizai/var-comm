"""Shared contracts for frozen-VAR decoder-only fine-tuning."""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CONDITIONS = ("full", "m8", "m9")
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
IMAGE_SIZE = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config root must be a mapping")
    if payload.get("status") != "preregistered_before_any_formal_cache_training_or_test_output":
        raise RuntimeError("decoder-only preregistration status changed")
    if bool(payload.get("official_imagenette_validation_accessed")):
        raise RuntimeError("official Imagenette validation must remain sealed")
    if tuple(payload["latent_contract"]["conditions"]) != CONDITIONS:
        raise RuntimeError("full/m8/m9 condition contract changed")
    if tuple(int(value) for value in payload["latent_contract"]["patch_nums"]) != PATCH_NUMS:
        raise RuntimeError("VAR patch-number contract changed")
    return payload


def require_sha(path: Path, expected: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = sha256_file(resolved)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {resolved}: {actual} != {expected}")
    return resolved


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    scale = IMAGE_SIZE / min(width, height)
    resized = (round(width * scale), round(height * scale))
    image = TF.resize(
        image,
        [resized[1], resized[0]],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    image = TF.center_crop(image, [IMAGE_SIZE, IMAGE_SIZE])
    return TF.pil_to_tensor(image).float().div_(127.5).sub_(1.0)


def preprocess_bytes(payload: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(payload)) as image:
        return preprocess_pil(image)


def preprocess_path(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess_pil(image)


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping) and candidate:
                return candidate
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise TypeError(f"unsupported checkpoint payload: {path}")


def add_var_source(var_source: Path) -> None:
    source_text = str(var_source.expanduser().resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def build_vae_only(var_source: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    add_var_source(var_source)
    from models.vqvae import VQVAE  # type: ignore[import-not-found]

    vae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        share_quant_resi=4,
        v_patch_nums=PATCH_NUMS,
        test_mode=True,
    )
    incompatible = vae.load_state_dict(_load_state_dict(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"VQ-VAE checkpoint mismatch: {incompatible}")
    return vae.to(device).eval().requires_grad_(False)


def build_vae_var(
    var_source: Path,
    vae_checkpoint: Path,
    var_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    add_var_source(var_source)
    import dist  # type: ignore[import-not-found]
    from models import build_vae_var as upstream_build  # type: ignore[import-not-found]

    if device.type == "cpu":
        setattr(dist, "__device", "cpu")
    vae, var = upstream_build(
        device=device,
        patch_nums=PATCH_NUMS,
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        num_classes=1000,
        depth=16,
        attn_l2_norm=True,
        flash_if_available=False,
        fused_if_available=False,
    )
    vae_incompatible = vae.load_state_dict(_load_state_dict(vae_checkpoint), strict=True)
    var_incompatible = var.load_state_dict(_load_state_dict(var_checkpoint), strict=True)
    if vae_incompatible.missing_keys or vae_incompatible.unexpected_keys:
        raise RuntimeError(f"VQ-VAE checkpoint mismatch: {vae_incompatible}")
    if var_incompatible.missing_keys or var_incompatible.unexpected_keys:
        raise RuntimeError(f"VAR checkpoint mismatch: {var_incompatible}")
    var.cond_drop_rate = 0.0
    return vae.eval().requires_grad_(False), var.eval().requires_grad_(False)


def source_tokens_to_fhat(
    vae: torch.nn.Module, token_scales: Sequence[torch.Tensor]
) -> torch.Tensor:
    batch_size = token_scales[0].shape[0]
    embedded_scales = []
    for tokens, patch_num in zip(token_scales, PATCH_NUMS):
        embedded = vae.quantize.embedding(tokens)
        embedded_scales.append(
            embedded.transpose(1, 2).reshape(batch_size, vae.Cvae, patch_num, patch_num)
        )
    return vae.quantize.embed_to_fhat(
        embedded_scales, all_to_max_scale=True, last_one=True
    )


def source_prefix_fhats(
    vae: torch.nn.Module,
    token_scales: Sequence[torch.Tensor],
    prefixes: Sequence[int] = (8, 9),
) -> torch.Tensor:
    """Accumulate only transmitted source-token scales, with no VAR completion.

    The returned tensor has shape ``[B, len(prefixes), C, 16, 16]``. Missing
    residual-scale contributions remain zero, so this is the natural paired
    control for measuring the receiver-side value added by VAR completion.
    """

    if len(token_scales) != len(PATCH_NUMS):
        raise ValueError("token scales do not match the VAR patch schedule")
    requested = tuple(int(value) for value in prefixes)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("prefixes must be non-empty and unique")
    if any(value < 1 or value > len(PATCH_NUMS) for value in requested):
        raise ValueError("prefix scale count is outside the VAR schedule")
    batch_size = token_scales[0].shape[0]
    for tokens, patch_num in zip(token_scales, PATCH_NUMS):
        if tokens.shape[0] != batch_size or tokens.shape[1] != patch_num * patch_num:
            raise ValueError("source-token scale shape changed")

    first_embedding = vae.quantize.embedding(token_scales[0])
    f_hat = first_embedding.new_zeros(
        batch_size, vae.Cvae, PATCH_NUMS[-1], PATCH_NUMS[-1]
    )
    captured: dict[int, torch.Tensor] = {}
    for scale_number, (tokens, patch_num) in enumerate(
        zip(token_scales, PATCH_NUMS), start=1
    ):
        embedded = vae.quantize.embedding(tokens)
        embedded = embedded.transpose(1, 2).reshape(
            batch_size, vae.Cvae, patch_num, patch_num
        )
        f_hat, _next = vae.quantize.get_next_autoregressive_input(
            scale_number - 1, len(PATCH_NUMS), f_hat, embedded
        )
        if scale_number in requested:
            captured[scale_number] = f_hat.clone()
        if len(captured) == len(requested):
            break
    return torch.stack([captured[value] for value in requested], dim=1)


def run_m8_m9_closed_loop_fhat(
    var: torch.nn.Module,
    vae: torch.nn.Module,
    source_scales: Sequence[torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return ``[B, 2, C, 16, 16]`` f_hat for m=8 and m=9."""

    anchors = (8, 9)
    variants = len(anchors)
    batch_size = labels.shape[0]
    expanded_labels = labels.repeat_interleave(variants)
    thresholds = labels.new_tensor(anchors).repeat(batch_size)
    expanded_source = [tokens.repeat_interleave(variants, dim=0) for tokens in source_scales]

    condition = var.class_emb(expanded_labels)
    level_positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_token_map = (
        condition.unsqueeze(1).expand(-1, var.first_l, -1)
        + var.pos_start.expand(condition.shape[0], var.first_l, -1)
        + level_positions[:, : var.first_l]
    )
    f_hat = condition.new_zeros(
        condition.shape[0], var.Cvae, PATCH_NUMS[-1], PATCH_NUMS[-1]
    )
    current_length = 0
    for block in var.blocks:
        block.attn.kv_caching(True)
    try:
        for scale_number, patch_num in enumerate(PATCH_NUMS, start=1):
            current_length += patch_num * patch_num
            conditioned = var.shared_ada_lin(condition)
            hidden = next_token_map
            for block in var.blocks:
                hidden = block(x=hidden, cond_BD=conditioned, attn_bias=None)
            source = expanded_source[scale_number - 1]
            if scale_number <= anchors[0]:
                chosen = source
            else:
                predicted = var.get_logits(hidden, condition).float().argmax(dim=-1)
                chosen = torch.where((scale_number <= thresholds)[:, None], source, predicted)
            embedding = vae.quantize.embedding(chosen)
            embedding = embedding.transpose(1, 2).reshape(
                condition.shape[0], var.Cvae, patch_num, patch_num
            )
            f_hat, next_token_map = vae.quantize.get_next_autoregressive_input(
                scale_number - 1, len(PATCH_NUMS), f_hat, embedding
            )
            if scale_number != len(PATCH_NUMS):
                next_token_map = next_token_map.reshape(
                    condition.shape[0], var.Cvae, -1
                ).transpose(1, 2)
                next_length = PATCH_NUMS[scale_number] ** 2
                next_token_map = (
                    var.word_embed(next_token_map)
                    + level_positions[:, current_length : current_length + next_length]
                )
    finally:
        for block in var.blocks:
            block.attn.kv_caching(False)
    return f_hat.reshape(
        batch_size, variants, var.Cvae, PATCH_NUMS[-1], PATCH_NUMS[-1]
    )


def generate_condition_fhats(
    vae: torch.nn.Module,
    var: torch.nn.Module,
    images_m11: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    continuous = vae.quant_conv(vae.encoder(images_m11))
    token_scales = vae.quantize.f_to_idxBl_or_fhat(
        continuous, to_fhat=False, v_patch_nums=PATCH_NUMS
    )
    full = source_tokens_to_fhat(vae, token_scales)
    m8_m9 = run_m8_m9_closed_loop_fhat(var, vae, token_scales, labels)
    return torch.cat((full[:, None], m8_m9), dim=1)


def dino_preprocess(images_01: torch.Tensor) -> torch.Tensor:
    images = F.interpolate(
        images_01.float(), size=(224, 224), mode="bicubic", align_corners=False
    )
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (images - mean) / std


def dino_cls(model: torch.nn.Module, images_01: torch.Tensor) -> torch.Tensor:
    result = model.forward_features(dino_preprocess(images_01))
    if not isinstance(result, Mapping) or "x_norm_clstoken" not in result:
        raise RuntimeError("DINOv2 did not expose x_norm_clstoken")
    return result["x_norm_clstoken"].float()


def load_dino(source: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not (source / "hubconf.py").is_file():
        raise FileNotFoundError(source / "hubconf.py")
    model = torch.hub.load(str(source), "dinov2_vits14", source="local", pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


def state_sha256(module: torch.nn.Module, *, exclude_prefix: str | None = None) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if exclude_prefix is not None and name.startswith(exclude_prefix):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def decoder_parameter_counts(vae: torch.nn.Module) -> tuple[int, int]:
    return (
        sum(parameter.numel() for parameter in vae.decoder.parameters()),
        sum(parameter.numel() for parameter in vae.post_quant_conv.parameters()),
    )


def freeze_decoder_only(vae: torch.nn.Module) -> None:
    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    vae.decoder.train()
    vae.post_quant_conv.eval()


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(array.mean()), float(lower), float(upper)


def paired_win_rates(
    tuned: Sequence[float],
    official: Sequence[float],
    *,
    higher_is_better: bool,
    tolerance: float = 1e-12,
) -> dict[str, float | int]:
    tuned_array = np.asarray(tuned, dtype=np.float64)
    official_array = np.asarray(official, dtype=np.float64)
    if tuned_array.shape != official_array.shape or tuned_array.ndim != 1:
        raise ValueError("paired vectors must have the same one-dimensional shape")
    signed = tuned_array - official_array
    beneficial = signed if higher_is_better else -signed
    wins = int(np.sum(beneficial > tolerance))
    losses = int(np.sum(beneficial < -tolerance))
    ties = int(beneficial.size - wins - losses)
    return {
        "n": int(beneficial.size),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / beneficial.size,
        "tie_rate": ties / beneficial.size,
        "loss_rate": losses / beneficial.size,
    }


def select_checkpoint(
    rows: Sequence[Mapping[str, Any]], eligibility: Mapping[str, float]
) -> dict[str, Any]:
    by_epoch: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_epoch.setdefault(int(row["epoch"]), {})[str(row["condition"])] = row
    if 0 not in by_epoch or set(by_epoch[0]) != set(CONDITIONS):
        raise RuntimeError("epoch-0 official calibration reference is incomplete")
    reference = by_epoch[0]
    candidates = []
    for epoch in sorted(value for value in by_epoch if value > 0):
        current = by_epoch[epoch]
        if set(current) != set(CONDITIONS):
            raise RuntimeError(f"epoch {epoch} calibration rows are incomplete")
        psnr_delta = float(np.mean([
            float(current[name]["psnr_db"]) - float(reference[name]["psnr_db"])
            for name in CONDITIONS
        ]))
        lpips_deltas = {
            name: float(current[name]["lpips_alex"]) - float(reference[name]["lpips_alex"])
            for name in CONDITIONS
        }
        dino_deltas = {
            name: float(current[name]["dino_cosine"]) - float(reference[name]["dino_cosine"])
            for name in CONDITIONS
        }
        full_psnr_delta = float(current["full"]["psnr_db"]) - float(
            reference["full"]["psnr_db"]
        )
        eligible = (
            psnr_delta > float(eligibility["require_aggregate_psnr_delta_db_gt"])
            and max(lpips_deltas.values())
            <= float(eligibility["require_each_condition_lpips_delta_lte"])
            and min(dino_deltas.values())
            >= float(eligibility["require_each_condition_dino_delta_gte"])
            and full_psnr_delta
            >= float(eligibility["require_full_psnr_delta_db_gte"])
        )
        candidates.append(
            {
                "epoch": epoch,
                "aggregate_psnr_delta_db": psnr_delta,
                "full_psnr_delta_db": full_psnr_delta,
                "lpips_delta_by_condition": lpips_deltas,
                "dino_delta_by_condition": dino_deltas,
                "eligible": eligible,
            }
        )
    if not candidates:
        raise RuntimeError("no trained checkpoint candidates")
    eligible_candidates = [item for item in candidates if item["eligible"]]
    selected = max(
        eligible_candidates or candidates,
        key=lambda item: (float(item["aggregate_psnr_delta_db"]), -int(item["epoch"])),
    )
    return {
        "selection_gate_passed": bool(eligible_candidates),
        "selected_epoch": int(selected["epoch"]),
        "selected": selected,
        "candidates": candidates,
    }


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {name}: {result}")
    return result
