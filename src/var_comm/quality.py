"""Frozen local LPIPS/DINO image metrics, with no pretrained-weight downloads."""

from __future__ import annotations

from pathlib import Path

import lpips
import numpy as np
import torch
from torch.nn import functional as functional


def load_quality_models(paths, device):
    linear_weights = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
    if not linear_weights.is_file():
        raise FileNotFoundError(linear_weights)
    perceptual = lpips.LPIPS(net="alex", pnet_rand=True, model_path=str(linear_weights), verbose=False)
    alex_weights = torch.load(paths["alexnet_checkpoint"], map_location="cpu", weights_only=True)
    translated = {name: alex_weights["features." + name.split(".", 1)[1]] for name in perceptual.net.state_dict()}
    perceptual.net.load_state_dict(translated, strict=True)
    del alex_weights, translated
    dino = torch.hub.load(paths["dino_source"], "dinov2_vits14", source="local", pretrained=False)
    dino.load_state_dict(torch.load(paths["dino_checkpoint"], map_location="cpu", weights_only=True), strict=True)
    return perceptual.to(device).eval().requires_grad_(False), dino.to(device).eval().requires_grad_(False), linear_weights


@torch.no_grad()
def dino_features(model, images):
    resized = functional.interpolate(images.float(), size=(224, 224), mode="bicubic", align_corners=False)
    mean = resized.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = resized.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return model.forward_features((resized - mean) / std)["x_norm_clstoken"].float()


@torch.no_grad()
def quality_metrics(source, images, perceptual, dino, device):
    original = torch.as_tensor(np.asarray(source), device=device, dtype=torch.float32)[None]
    source_features = dino_features(dino, original)
    rows, embeddings = [], []
    for start in range(0, len(images), 8):
        reconstructed = torch.as_tensor(np.stack(images[start:start + 8]), device=device, dtype=torch.float32)
        reference = original.expand(len(reconstructed), -1, -1, -1)
        psnr = -10 * torch.log10((reconstructed - reference).square().flatten(1).mean(1))
        perceptual_values = perceptual(reconstructed * 2 - 1, reference * 2 - 1).reshape(-1)
        features = dino_features(dino, reconstructed)
        similarity = functional.cosine_similarity(features, source_features.expand_as(features), dim=1)
        for index in range(len(reconstructed)):
            rows.append({"psnr_db": float(psnr[index]), "lpips_alex": float(perceptual_values[index]),
                         "dino_cosine": float(similarity[index])})
        embeddings.extend(features.cpu().numpy())
    return rows, source_features[0].cpu().numpy(), np.asarray(embeddings)
