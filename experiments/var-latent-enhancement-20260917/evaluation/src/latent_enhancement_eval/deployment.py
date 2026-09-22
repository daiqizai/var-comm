"""Thin deployment TX/channel/RX boundary shared by quality and timing runs."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import torch

from latent_enhancement.latent import enhancement_noise
from var_comm.progressive import receive_whole
from var_comm.study import seeded_noise


DEFAULT_LATENT_SHAPE = (1, 32, 16, 16)
BASE_USES = 3060


def tx_encode(base_waveform: np.ndarray, enhancement_waveform: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Register a base-only or base+enhancement transmission."""
    base = np.asarray(base_waveform)
    if base.ndim != 2 or base.shape[1] != 2 or (enhancement_waveform is not None and base.shape[0] != BASE_USES):
        raise ValueError("base waveform must be [uses,2], and continuous base must use 3060 uses")
    if not np.isfinite(base).all():
        raise FloatingPointError("nonfinite base waveform")
    result = {"base": base}
    if enhancement_waveform is not None:
        enhancement = np.asarray(enhancement_waveform)
        if enhancement.ndim != 2 or enhancement.shape[1] != 2 or enhancement.shape[0] not in (512, 1024):
            raise ValueError("enhancement waveform must be [512|1024,2]")
        if not np.isfinite(enhancement).all():
            raise FloatingPointError("nonfinite enhancement waveform")
        result["enhancement"] = enhancement
    result["signal"] = np.concatenate(tuple(result.values()))
    expected_uses = int(base.shape[0]) + (0 if enhancement_waveform is None else int(np.asarray(enhancement_waveform).shape[0]))
    if result["signal"].shape != (expected_uses, 2):
        raise RuntimeError("TX resource ledger changed")
    return result


def snr_scale(snr_db: float) -> float:
    value = float(snr_db)
    if not np.isfinite(value):
        raise ValueError("SNR must be finite")
    return float(np.sqrt(10.0 ** (value / 10.0)))


def channel_apply(encoded: Mapping[str, np.ndarray], image_id: str, noise_seed: int, snr_db: float,
                  *, base_received: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Apply one canonical noise realization to each transmitted segment."""
    base = np.asarray(encoded["base"])
    scale = snr_scale(snr_db)
    if base_received is None:
        base_received = base + seeded_noise(image_id, int(noise_seed), base.shape) / scale
    else:
        base_received = np.asarray(base_received)
        if base_received.shape != base.shape:
            raise ValueError("reused base observation shape does not match TX")
    output = {"base_received": base_received}
    if "enhancement" in encoded:
        enhancement = np.asarray(encoded["enhancement"])
        output["enhancement_received"] = enhancement + enhancement_noise(image_id, int(noise_seed), enhancement.shape[0]) / scale
        output["received"] = np.concatenate((base_received, output["enhancement_received"]))
    else:
        output["received"] = base_received
    if output["received"].shape != encoded["signal"].shape:
        raise RuntimeError("channel output shape does not match TX ledger")
    return output


def base_status(reception: Mapping[str, object]) -> np.ndarray:
    header = reception["header"]
    accepted = bool(header["accepted"])
    events = reception.get("events", ())
    body_ok = accepted and bool(events and events[0]["accepted"])
    return np.asarray([float(accepted), float(body_ok),
                       float(header["mode"]) if accepted else 0.0], dtype=np.float32)


def rx_base(observation: np.ndarray, snr_db: float, *, receive_fn: Callable = receive_whole,
            complete_fn: Callable | None = None, device: torch.device | str = "cpu",
            latent_shape: tuple[int, ...] = DEFAULT_LATENT_SHAPE) -> dict[str, object]:
    """Decode base m8 using an explicit model/protocol latent shape."""
    reception = receive_fn(np.asarray(observation), float(snr_db))
    status = base_status(reception)
    label = reception.get("label")
    if label is None:
        latent = torch.zeros(latent_shape, device=device, dtype=torch.float32)
    elif complete_fn is None:
        latent = None
    else:
        latent = complete_fn(reception["prefix"], label)
    return {"reception": reception, "latent": latent, "status": status,
            "header_ok": bool(status[0]), "body_ok": bool(status[1])}


def rx_continuous(observation: np.ndarray, base: Mapping[str, object], arm, snr_db: float,
                  *, device: torch.device | str, latent_shape: tuple[int, ...] = DEFAULT_LATENT_SHAPE) -> torch.Tensor:
    """Run the same enhancement receiver used by normal quality evaluation."""
    base_latent = base["latent"]
    if base_latent is None:
        base_latent = torch.zeros(latent_shape, device=device, dtype=torch.float32)
    status = torch.as_tensor(base["status"][None], device=device, dtype=torch.float32)
    observed = torch.as_tensor(np.asarray(observation)[None], device=device, dtype=torch.float32)
    snr = torch.tensor([float(snr_db)], device=device, dtype=torch.float32)
    return arm.receiver(observed, base_latent, snr, status)


def cpu_rgb_to_device(pixels_u8: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """The TX timing input endpoint: CPU uint8 RGB to model input tensor."""
    pixels = np.asarray(pixels_u8)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError("timing input must be CPU uint8 CHW RGB")
    return torch.from_numpy(pixels[None].astype(np.float32) / 127.5 - 1.0).to(device)


def device_rgb_to_cpu(image: torch.Tensor) -> np.ndarray:
    """The RX timing output endpoint: model image tensor to CPU RGB."""
    return image.detach().cpu().numpy()


def synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
