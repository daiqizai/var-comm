from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch

from .strong_jscc import StrongJSCC, StrongJSCCObservation


def uniform_complex_pair_indices(
    native_real_symbols: int, active_real_symbols: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if native_real_symbols % 2 or active_real_symbols % 2:
        raise ValueError("native and active real-symbol counts must be even")
    native_pairs = native_real_symbols // 2
    active_pairs = active_real_symbols // 2
    if not 0 < active_pairs <= native_pairs:
        raise ValueError("active complex-use count is outside the native range")
    pair_indices = torch.div(
        torch.arange(active_pairs, dtype=torch.int64) * native_pairs,
        active_pairs,
        rounding_mode="floor",
    )
    if pair_indices.unique().numel() != active_pairs:
        raise RuntimeError("uniform complex-pair mask contains duplicates")
    real_indices = torch.stack((2 * pair_indices, 2 * pair_indices + 1), dim=1).reshape(-1)
    return pair_indices, real_indices


def index_sha256(indices: torch.Tensor) -> str:
    return hashlib.sha256(indices.cpu().contiguous().numpy().tobytes()).hexdigest()


class ExactBudgetStrongJSCC(StrongJSCC):
    """StrongJSCC with an exact, fixed, I/Q-paired active-symbol budget."""

    def __init__(
        self,
        *,
        active_real_symbols: int,
        image_size: int = 256,
        latent_channels: int = 24,
        stage_channels: tuple[int, int, int, int] = (64, 128, 256, 384),
        stage_blocks: tuple[int, int, int, int] = (1, 1, 2, 4),
        condition_dim: int = 128,
    ) -> None:
        super().__init__(
            image_size=image_size,
            latent_channels=latent_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            condition_dim=condition_dim,
        )
        self.native_real_symbols = int(self.real_symbols)
        pair_indices, real_indices = uniform_complex_pair_indices(
            self.native_real_symbols, int(active_real_symbols)
        )
        self.register_buffer("active_complex_pair_indices", pair_indices, persistent=True)
        self.register_buffer("active_real_indices", real_indices, persistent=True)
        self.real_symbols = int(active_real_symbols)

    @property
    def mask_sha256(self) -> str:
        return index_sha256(self.active_complex_pair_indices)

    def normalize_channel_input(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = latent.flatten(start_dim=1)
        if flat.shape[1] != self.native_real_symbols:
            raise ValueError("latent tensor does not match the native real-symbol count")
        indices = self.active_real_indices.to(flat.device)
        active = flat.index_select(1, indices)
        power = active.float().square().mean(dim=1, keepdim=True).clamp_min(1e-12)
        active = active / torch.sqrt(power).to(active.dtype)
        transmitted = torch.zeros_like(flat)
        transmitted.index_copy_(1, indices, active)
        return transmitted.reshape_as(latent), power.reshape(-1)

    def transmit(
        self,
        transmitted: torch.Tensor,
        snr_db: torch.Tensor | float,
        standard_normal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat = transmitted.flatten(start_dim=1)
        if flat.shape[1] != self.native_real_symbols:
            raise ValueError("transmitted tensor does not match the native real-symbol count")
        indices = self.active_real_indices.to(flat.device)
        active = flat.index_select(1, indices)
        if standard_normal is None:
            noise = torch.randn_like(active)
        else:
            supplied = standard_normal.to(device=flat.device, dtype=flat.dtype)
            if supplied.shape == transmitted.shape:
                noise = supplied.flatten(start_dim=1).index_select(1, indices)
            elif supplied.numel() == active.numel():
                noise = supplied.reshape_as(active)
            else:
                raise ValueError("standard-normal tensor does not match active symbols")
        snr = self._snr_tensor(snr_db, transmitted.shape[0], transmitted.device)
        power = active.float().square().mean(dim=1)
        gamma = torch.pow(10.0, snr / 10.0)
        sigma = torch.sqrt(power / (2.0 * gamma)).to(active.dtype)
        received_active = active + noise * sigma[:, None]
        received = torch.zeros_like(flat)
        received.index_copy_(1, indices, received_active)
        return received.reshape_as(transmitted)

    def forward_with_observation(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
        standard_normal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, StrongJSCCObservation]:
        latent = self.encode(image, snr_db)
        transmitted, _unnormalized_power = self.normalize_channel_input(latent)
        received = self.transmit(transmitted, snr_db, standard_normal)
        reconstruction = self.decode(received, snr_db)
        flat = transmitted.flatten(start_dim=1)
        indices = self.active_real_indices.to(flat.device)
        normalized_power = flat.index_select(1, indices).float().square().mean(dim=1)
        snr = self._snr_tensor(snr_db, image.shape[0], image.device)
        observation = StrongJSCCObservation(
            transmitted=transmitted,
            received=received,
            normalized_power=normalized_power,
            snr_db=snr,
        )
        return reconstruction, observation


def build_exact_budget_model(checkpoint: Mapping[str, Any]) -> ExactBudgetStrongJSCC:
    config = checkpoint["config"]
    model_config = config["model"]
    model = ExactBudgetStrongJSCC(
        image_size=int(config["image_size"]),
        latent_channels=int(model_config["latent_channels"]),
        active_real_symbols=int(model_config["active_real_symbols"]),
        stage_channels=tuple(int(value) for value in model_config["stage_channels"]),
        stage_blocks=tuple(int(value) for value in model_config["stage_blocks"]),
        condition_dim=int(model_config["condition_dim"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
