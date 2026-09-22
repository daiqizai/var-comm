from __future__ import annotations

import torch
from torch import nn

from latent_enhancement.latent import normalize_enhancement


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1),
                                    nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, hidden):
        return hidden + self.layers(hidden)


def coordinate_channels(latent_shape, uses):
    _, height, width = latent_shape
    if uses not in (512, 1024) or 2 * uses % (height * width):
        raise ValueError("additional budget must map to an exact number of measured spatial coordinates")
    return 2 * uses // (height * width)


class EnhancementEncoder(nn.Module):
    def __init__(self, latent_shape, uses, scale, width=64, blocks=3):
        super().__init__()
        channels, self.height, self.width = latent_shape
        self.uses = uses
        self.register_buffer("scale", scale.reshape(1, channels, 1, 1).clone())
        self.stem = nn.Conv2d(2 * channels, width, 3, padding=1)
        self.trunk = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.projection = nn.Conv2d(width, coordinate_channels(latent_shape, uses), 3, padding=1)

    def forward(self, residual, base_tx):
        if residual.shape != base_tx.shape or tuple(base_tx.shape[-2:]) != (self.height, self.width):
            raise ValueError("TX latent shape mismatch")
        hidden = self.stem(torch.cat((residual / self.scale, base_tx / self.scale), dim=1))
        coordinates = self.projection(self.trunk(hidden)).reshape(len(base_tx), self.uses, 2)
        return normalize_enhancement(coordinates)


class ConditionalReceiver(nn.Module):
    def __init__(self, latent_shape, uses, scale, width=64, blocks=3):
        super().__init__()
        channels, self.height, self.width = latent_shape
        self.uses = uses
        self.register_buffer("scale", scale.reshape(1, channels, 1, 1).clone())
        self.base_stem = nn.Conv2d(channels, width, 3, padding=1)
        self.observation_stem = (nn.Conv2d(coordinate_channels(latent_shape, uses), width, 3, padding=1) if uses else None)
        self.condition = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.trunk = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.correction = nn.Conv2d(width, channels, 3, padding=1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, observation, base_rx, snr_db, rx_status):
        if rx_status.shape != (len(base_rx), 3) or snr_db.shape != (len(base_rx),):
            raise ValueError("RX condition shape mismatch")
        conditions = torch.stack((snr_db / 20, rx_status[:, 0], rx_status[:, 1], rx_status[:, 2] / 10), dim=1)
        hidden = self.base_stem(base_rx / self.scale) + self.condition(conditions)[:, :, None, None]
        if self.uses:
            if observation is None or observation.shape != (len(base_rx), self.uses, 2):
                raise ValueError("RX observation must use exactly its registered additional budget")
            hidden = hidden + self.observation_stem(observation.reshape(len(base_rx), -1, self.height, self.width))
        elif observation is not None:
            raise ValueError("receiver-only control must not see additional observations")
        return base_rx + self.correction(self.trunk(hidden)) * self.scale


class CommunicationArm(nn.Module):
    def __init__(self, latent_shape, uses, scale, width=64, blocks=3):
        super().__init__()
        self.uses = uses
        self.encoder = EnhancementEncoder(latent_shape, uses, scale, width, blocks) if uses else None
        self.receiver = ConditionalReceiver(latent_shape, uses, scale, width, blocks)

    def receive_training_sample(self, continuous, base_tx, base_rx, snr_db, rx_status, standard_noise):
        waveform = None
        observation = None
        if self.uses:
            waveform = self.encoder(continuous - base_tx, base_tx)
            noise_scale = torch.pow(10.0, -snr_db / 20).reshape(-1, 1, 1)
            observation = waveform + standard_noise[:, :self.uses] * noise_scale
        reconstructed = self.receiver(observation, base_rx, snr_db, rx_status)
        return reconstructed, waveform

    def receive_training_residual_sample(self, residual, base_tx, base_rx, snr_db, rx_status, standard_noise):
        """Use an explicit branch residual with a shared TX condition."""
        waveform = None
        observation = None
        if self.uses:
            waveform = self.encoder(residual, base_tx)
            noise_scale = torch.pow(10.0, -snr_db / 20).reshape(-1, 1, 1)
            observation = waveform + standard_noise[:, :self.uses] * noise_scale
        reconstructed = self.receiver(observation, base_rx, snr_db, rx_status)
        return reconstructed, waveform


def build_arms(latent_shape, scale, recipe):
    arms = nn.ModuleDict()
    for name, uses in (("enhancement512", 512), ("enhancement1024", 1024), ("receiver_only_refiner", 0)):
        torch.manual_seed(recipe["initialization_seed"] + uses)
        arms[name] = CommunicationArm(latent_shape, uses, scale, recipe["model_hidden_channels"], recipe["model_residual_blocks"])
    reference = arms["enhancement512"].state_dict()
    for name in ("enhancement1024", "receiver_only_refiner"):
        state = arms[name].state_dict()
        for key in state:
            if key in reference and state[key].shape == reference[key].shape:
                state[key] = reference[key].clone()
        arms[name].load_state_dict(state, strict=True)
    return arms


def render_received(decoder, latent, rx_status):
    reconstructed = decoder(latent)
    accepts_header = (rx_status[:, 0] > 0.5).reshape(-1, 1, 1, 1)
    return torch.where(accepts_header, reconstructed, torch.full_like(reconstructed, 0.5))


def latent_errors(latent, truth, scale, rx_status):
    errors = ((latent - truth) / scale.reshape(1, -1, 1, 1)).square().flatten(1).mean(1)
    return errors * (rx_status[:, 0] > 0.5).to(errors.dtype)
