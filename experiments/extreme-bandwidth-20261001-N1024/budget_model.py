"""Fresh budget model using the original P2048 construction and physical channel."""
import math
import torch
from torch import nn
from latent_research.models import PureContinuous
from latent_enhancement.latent import normalize_enhancement


class BudgetContinuous(PureContinuous):
    def __init__(self, scale, N, width=64, blocks=3):
        if N not in (512, 1024, 2048, 3060):
            raise ValueError('Explicit budget interface required')
        # Preserve P2048 initialization order: construct the whole original family,
        # then freshly initialize the two budget-specific convolutional heads.
        super().__init__(scale, width, blocks)
        self.uses = N
        self.channels = math.ceil(2*N/256)
        self.encode[-1] = nn.Conv2d(width, self.channels, 3, padding=1)
        self.rx_stem = nn.Conv2d(self.channels, width, 3, padding=1)

    def transmit(self, F):
        values = self.encode(F/self.scale).flatten(1)[:, :2*self.uses]
        return normalize_enhancement(values.reshape(-1, self.uses, 2))

    def receive(self, y, snr):
        if y.ndim != 3 or y.shape[1:] != (self.uses, 2) or not torch.isfinite(y).all():
            raise ValueError('Exact finite physical receive length required')
        if snr.shape != (len(y),) or not torch.isfinite(snr).all():
            raise ValueError('Nominal SNR vector required')
        padded = torch.nn.functional.pad(y.flatten(1), (0, self.channels*256-2*self.uses))
        padded = padded.reshape(-1, self.channels, 16, 16)
        return self.decode(self.rx_stem(padded)+self.snr_condition(snr[:, None]/20)[:, :, None, None])*self.scale

    def forward(self, F, snr, noise):
        if noise.shape != (len(F), self.uses, 2) or not torch.isfinite(noise).all():
            raise ValueError('One finite full-length physical noise realization required')
        return super().forward(F, snr, noise)
