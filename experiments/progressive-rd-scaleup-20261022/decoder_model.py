"""Capacity-matched endpoint conditioning; zero initialization preserves Dc."""
import torch
from torch import nn

class ProfileDecoder(nn.Module):
    def __init__(self, decoder, conditional):
        super().__init__()
        self.decoder = decoder
        self.conditional = bool(conditional)
        self.profile_affine = nn.Linear(8, 64)
        nn.init.zeros_(self.profile_affine.weight)
        nn.init.zeros_(self.profile_affine.bias)

    def forward(self, z, endpoint_index):
        if self.conditional:
            profile = torch.nn.functional.one_hot(endpoint_index, 8).to(z.dtype)
        else:
            # The constant control has exactly the same trainable architecture.
            profile = z.new_full((len(z), 8), 8 ** -0.5)
        gain, bias = self.profile_affine(profile).chunk(2, dim=1)
        return self.decoder(z * (1 + gain[:, :, None, None]) + bias[:, :, None, None])
