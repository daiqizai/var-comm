"""Same-information ordinary full-grid and existing innovation receivers with Joint E/R."""

import torch
from torch.nn import functional

from innovation_comm.model import InnovationSystem
from wetok_comm.native import hard_native_st


VARIANTS = ('full_grid_state_history', 'full_grid_innovation')


class GridJointSystem(InnovationSystem):
    def __init__(self, parent, variant, fusion_config, update_sender=True):
        if variant not in VARIANTS:
            raise ValueError('unknown registered Joint grid control')
        inherited_variant = 'multiscale_state_history' if variant == 'full_grid_state_history' else variant
        super().__init__(parent, inherited_variant, fusion_config)
        self.variant = variant
        self.update_sender = bool(update_sender)
        self.encoder.requires_grad_(self.update_sender)

    def receive(self, received, snrs):
        if self.variant == 'full_grid_innovation':
            return super().receive(received, snrs)
        if tuple(received.shape[1:]) != (3060, 2) or snrs.shape != (len(received),):
            raise ValueError('RX only accepts the paid full waveform and nominal SNR')
        guessed, memory = self.receiver.prepare(received, snrs)
        previous, states = None, []
        for stage in range(3):
            logits = self.receiver.read(guessed, memory, snrs, 16, previous)
            mean = torch.tanh(logits / 2)
            if stage < 2:
                states.append(functional.adaptive_avg_pool2d(mean, (4, 8)[stage]))
            previous = mean
        return {'logits': logits, 'native_fq': hard_native_st(logits), 'receiver_features': mean,
                'states': states, 'predicted_symbols': [], 'source_hypotheses': [],
                'feature_gates': [], 'residual_noise_ratios': []}
