"""Receiver-local source-hypothesis features; the transmitter is frozen and shared in value."""

import copy

import torch
from torch import nn
from torch.nn import functional

from wetok_comm.native import hard_native_st


VARIANTS = ('single_pass', 'multiscale_no_history', 'multiscale_state_history',
            'multiscale_prediction_features', 'multiscale_innovation', 'full_grid_innovation')


class InnovationSystem(nn.Module):
    def __init__(self, parent, variant, fusion_config):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('unknown receiver innovation arm')
        self.variant = variant
        self.encoder = copy.deepcopy(parent.encoder).eval().requires_grad_(False)
        self.receiver = copy.deepcopy(parent.receiver).requires_grad_(True)
        self.interface = 'continuous_mean'
        self.has_feature = variant in ('multiscale_prediction_features', 'multiscale_innovation', 'full_grid_innovation')
        self.ratio_maximum = fusion_config['ratio_maximum']
        if self.has_feature:
            width = self.receiver.output.in_features
            self.fusion_projection = nn.Linear(32, width)
            nn.init.zeros_(self.fusion_projection.weight)
            nn.init.zeros_(self.fusion_projection.bias)
            self.feature_gate = nn.Sequential(nn.Linear(3, fusion_config['gate_hidden']), nn.SiLU(),
                                              nn.Linear(fusion_config['gate_hidden'], 1))
            nn.init.zeros_(self.feature_gate[-1].weight)
            nn.init.constant_(self.feature_gate[-1].bias, fusion_config['gate_initial_bias'])

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def transmit(self, source_fq, snrs):
        if tuple(source_fq.shape[1:]) != (32, 16, 16) or snrs.shape != (len(source_fq),):
            raise ValueError('TX only accepts the fixed source Fq and nominal SNR')
        return self.encoder(source_fq, snrs)

    def _backproject(self, values):
        values = values.reshape(len(values), self.receiver.positions, self.receiver.features)
        decoded = self.receiver.channel_inverse(torch.einsum('nm,bmc->bnc', self.receiver.spatial_inverse, values))
        return decoded.transpose(1, 2).reshape(len(values), 32, 16, 16)

    def _hypothesis(self, initial_mean, previous):
        if previous.shape[-2:] == (16, 16):
            return previous
        correction = previous - functional.adaptive_avg_pool2d(initial_mean, previous.shape[-2:])
        return (initial_mean + functional.interpolate(correction, (16, 16), mode='bilinear', align_corners=False)).clamp(-1, 1)

    def _read_feature(self, guessed, memory, snrs, size, previous, feature, gate):
        receiver = self.receiver
        base = functional.adaptive_avg_pool2d(guessed, size)
        context = functional.interpolate(previous, (size, size), mode='bilinear', align_corners=False)
        positions = functional.adaptive_avg_pool2d(receiver.query_position, size).flatten(2).transpose(1, 2)
        scale = guessed.new_tensor([[size / 16, 16 / size]]).expand(len(guessed), -1)
        query = receiver.query_input(base.flatten(2).transpose(1, 2)) + positions
        query = query + receiver.context_input(context.flatten(2).transpose(1, 2)) + receiver.scale_condition(scale)[:, None]
        projected = self.fusion_projection(functional.adaptive_avg_pool2d(self._backproject(feature), size).flatten(2).transpose(1, 2))
        query = receiver.query_snr(query + gate[:, None] * projected, snrs)
        for block in receiver.blocks:
            query = block(query, memory)
        return base + receiver.output(query).transpose(1, 2).reshape(len(guessed), 32, size, size)

    def receive(self, received, snrs):
        if tuple(received.shape[1:]) != (3060, 2) or snrs.shape != (len(received),):
            raise ValueError('RX may only receive the paid waveform and nominal SNR')
        guessed, memory = self.receiver.prepare(received, snrs)
        predictions, hypotheses, gates, ratios = [], [], [], []
        if self.variant == 'single_pass':
            logits = self.receiver.read(guessed, memory, snrs, 16)
            mean = torch.tanh(logits / 2)
            states = [functional.adaptive_avg_pool2d(mean, size) for size in (4, 8)]
        else:
            sizes = (16, 16, 16) if self.variant == 'full_grid_innovation' else (4, 8, 16)
            initial_mean = torch.tanh(guessed / 2)
            previous, states = None, []
            for stage, size in enumerate(sizes):
                context = None if self.variant == 'multiscale_no_history' else previous
                if self.has_feature and previous is not None:
                    hypothesis = self._hypothesis(initial_mean, previous)
                    predicted = self.encoder(hypothesis, snrs)
                    innovation = received - predicted
                    expected_noise = torch.pow(10., -snrs / 10)
                    ratio = innovation.detach().square().mean((1, 2)) / expected_noise
                    gate_inputs = torch.stack((snrs / 20, torch.full_like(snrs, stage / 2), ratio.clamp_max(self.ratio_maximum).log1p()), dim=1)
                    gate = self.feature_gate(gate_inputs).sigmoid()
                    feature = predicted if self.variant == 'multiscale_prediction_features' else innovation
                    logits = self._read_feature(guessed, memory, snrs, size, previous, feature, gate)
                    predictions.append(predicted)
                    hypotheses.append(hypothesis)
                    gates.append(gate)
                    ratios.append(ratio)
                else:
                    logits = self.receiver.read(guessed, memory, snrs, size, context)
                mean = torch.tanh(logits / 2)
                if stage < 2:
                    target_size = (4, 8)[stage]
                    states.append(functional.adaptive_avg_pool2d(mean, target_size))
                previous = mean
        return {'logits': logits, 'native_fq': hard_native_st(logits), 'receiver_features': mean,
                'states': states, 'predicted_symbols': predictions, 'source_hypotheses': hypotheses,
                'feature_gates': gates, 'residual_noise_ratios': ratios}
