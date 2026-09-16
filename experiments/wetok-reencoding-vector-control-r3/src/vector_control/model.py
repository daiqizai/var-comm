"""Change only the fused vector; the scalar residual gate and full-grid receiver work remain."""

import torch
from torch.nn import functional

from grid_controls.model import GridJointSystem
from wetok_comm.native import hard_native_st


VARIANT = 'full_grid_prediction_features'


class PredictionVectorSystem(GridJointSystem):
    def __init__(self, parent, fusion_config):
        super().__init__(parent, 'full_grid_innovation', fusion_config, update_sender=True)
        self.variant = VARIANT

    def receive(self, received, snrs):
        if tuple(received.shape[1:]) != (3060, 2) or snrs.shape != (len(received),):
            raise ValueError('RX only accepts the paid waveform and nominal SNR')
        guessed, memory = self.receiver.prepare(received, snrs)
        predictions, hypotheses, gates, ratios = [], [], [], []
        initial_mean = torch.tanh(guessed / 2)
        previous, states = None, []
        for stage, size in enumerate((16, 16, 16)):
            if previous is not None:
                hypothesis = self._hypothesis(initial_mean, previous)
                predicted = self.encoder(hypothesis, snrs)
                innovation = received - predicted
                expected_noise = torch.pow(10., -snrs / 10)
                ratio = innovation.detach().square().mean((1, 2)) / expected_noise
                gate_inputs = torch.stack((snrs / 20, torch.full_like(snrs, stage / 2), ratio.clamp_max(self.ratio_maximum).log1p()), dim=1)
                gate = self.feature_gate(gate_inputs).sigmoid()
                logits = self._read_feature(guessed, memory, snrs, size, previous, predicted, gate)
                predictions.append(predicted)
                hypotheses.append(hypothesis)
                gates.append(gate)
                ratios.append(ratio)
            else:
                logits = self.receiver.read(guessed, memory, snrs, size, previous)
            mean = torch.tanh(logits / 2)
            if stage < 2:
                states.append(functional.adaptive_avg_pool2d(mean, (4, 8)[stage]))
            previous = mean
        return {'logits': logits, 'native_fq': hard_native_st(logits), 'receiver_features': mean, 'states': states,
            'predicted_symbols': predictions, 'source_hypotheses': hypotheses, 'feature_gates': gates, 'residual_noise_ratios': ratios}
