import inspect
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src')]

from innovation_comm.model import InnovationSystem, VARIANTS
from wetok_comm.geometry_candidate import GeometryJSCC
from wetok_comm.native import indices_to_features


class ReceiverInnovationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.manual_seed(127)
        self.parent = GeometryJSCC('single_pass', dict(width=24, heads=2, encoder_layers=1, receiver_layers=1,
            channel_positions=204, channel_features=30)).eval().requires_grad_(False)
        with torch.no_grad():
            self.parent.receiver.output.weight.normal_(0, .02)
        self.config = dict(ratio_maximum=10000., gate_hidden=16, gate_initial_bias=-2.)
        self.source = indices_to_features(torch.randint(256, (1, 16, 16, 4), dtype=torch.uint8))
        self.snrs = torch.tensor([7.])
        self.signal = self.parent.transmit(self.source, self.snrs)
        self.received = self.signal + torch.randn_like(self.signal) * 10 ** (-7 / 20)

    def system(self, variant):
        torch.manual_seed(219)
        return InnovationSystem(self.parent, variant, self.config)

    def test_same_transmitter_and_zero_feature_initialization(self):
        outputs, extra_states = {}, []
        with torch.no_grad():
            for variant in VARIANTS:
                system = self.system(variant).eval()
                self.assertTrue(torch.equal(system.transmit(self.source, self.snrs), self.signal))
                result = system.receive(self.received, self.snrs)
                outputs[variant] = result['receiver_features']
                self.assertEqual([tuple(state.shape[-2:]) for state in result['states']], [(4, 4), (8, 8)])
                self.assertEqual(tuple(result['receiver_features'].shape), (1, 32, 16, 16))
                if system.has_feature:
                    self.assertEqual(len(result['predicted_symbols']), 2)
                    extra_states.append({key: value.clone() for key, value in system.state_dict().items() if key.startswith(('fusion_projection.', 'feature_gate.'))})
                else:
                    self.assertEqual(len(result['predicted_symbols']), 0)
            parent = self.parent.receive(self.received, self.snrs)['receiver_features']
        self.assertTrue(torch.equal(outputs['single_pass'], parent))
        self.assertTrue(torch.equal(outputs['multiscale_no_history'], parent))
        self.assertTrue(torch.equal(outputs['multiscale_state_history'], outputs['multiscale_prediction_features']))
        self.assertTrue(torch.equal(outputs['multiscale_state_history'], outputs['multiscale_innovation']))
        self.assertTrue(all(torch.equal(extra_states[0][key], values[key]) for values in extra_states[1:] for key in values))
        self.assertEqual(list(inspect.signature(InnovationSystem.receive).parameters), ['self', 'received', 'snrs'])

    def test_encoder_is_frozen_but_source_hypothesis_gradient_survives(self):
        system = self.system('multiscale_innovation').train()
        self.assertFalse(system.encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in system.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in system.receiver.parameters()))
        with torch.no_grad():
            system.fusion_projection.weight.normal_(0, .02)
        result = system.receive(self.received, self.snrs)
        for value in result['predicted_symbols'] + result['source_hypotheses']:
            self.assertTrue(value.requires_grad)
            value.retain_grad()
        decoder = nn.Conv2d(32, 3, 1).requires_grad_(False)
        image = functional.interpolate(decoder(result['receiver_features']).sigmoid(), (64, 64), mode='bilinear', align_corners=False)
        (image - .3).square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in system.encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        self.assertGreater(sum(float(parameter.grad.square().sum()) for parameter in system.receiver.parameters() if parameter.grad is not None), 0)
        self.assertTrue(all(value.grad is not None and float(value.grad.norm()) > 0 for value in result['predicted_symbols']))
        self.assertTrue(all(value.grad is not None and float(value.grad.norm()) > 0 for value in result['source_hypotheses']))

    def test_actual_observation_residual_changes_the_feature(self):
        predicted = self.system('multiscale_prediction_features').eval()
        innovation = self.system('multiscale_innovation').eval()
        with torch.no_grad():
            predicted.fusion_projection.weight.normal_(0, .02)
            innovation.load_state_dict(predicted.state_dict())
            first = predicted.receive(self.received, self.snrs)
            second = innovation.receive(self.received, self.snrs)
        self.assertGreater(float((first['receiver_features'] - second['receiver_features']).abs().max()), 1e-6)
        expected = (self.received - second['predicted_symbols'][0]).square().mean((1, 2)) / torch.pow(10., -self.snrs / 10)
        torch.testing.assert_close(second['residual_noise_ratios'][0], expected)
        for hypothesis in second['source_hypotheses']:
            self.assertLessEqual(float(hypothesis.abs().max()), 1.)


if __name__ == '__main__':
    unittest.main()
