import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path.insert(0, str(EXPERIMENT / 'src'))

import torch
from torch import nn

from grid_controls.model import GridJointSystem
from joint_sender.runtime import backward_paired_batch, observation
from vector_control.model import PredictionVectorSystem
from wetok_comm.geometry_candidate import GeometryJSCC
from wetok_comm.native import indices_to_features
from wetok_comm.training import module_sha256


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(32, 3, 1)
        self.requires_grad_(False)

    def decode(self, features):
        return self.output(features).sigmoid()


class ToyPerceptual(nn.Module):
    def forward(self, images, targets):
        return (images - targets).square().mean((1, 2, 3), keepdim=True)


def parent_factory():
    torch.manual_seed(312)
    parent = GeometryJSCC('single_pass', dict(width=24, heads=2, encoder_layers=1, receiver_layers=1,
        channel_positions=204, channel_features=30)).eval().requires_grad_(False)
    with torch.no_grad():
        parent.receiver.output.weight.normal_(0, .02)
    return parent


class VectorModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        self.parent = parent_factory()
        self.fusion = {'ratio_maximum': 8., 'gate_hidden': 16, 'gate_initial_bias': -2.}
        torch.manual_seed(81)
        self.new = PredictionVectorSystem(self.parent, self.fusion).eval()
        torch.manual_seed(81)
        self.old = GridJointSystem(self.parent, 'full_grid_innovation', self.fusion).eval()
        self.inputs = {'source_fq': indices_to_features(torch.randint(256, (4, 16, 16, 4), dtype=torch.uint8)),
            'images': torch.rand(4, 3, 16, 16), 'snrs': torch.tensor([1., 4., 13., 19.]), 'noise': torch.randn(4, 3060, 2)}

    def test_exact_initial_weights_outputs_and_two_reencodings_but_different_vector(self):
        self.assertEqual(module_sha256(self.new), module_sha256(self.old))
        self.assertEqual(list(inspect.signature(PredictionVectorSystem.receive).parameters), ['self', 'received', 'snrs'])
        with torch.no_grad():
            signal, received = observation(self.new, self.inputs)
            self.assertLess(float((signal.square().sum(-1).mean(-1) - 2).abs().max()), 1e-5)
            projected = []
            original = self.new._read_feature

            def read_feature(guessed, memory, snrs, size, previous, feature, gate):
                projected.append((size, previous.clone(), feature.clone(), gate.clone()))
                return original(guessed, memory, snrs, size, previous, feature, gate)

            with mock.patch.object(self.new, '_read_feature', side_effect=read_feature), \
                    mock.patch.object(self.new.encoder, 'forward', wraps=self.new.encoder.forward) as new_calls:
                actual = self.new.receive(received, self.inputs['snrs'])
            with mock.patch.object(self.old.encoder, 'forward', wraps=self.old.encoder.forward) as old_calls:
                expected = self.old.receive(received, self.inputs['snrs'])
        self.assertEqual((new_calls.call_count, old_calls.call_count), (2, 2))
        self.assertEqual([row[0] for row in projected], [16, 16])
        for key in ('logits', 'native_fq', 'receiver_features'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        for key in ('states', 'predicted_symbols', 'source_hypotheses', 'feature_gates', 'residual_noise_ratios'):
            self.assertEqual(len(actual[key]), 2)
            for current, reference in zip(actual[key], expected[key]):
                torch.testing.assert_close(current, reference, atol=0, rtol=0)
        for stage, row in enumerate(projected):
            torch.testing.assert_close(row[1], actual['source_hypotheses'][stage], atol=0, rtol=0)
            torch.testing.assert_close(row[2], actual['predicted_symbols'][stage], atol=0, rtol=0)
            self.assertFalse(torch.equal(row[2], received - actual['predicted_symbols'][stage]))
            expected_ratio = (received - actual['predicted_symbols'][stage]).square().mean((1, 2)) / torch.pow(10., -self.inputs['snrs'] / 10)
            torch.testing.assert_close(actual['residual_noise_ratios'][stage], expected_ratio, atol=0, rtol=0)

    def test_receiver_cannot_read_last_transmitted_source_or_free_extra_symbols(self):
        with torch.no_grad():
            self.new.fusion_projection.weight.normal_(0, .03)
            unused, received = observation(self.new, self.inputs)
            expected = self.new.receive(received, self.inputs['snrs'])
            self.new.transmit(-self.inputs['source_fq'], self.inputs['snrs'])
            actual = self.new.receive(received, self.inputs['snrs'])
        torch.testing.assert_close(actual['receiver_features'], expected['receiver_features'], atol=0, rtol=0)
        for current, reference in zip(actual['predicted_symbols'], expected['predicted_symbols']):
            torch.testing.assert_close(current, reference, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.new.receive(received[:, :-1], self.inputs['snrs'])

    def test_image_only_loss_reaches_sender_receiver_and_waveform_without_visual_updates(self):
        decoder, perceptual = ToyDecoder(), ToyPerceptual()
        before = module_sha256(decoder)
        weights = {'mse': 1., 'lpips': .01, 'bits': 0., 'state': 0.}
        result = backward_paired_batch(self.new, self.inputs, decoder, perceptual, weights)
        for key in ('encoder_gradient_norm', 'receiver_gradient_norm', 'waveform_gradient_norm'):
            self.assertGreater(result[key], 0.)
        self.assertEqual(module_sha256(decoder), before)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        optimizer = torch.optim.AdamW(self.new.parameters(), lr=1e-4, betas=(.9, .99), weight_decay=1e-4)
        self.assertEqual(len(optimizer.state), 0)
        optimizer.step()
        self.assertTrue(any(torch.count_nonzero(parameter).item() for parameter in self.new.fusion_projection.parameters()))


if __name__ == '__main__':
    unittest.main()
