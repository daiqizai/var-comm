import copy
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from grid_controls.model import GridJointSystem, VARIANTS
from innovation_comm.model import InnovationSystem
from joint_sender.model import JointSenderSystem
from joint_sender.runtime import backward_paired_batch, observation
from wetok_comm.geometry_candidate import GeometryJSCC
from wetok_comm.interface_study import interface_losses
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


class GridControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.manual_seed(312)
        self.parent = GeometryJSCC('single_pass', dict(width=24, heads=2, encoder_layers=1, receiver_layers=1,
            channel_positions=204, channel_features=30)).eval().requires_grad_(False)
        with torch.no_grad():
            self.parent.receiver.output.weight.normal_(0, .02)
        self.fusion = {'ratio_maximum': 8., 'gate_hidden': 16, 'gate_initial_bias': -2.}
        self.inputs = {'source_fq': indices_to_features(torch.randint(256, (4, 16, 16, 4), dtype=torch.uint8)),
            'images': torch.rand(4, 3, 16, 16), 'snrs': torch.tensor([1., 4., 13., 19.]), 'noise': torch.randn(4, 3060, 2)}
        self.decoder, self.perceptual = ToyDecoder(), ToyPerceptual()
        self.weights = {'mse': 1., 'lpips': .01, 'bits': .01, 'state': .01}

    def test_ordinary_grid_adds_no_parameters_and_uses_only_own_history(self):
        ordinary = GridJointSystem(self.parent, 'full_grid_state_history', self.fusion)
        multiscale = JointSenderSystem(self.parent, 'multiscale_state_history', self.fusion)
        self.assertEqual(module_sha256(ordinary), module_sha256(multiscale))
        self.assertEqual(list(inspect.signature(GridJointSystem.receive).parameters), ['self', 'received', 'snrs'])
        signal, received = observation(ordinary, self.inputs)
        history = []
        original_read = ordinary.receiver.read

        def record_read(guessed, memory, snrs, size, previous=None):
            logits = original_read(guessed, memory, snrs, size, previous)
            history.append((size, previous, torch.tanh(logits / 2)))
            return logits

        with mock.patch.object(ordinary.receiver, 'read', side_effect=record_read), \
                mock.patch.object(ordinary.encoder, 'forward', wraps=ordinary.encoder.forward) as internal_encoder:
            result = ordinary.receive(received, self.inputs['snrs'])
        internal_encoder.assert_not_called()
        self.assertEqual([entry[0] for entry in history], [16, 16, 16])
        self.assertIsNone(history[0][1])
        for stage in (1, 2):
            torch.testing.assert_close(history[stage][1], history[stage - 1][2], atol=0, rtol=0)
        for stage, size in enumerate((4, 8)):
            torch.testing.assert_close(result['states'][stage], functional.adaptive_avg_pool2d(history[stage][2], size), atol=0, rtol=0)
        torch.testing.assert_close(result['receiver_features'], history[-1][2], atol=0, rtol=0)
        self.assertLess(float((signal.detach().square().sum(-1).mean(-1)-2).abs().max()), 1e-5)
        with self.assertRaises(ValueError):
            ordinary.receive(received[:, :-1], self.inputs['snrs'])

    def test_innovation_frozen_mode_is_the_existing_control(self):
        torch.manual_seed(1234)
        new = GridJointSystem(self.parent, 'full_grid_innovation', self.fusion, update_sender=False)
        torch.manual_seed(1234)
        old = InnovationSystem(self.parent, 'full_grid_innovation', self.fusion)
        self.assertEqual(module_sha256(new), module_sha256(old))
        with torch.no_grad():
            signal, received = observation(new, self.inputs)
            expected = old.receive(received, self.inputs['snrs'])
            actual = new.receive(received, self.inputs['snrs'])
        torch.testing.assert_close(actual['receiver_features'], expected['receiver_features'], atol=0, rtol=0)
        self.assertEqual(len(actual['predicted_symbols']), 2)
        self.assertTrue(all(not parameter.requires_grad for parameter in new.encoder.parameters()))

    def test_microbatch_vjp_includes_transmit_and_internal_encoder_gradients(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                micro = GridJointSystem(self.parent, variant, self.fusion)
                if micro.has_feature:
                    with torch.no_grad():
                        micro.fusion_projection.weight.normal_(0, .01)
                batch = copy.deepcopy(micro)
                calls = []
                hook = micro.encoder.register_forward_hook(lambda module, inputs, output: calls.append(len(inputs[0])))
                report = backward_paired_batch(micro, self.inputs, self.decoder, self.perceptual, self.weights)
                hook.remove()
                self.assertEqual(calls, [4] + ([1] * 8 if micro.has_feature else []))
                signal, received = observation(batch, self.inputs)
                result = batch.receive(received, self.inputs['snrs'])
                objective, components, diagnostic = interface_losses(result, self.inputs['source_fq'], self.inputs['images'],
                    self.decoder, self.perceptual, self.weights)
                objective.backward()
                self.assertAlmostEqual(report['loss'], float(objective.detach()), places=6)
                expected = dict(batch.named_parameters())
                for name, parameter in micro.named_parameters():
                    if parameter.grad is None:
                        self.assertIsNone(expected[name].grad)
                    else:
                        torch.testing.assert_close(parameter.grad, expected[name].grad, atol=3e-6, rtol=3e-4)
                self.assertGreater(report['encoder_gradient_norm'], 0)
                self.assertGreater(report['receiver_gradient_norm'], 0)

    def test_image_only_gradient_and_internal_encoder_path_are_not_cut(self):
        image_weights = {'mse': 1., 'lpips': .01, 'bits': 0., 'state': 0.}
        for variant in VARIANTS:
            system = GridJointSystem(self.parent, variant, self.fusion)
            before = module_sha256(system)
            report = backward_paired_batch(system, self.inputs, self.decoder, self.perceptual, image_weights)
            self.assertGreater(report['encoder_gradient_norm'], 0)
            self.assertGreater(report['receiver_gradient_norm'], 0)
            self.assertEqual(module_sha256(system), before)
        system = GridJointSystem(self.parent, 'full_grid_innovation', self.fusion)
        with torch.no_grad():
            system.fusion_projection.weight.normal_(0, .02)
            signal, received = observation(system, self.inputs)
        result = system.receive(received.detach(), self.inputs['snrs'])
        objective, components, diagnostic = interface_losses(result, self.inputs['source_fq'], self.inputs['images'],
            self.decoder, self.perceptual, image_weights)
        objective.backward()
        gradient = sum(float(parameter.grad.square().sum()) for parameter in system.encoder.parameters() if parameter.grad is not None)
        self.assertGreater(gradient, 0.)
        self.assertTrue(all(parameter.grad is None for parameter in self.decoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.parent.parameters()))

    def test_frozen_sender_stays_frozen_through_innovation_backprop(self):
        system = GridJointSystem(self.parent, 'full_grid_innovation', self.fusion, update_sender=False)
        with torch.no_grad():
            system.fusion_projection.weight.normal_(0, .01)
        report = backward_paired_batch(system, self.inputs, self.decoder, self.perceptual, self.weights)
        self.assertEqual(report['encoder_gradient_norm'], 0.)
        self.assertGreater(report['receiver_gradient_norm'], 0.)
        self.assertTrue(all(parameter.grad is None for parameter in system.encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
