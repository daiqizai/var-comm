import inspect
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from innovation_comm.model import InnovationSystem
from innovation_comm.runtime import actual_observation, receiver_loss
from joint_sender.model import JointSenderSystem, VARIANTS
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


class JointSenderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.manual_seed(312)
        self.parent = GeometryJSCC('single_pass', dict(width=24, heads=2, encoder_layers=1, receiver_layers=1,
            channel_positions=204, channel_features=30)).eval().requires_grad_(False)
        with torch.no_grad():
            self.parent.receiver.output.weight.normal_(0, .02)
        self.fusion = {'ratio_maximum': 10000., 'gate_hidden': 16, 'gate_initial_bias': -2.}
        self.inputs = {'source_fq': indices_to_features(torch.randint(256, (4, 16, 16, 4), dtype=torch.uint8)),
            'images': torch.rand(4, 3, 16, 16), 'snrs': torch.tensor([1., 4., 13., 19.]), 'noise': torch.randn(4, 3060, 2)}
        self.decoder, self.perceptual = ToyDecoder(), ToyPerceptual()
        self.weights = {'mse': 1., 'lpips': .01, 'bits': .01, 'state': .01}
        self.waveform_tolerance = yaml.safe_load((INNOVATION / 'configs/evaluation.yaml').read_text())['parent_signal_replay_tolerance']

    def test_no_architecture_change_at_the_common_start(self):
        with torch.no_grad():
            expected = self.parent.transmit(self.inputs['source_fq'], self.inputs['snrs'])
            for variant in VARIANTS:
                control = InnovationSystem(self.parent, variant, self.fusion).eval()
                joint = JointSenderSystem(self.parent, variant, self.fusion).eval()
                self.assertEqual(module_sha256(control), module_sha256(joint))
                self.assertTrue(torch.equal(control.transmit(self.inputs['source_fq'], self.inputs['snrs']), expected))
                torch.testing.assert_close(joint.transmit(self.inputs['source_fq'], self.inputs['snrs']), expected,
                                           atol=self.waveform_tolerance, rtol=0)
                received = expected + self.inputs['noise'] * torch.pow(10., self.inputs['snrs'] / 10).rsqrt()[:, None, None]
                self.assertTrue(torch.equal(control.receive(received, self.inputs['snrs'])['receiver_features'],
                                            joint.receive(received, self.inputs['snrs'])['receiver_features']))
                self.assertTrue(all(value.requires_grad for value in joint.encoder.parameters()))
                self.assertTrue(all(not value.requires_grad for value in control.encoder.parameters()))
                torch.testing.assert_close(joint.double().transmit(self.inputs['source_fq'].double(), self.inputs['snrs'].double()),
                    control.double().transmit(self.inputs['source_fq'].double(), self.inputs['snrs'].double()), atol=1e-12, rtol=0)
        self.assertEqual(list(inspect.signature(JointSenderSystem.receive).parameters), ['self', 'received', 'snrs'])

    def test_image_loss_reaches_sender_and_microbatch_gradients_match_batch4(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                joint = JointSenderSystem(self.parent, variant, self.fusion)
                vectorized = JointSenderSystem(self.parent, variant, self.fusion).train()
                before = module_sha256(joint)
                backward_calls = []
                handle = next(joint.encoder.parameters()).register_hook(lambda gradient: backward_calls.append(1))
                report = backward_paired_batch(joint, self.inputs, self.decoder, self.perceptual, self.weights)
                handle.remove()
                self.assertEqual(len(backward_calls), 1)
                self.assertGreater(report['encoder_gradient_norm'], 0)
                self.assertGreater(report['receiver_gradient_norm'], 0)
                self.assertEqual(module_sha256(joint), before)
                signal, received = observation(vectorized, self.inputs)
                result = vectorized.receive(received, self.inputs['snrs'])
                objective, components, diagnostic = interface_losses(result, self.inputs['source_fq'], self.inputs['images'],
                    self.decoder, self.perceptual, self.weights)
                objective.backward()
                self.assertAlmostEqual(report['loss'], float(objective.detach()), places=6)
                expected = dict(vectorized.named_parameters())
                for name, parameter in joint.named_parameters():
                    if parameter.grad is None:
                        self.assertIsNone(expected[name].grad)
                    else:
                        torch.testing.assert_close(parameter.grad, expected[name].grad, atol=2e-6, rtol=2e-4)
                image_only = JointSenderSystem(self.parent, variant, self.fusion)
                image_weights = {'mse': 1., 'lpips': .01, 'bits': 0., 'state': 0.}
                report = backward_paired_batch(image_only, self.inputs, self.decoder, self.perceptual, image_weights)
                self.assertGreater(report['encoder_gradient_norm'], 0)
                self.assertTrue(all(parameter.grad is None for parameter in self.decoder.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in self.parent.parameters()))

    def test_leaf_flag_roundoff_is_localized_not_changed_weights(self):
        frozen = JointSenderSystem(self.parent, 'single_pass', self.fusion, False).eval()
        joint = JointSenderSystem(self.parent, 'single_pass', self.fusion, True).eval()
        self.assertEqual(module_sha256(frozen), module_sha256(joint))
        values = self.inputs['source_fq'].flatten(2).transpose(1, 2)
        with torch.no_grad():
            torch.testing.assert_close(frozen.encoder.channel_lift(values), joint.encoder.channel_lift(values),
                                       atol=self.waveform_tolerance, rtol=0)
            self.assertTrue(torch.equal(frozen.encoder.channel_lift(values.contiguous()), joint.encoder.channel_lift(values.contiguous())))

    def test_frozen_mode_matches_the_existing_receiver_only_training_math(self):
        for variant in VARIANTS:
            policy = JointSenderSystem(self.parent, variant, self.fusion, update_sender=False)
            old = InnovationSystem(self.parent, variant, self.fusion).train()
            report = backward_paired_batch(policy, self.inputs, self.decoder, self.perceptual, self.weights)
            signal, received = actual_observation(self.parent, self.inputs)
            old_objective = 0.
            for index in range(4):
                part = {name: value[index:index + 1] for name, value in self.inputs.items()}
                objective, components, diagnostics, result = receiver_loss(old, received[index:index + 1], part,
                    self.decoder, self.perceptual, self.weights)
                (objective / 4).backward()
                old_objective += float(objective.detach()) / 4
            self.assertEqual(report['encoder_gradient_norm'], 0)
            self.assertEqual(report['loss'], old_objective)
            expected = dict(old.named_parameters())
            for name, parameter in policy.named_parameters():
                if parameter.grad is None:
                    self.assertIsNone(expected[name].grad)
                else:
                    torch.testing.assert_close(parameter.grad, expected[name].grad, atol=0, rtol=0)

    def test_optimizer_changes_only_allowed_parameters_and_preserves_power(self):
        for update_sender in (False, True):
            system = JointSenderSystem(self.parent, 'single_pass', self.fusion, update_sender=update_sender)
            encoder_before = module_sha256(system.encoder)
            receiver_before = module_sha256(system.receiver)
            backward_paired_batch(system, self.inputs, self.decoder, self.perceptual, self.weights)
            optimizer = torch.optim.AdamW([parameter for parameter in system.parameters() if parameter.requires_grad], lr=1e-4)
            optimizer.step()
            self.assertEqual(module_sha256(system.encoder) != encoder_before, update_sender)
            self.assertNotEqual(module_sha256(system.receiver), receiver_before)
            signal = system.transmit(self.inputs['source_fq'], self.inputs['snrs'])
            self.assertLess(float((signal.detach().square().sum(-1).mean(-1) - 2).abs().max()), 1e-5)


if __name__ == '__main__':
    unittest.main()
