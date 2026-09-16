import inspect
from pathlib import Path
import sys
import unittest

import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

from wetok_comm.geometry_candidate import GeometryJSCC, matched_geometry_initializations
from wetok_comm.interface_study import InterfaceJSCC, interface_losses
from wetok_comm.model import NativeWeTokJSCC, communicate
from wetok_comm.native import indices_to_features
from wetok_comm.objective import losses
from test_model import DecoderFixture, PerceptualFixture


class GeometryCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        self.base = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())

    def test_original_geometry_is_an_exact_unchanged_control(self):
        for variant in self.base['arms']:
            torch.manual_seed(self.base['training']['initialization_seed'])
            original = InterfaceJSCC(variant, self.base['model'], 'continuous_mean').eval()
            control, candidate, evidence = matched_geometry_initializations(self.base, variant)
            control.eval()
            self.assertEqual(evidence['control_parameters'], 2895944)
            self.assertTrue(all(torch.equal(value, control.state_dict()[name]) for name, value in original.state_dict().items()))
            source = indices_to_features(torch.randint(256, (1, 16, 16, 4), dtype=torch.uint8))
            snrs, noise = torch.tensor([19.]), torch.randn(1, 3060, 2)
            with torch.no_grad():
                first = communicate(original, source, snrs, noise)
                second = communicate(control, source, snrs, noise)
            self.assertTrue(torch.equal(first['transmitted'], second['transmitted']))
            self.assertTrue(torch.equal(first['receiver_features'], second['receiver_features']))

    def test_row_orthogonal_geometry_and_shared_initial_neural_parameters(self):
        control, candidate, evidence = matched_geometry_initializations(self.base, 'multiscale_conditioned')
        spatial = candidate.encoder.spatial_mixing.detach()
        channel = candidate.encoder.channel_lift.weight.detach()
        self.assertEqual(tuple(spatial.shape), (204, 256))
        self.assertEqual(tuple(channel.shape), (30, 32))
        torch.testing.assert_close(spatial @ spatial.t(), torch.eye(204), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(channel @ channel.t(), torch.eye(30), atol=1e-6, rtol=1e-5)
        self.assertEqual(int(torch.linalg.matrix_rank(spatial)) * int(torch.linalg.matrix_rank(channel)), 6120)
        self.assertTrue(torch.equal(spatial[:153], control.encoder.spatial_mixing.detach()))
        for name in evidence['shared_initial_tensors']:
            self.assertTrue(torch.equal(candidate.state_dict()[name], control.state_dict()[name]))
        self.assertGreater(evidence['candidate_parameters'], evidence['control_parameters'])
        print('PREPARED_GEOMETRY_PARAMETERS', evidence['control_parameters'], evidence['candidate_parameters'])

    def test_physical_budget_interface_and_image_gradient(self):
        unused, network, unused_evidence = matched_geometry_initializations(self.base, 'multiscale_conditioned')
        source = indices_to_features(torch.randint(256, (1, 16, 16, 4), dtype=torch.uint8))
        result = communicate(network, source, torch.tensor([7.]), torch.randn(1, 3060, 2))
        self.assertEqual(tuple(result['transmitted'].shape), (1, 3060, 2))
        torch.testing.assert_close(result['transmitted'].square().sum(-1).mean(-1), torch.tensor([2.]), atol=1e-6, rtol=0)
        self.assertEqual(tuple(result['receiver_features'].shape), (1, 32, 16, 16))
        decoder = DecoderFixture()
        objective, unused, unused = interface_losses(result, source, torch.rand(1, 3, 256, 256), decoder, PerceptualFixture(),
            {'mse': 1., 'lpips': .01, 'bits': 0., 'state': 0.})
        objective.backward()
        for component in (network.encoder, network.receiver):
            self.assertGreater(sum(float(value.grad.square().sum()) for value in component.parameters() if value.grad is not None), 0)
        self.assertTrue(all(value.grad is None for value in decoder.module.parameters()))
        self.assertEqual(list(inspect.signature(GeometryJSCC.receive).parameters), ['self', 'received', 'snrs'])

    def test_original_representation_loss_gradients_and_adam_update_match(self):
        for variant in self.base['arms']:
            torch.manual_seed(self.base['training']['initialization_seed'])
            original = NativeWeTokJSCC(variant, self.base['model'])
            control, unused, unused_evidence = matched_geometry_initializations(self.base, variant)
            sources = indices_to_features(torch.randint(256, (4, 16, 16, 4), dtype=torch.uint8))
            snrs, noise = torch.tensor([1., 4., 7., 19.]), torch.randn(4, 3060, 2)
            targets = torch.zeros(4, 3, 256, 256)
            weights = self.base['training']['representation_weights']
            optimizers = [torch.optim.AdamW(network.parameters(), lr=self.base['training']['representation_learning_rate'],
                betas=tuple(self.base['training']['betas']), weight_decay=self.base['training']['weight_decay'])
                for network in (original, control)]
            for index in range(4):
                old = communicate(original, sources[index:index + 1], snrs[index:index + 1], noise[index:index + 1])
                new = communicate(control, sources[index:index + 1], snrs[index:index + 1], noise[index:index + 1])
                old_loss, unused, unused = losses(old, sources[index:index + 1], targets[index:index + 1], None, None, weights)
                new_loss, unused, unused = interface_losses(new, sources[index:index + 1], targets[index:index + 1], None, None, weights)
                self.assertTrue(torch.equal(old_loss.detach(), new_loss.detach()))
                (old_loss / 4).backward()
                (new_loss / 4).backward()
            for first, second in zip(original.parameters(), control.parameters()):
                self.assertIsNotNone(first.grad)
                self.assertTrue(torch.equal(first.grad, second.grad))
            for network, optimizer in zip((original, control), optimizers):
                torch.nn.utils.clip_grad_norm_(network.parameters(), self.base['training']['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
            self.assertTrue(all(torch.equal(value, control.state_dict()[name]) for name, value in original.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
