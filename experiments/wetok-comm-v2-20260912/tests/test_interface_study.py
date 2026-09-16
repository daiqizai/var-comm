import inspect
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wetok_comm.interface_study import INTERFACES, InterfaceJSCC, interface_losses, receiver_features
from wetok_comm.model import communicate
from wetok_comm.native import features_to_indices, indices_to_features
from test_model import DecoderFixture, PerceptualFixture


class InterfaceStudyTests(unittest.TestCase):
    def test_hard_forward_identity_and_analytic_gradients(self):
        for interface in INTERFACES:
            logits = torch.tensor([-100., -8., -1., 0., .1, 2., 100.], requires_grad=True)
            result = receiver_features(logits, interface)
            expected = torch.tanh(logits.detach() / 2) if interface == 'continuous_mean' else torch.where(logits.detach() > 0, 1., -1.)
            self.assertTrue(torch.equal(result.detach(), expected))
            result.sum().backward()
            derivative = torch.ones_like(logits) if interface == 'hard_identity' else .5 * (1 - torch.tanh(logits.detach() / 2).square())
            torch.testing.assert_close(logits.grad, derivative)
            with torch.no_grad():
                self.assertTrue(torch.equal(result.detach(), receiver_features(logits, interface)))

    def test_continuous_is_not_a_native_group_serialization(self):
        with self.assertRaises(ValueError):
            features_to_indices(receiver_features(torch.zeros(1, 32, 16, 16), 'continuous_mean'))
        with self.assertRaises(ValueError):
            receiver_features(torch.zeros(1), 'unregistered')

    def test_same_parameters_physics_and_real_decoder_input_gradients(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        config = dict(width=24, heads=2, encoder_layers=1, receiver_layers=1, channel_positions=153, channel_features=40)
        truth = indices_to_features(torch.randint(256, (1, 16, 16, 4), dtype=torch.uint8))
        initial, hard_output = None, None
        snrs, noise = torch.tensor([7.]), torch.randn(1, 3060, 2)
        for interface in INTERFACES:
            torch.manual_seed(41)
            network = InterfaceJSCC('multiscale_conditioned', config, interface)
            if initial is None:
                initial = {key: value.clone() for key, value in network.state_dict().items()}
            self.assertTrue(all(torch.equal(initial[key], value) for key, value in network.state_dict().items()))
            result = communicate(network, truth, snrs, noise)
            torch.testing.assert_close(result['transmitted'].square().sum(-1).mean(-1), torch.tensor([2.]), atol=1e-6, rtol=0)
            if interface.startswith('hard'):
                if hard_output is None:
                    hard_output = result['receiver_features'].detach().clone()
                self.assertTrue(torch.equal(hard_output, result['receiver_features'].detach()))
            decoder = DecoderFixture()
            objective, unused, unused = interface_losses(result, truth, torch.rand(1, 3, 256, 256), decoder,
                PerceptualFixture(), dict(mse=1., lpips=.01, bits=0., state=0.))
            objective.backward()
            for part in (network.encoder, network.receiver):
                self.assertGreater(sum(float(value.grad.square().sum()) for value in part.parameters() if value.grad is not None), 0)
            self.assertTrue(all(value.grad is None for value in decoder.module.parameters()))
        self.assertEqual(list(inspect.signature(InterfaceJSCC.receive).parameters), ['self', 'received', 'snrs'])


if __name__ == '__main__':
    unittest.main()
