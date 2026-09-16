import inspect
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))
from wetok_comm.model import NativeWeTokJSCC, communicate
from wetok_comm.native import indices_to_features
from wetok_comm.objective import losses


class DecoderFixture:
    def __init__(self):
        self.module = nn.Conv2d(32, 3, 1).requires_grad_(False)
        self.calls = 0

    def decode(self, features):
        self.calls += 1
        return functional.interpolate(self.module(features).sigmoid(), (256, 256), mode='bilinear', align_corners=False)


class PerceptualFixture(nn.Module):
    def forward(self, image, target):
        return (image - target).abs().flatten(1).mean(1)[:, None]


class CommunicationTests(unittest.TestCase):
    def test_equal_initial_state_resource_and_gradient_paths(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        native = indices_to_features(torch.randint(256, (1, 16, 16, 4), dtype=torch.uint8))
        snrs, noise = torch.tensor([7.]), torch.randn(1, 3060, 2)
        initial, count = None, None
        for variant in config['arms']:
            torch.manual_seed(83)
            network = NativeWeTokJSCC(variant, config['model'])
            current = network.state_dict()
            if initial is None:
                initial = {name: value.clone() for name, value in current.items()}
                count = sum(value.numel() for value in network.parameters())
            else:
                self.assertTrue(all(torch.equal(initial[name], value) for name, value in current.items()))
                self.assertEqual(sum(value.numel() for value in network.parameters()), count)
            result = communicate(network, native, snrs, noise)
            torch.testing.assert_close(result['transmitted'].square().sum(-1).mean(-1), torch.tensor([2.]), atol=1e-6, rtol=0)
            self.assertEqual(tuple(result['native_fq'].shape), (1, 32, 16, 16))
            self.assertEqual([tuple(value.shape[-2:]) for value in result['states']], [(4, 4), (8, 8)])
            decoder = DecoderFixture()
            loss, components, diagnostics = losses(result, native, torch.rand(1, 3, 256, 256), decoder,
                PerceptualFixture(), config['training']['representation_weights'])
            self.assertEqual(decoder.calls, 0)
            loss.backward()
            self.assertTrue(all(value.grad is not None for value in network.parameters()))
            self.assertGreater(sum(float(value.grad.square().sum()) for value in network.encoder.parameters()), 0)
            self.assertGreater(sum(float(value.grad.square().sum()) for value in network.receiver.parameters()), 0)
            network.zero_grad(set_to_none=True)
            result = communicate(network, native, snrs, noise)
            weights = dict(mse=1., lpips=.01, bits=0., state=0.)
            image_loss, unused, unused = losses(result, native, torch.rand(1, 3, 256, 256), decoder, PerceptualFixture(), weights)
            image_loss.backward()
            self.assertGreater(sum(float(value.grad.square().sum()) for value in network.encoder.parameters() if value.grad is not None), 0)
            self.assertTrue(all(value.grad is None for value in decoder.module.parameters()))
        print('COMMUNICATION_PARAMETERS_PER_ARM', count)

    def test_receiver_does_not_accept_source_or_class(self):
        parameters = list(inspect.signature(NativeWeTokJSCC.receive).parameters)
        self.assertEqual(parameters, ['self', 'received', 'snrs'])


if __name__ == '__main__':
    unittest.main()
