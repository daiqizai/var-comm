import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wetok_comm.native import features_to_indices, hard_native_st, indices_to_features


class NativeInterfaceTests(unittest.TestCase):
    def test_complete_native_group_roundtrip(self):
        indices = torch.arange(256, dtype=torch.int64).reshape(1, 16, 16, 1).expand(2, -1, -1, 4).to(torch.uint8)
        features = indices_to_features(indices)
        self.assertEqual(tuple(features.shape), (2, 32, 16, 16))
        self.assertTrue(torch.equal(features_to_indices(features), indices))
        self.assertTrue(torch.all((features == -1) | (features == 1)))

    def test_actual_little_endian_channel_order(self):
        indices = torch.zeros((1, 16, 16, 4), dtype=torch.uint8)
        indices[0, 0, 0, 0] = 1
        indices[0, 0, 0, 1] = 128
        features = indices_to_features(indices)
        self.assertEqual(features[0, 0, 0, 0], 1)
        self.assertEqual(features[0, 15, 0, 0], 1)
        self.assertEqual(int((features[0, :, 0, 0] == 1).sum()), 2)

    def test_hard_values_and_identity_ST_gradient(self):
        logits = torch.tensor([-2., -0., 0., .1, 3.], requires_grad=True)
        hard = hard_native_st(logits)
        self.assertTrue(torch.equal(hard, torch.tensor([-1., -1., -1., 1., 1.])))
        hard.sum().backward()
        self.assertTrue(torch.equal(logits.grad, torch.ones_like(logits)))

    def test_continuous_features_cannot_be_called_native_indices(self):
        with self.assertRaises(ValueError):
            features_to_indices(torch.zeros(1, 32, 16, 16))


if __name__ == '__main__':
    unittest.main()
