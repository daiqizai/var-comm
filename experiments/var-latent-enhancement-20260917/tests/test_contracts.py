import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from latent_enhancement.latent import (
    ContinuousDecoder, append_observations, enhancement_noise, normalize_enhancement, observed_status,
)
from var_comm.study import seeded_noise


class ContractTests(unittest.TestCase):
    def test_copy_storage_and_input_gradient(self):
        vae = SimpleNamespace(post_quant_conv=torch.nn.Conv2d(3, 3, 1), decoder=torch.nn.Conv2d(3, 3, 1))
        original = copy.deepcopy(vae.post_quant_conv.state_dict())
        decoder = ContinuousDecoder(vae)
        self.assertNotEqual(decoder.post_quant_conv.weight.data_ptr(), vae.post_quant_conv.weight.data_ptr())
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001)
        decoder(torch.randn(2, 3, 4, 4)).mean().backward()
        optimizer.step()
        for name, parameter in vae.post_quant_conv.state_dict().items():
            self.assertTrue(torch.equal(parameter, original[name]))
        decoder.requires_grad_(False)
        latent = torch.randn(2, 3, 4, 4, requires_grad=True)
        decoder(latent).mean().backward()
        self.assertGreater(float(latent.grad.abs().sum()), 0)

    def test_energy_and_degenerate_signal(self):
        for uses in (512, 1024):
            for values in (torch.randn(3, uses, 2), torch.zeros(3, uses, 2)):
                waveform = normalize_enhancement(values)
                torch.testing.assert_close(waveform.square().sum((1, 2)), torch.full((3,), 2.0 * uses))

    def test_nested_independent_noise_and_unchanged_base(self):
        base = np.random.default_rng(22).choice([-1.0, 1.0], (3060, 2))
        large_noise = enhancement_noise("example", 2001, 1024)
        np.testing.assert_array_equal(large_noise[:512], enhancement_noise("example", 2001, 512))
        self.assertFalse(np.array_equal(large_noise, seeded_noise("example", 2001, (3060, 2))[:1024]))
        for uses in (512, 1024):
            observation, waveform = append_observations(base, np.ones((uses, 2)), "example", 2001, 7)
            np.testing.assert_array_equal(waveform[:3060], base)
            np.testing.assert_array_equal(observation[:3060], base + seeded_noise("example", 2001, (3060, 2)) / np.sqrt(10 ** 0.7))
            self.assertEqual(float(np.square(waveform).sum()), 2 * (3060 + uses))

    def test_status_is_observable_only(self):
        failed = {"header": {"accepted": False, "mode": 8}, "events": []}
        np.testing.assert_array_equal(observed_status(failed), [0, 0, 0])
        body_failed = {"header": {"accepted": True, "mode": 7}, "events": [{"accepted": False}]}
        np.testing.assert_array_equal(observed_status(body_failed), [1, 0, 7])


if __name__ == "__main__":
    unittest.main()
