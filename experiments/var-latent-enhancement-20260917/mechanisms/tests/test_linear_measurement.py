import unittest

import numpy as np
import torch

from latent_mechanisms.linear_measurement import (
    ENHANCEMENT_USES,
    MEASUREMENT_DIM,
    receive_projection,
    fit_scalar_w,
    fit_training_norm_range,
    load_protocol_config,
    quantize_log_norm,
    transmit_projection,
)


class LinearMeasurementTests(unittest.TestCase):
    def test_protocol_pays_control_uses(self):
        protocol = load_protocol_config()
        self.assertEqual(protocol["control_uses"], 32)
        self.assertEqual(protocol["measurement_uses"], 992)
        self.assertEqual(protocol["measurement_dim"], 1984)
        self.assertEqual(2 * ENHANCEMENT_USES, 2048)

    def test_quantization_zero_and_amplitude_codes(self):
        protocol = load_protocol_config()
        codes = quantize_log_norm(torch.tensor([0.0, 1.0, 3.0]))
        self.assertEqual(int(codes[0]), 0)
        self.assertLess(int(codes[1]), int(codes[2]))
        with self.assertRaises(FloatingPointError):
            quantize_log_norm(torch.tensor([float("nan")]))

    def test_norm_range_comes_from_training_values(self):
        low, high = fit_training_norm_range(torch.tensor([1.0, 2.0, 4.0, 8.0]))
        self.assertLess(low, high)
        fit_training_norm_range(torch.tensor([0.0, 1.0, 2.0]))
        with self.assertRaises(ValueError):
            fit_training_norm_range(torch.tensor([-1.0, 1.0]))

    def test_noiseless_phy_roundtrip_and_resource_energy(self):
        protocol = load_protocol_config()
        measurement = torch.randn(2, MEASUREMENT_DIM)
        noise = torch.zeros(2, ENHANCEMENT_USES, 2)
        signal, observed, metadata = transmit_projection(measurement, noise, 19.0, protocol=protocol)
        self.assertEqual(tuple(signal.shape), (2, ENHANCEMENT_USES, 2))
        torch.testing.assert_close(signal.square().sum((1, 2)), torch.full((2,), 2 * ENHANCEMENT_USES, dtype=signal.dtype), atol=1e-4, rtol=1e-6)
        received = receive_projection(observed, 19.0, protocol=protocol)
        self.assertTrue(bool(received["control_ok"].all()))
        self.assertEqual(tuple(received["measurement"].shape), (2, MEASUREMENT_DIM))
        self.assertTrue(torch.isfinite(received["measurement"]).all())
        self.assertEqual(tuple(metadata["norm_code"].shape), (2,))

    def test_control_corruption_is_fixed_fallback(self):
        protocol = load_protocol_config()
        measurement = torch.ones(1, MEASUREMENT_DIM)
        signal, observed, _ = transmit_projection(measurement, torch.zeros(1, ENHANCEMENT_USES, 2), 19.0, protocol=protocol)
        corrupted = observed.clone()
        corrupted[:, :protocol["control_uses"], :] *= -1
        received = receive_projection(corrupted, 19.0, protocol=protocol)
        self.assertFalse(bool(received["control_ok"].item()))
        self.assertTrue(torch.equal(received["measurement"], torch.zeros_like(received["measurement"])))

    def test_pooled_w_is_batch_partition_invariant(self):
        generator = torch.Generator().manual_seed(7)
        target = torch.randn(8, 16, generator=generator)
        correction = torch.randn(8, 16, generator=generator)
        valid = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.bool)
        whole = fit_scalar_w(target, correction, valid)
        pieces = [fit_scalar_w(target[start:start + 2], correction[start:start + 2], valid[start:start + 2])
                  for start in range(0, 8, 2)]
        numerator = sum(float((target[start:start + 2][valid[start:start + 2]] * correction[start:start + 2][valid[start:start + 2]]).sum())
                        for start in range(0, 8, 2))
        denominator = sum(float(correction[start:start + 2][valid[start:start + 2]].square().sum())
                          for start in range(0, 8, 2))
        self.assertAlmostEqual(whole, numerator / denominator)
        self.assertNotEqual(len(pieces), 0)
        self.assertLessEqual(fit_scalar_w(target * 100, correction, valid), 4.0)
        self.assertAlmostEqual(fit_scalar_w(torch.cat((target, target)), torch.cat((correction, correction)), torch.cat((valid, valid))), whole)


if __name__ == "__main__":
    unittest.main()
