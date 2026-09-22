import unittest

import numpy as np
import torch

from latent_enhancement_eval.deployment import (
    DEFAULT_LATENT_SHAPE, channel_apply, rx_base, tx_encode,
)


class DeploymentBoundaryTests(unittest.TestCase):
    def test_resource_ledger_and_segment_noise(self):
        base = np.zeros((3060, 2), dtype=np.float32)
        enhancement = np.zeros((512, 2), dtype=np.float32)
        encoded = tx_encode(base, enhancement)
        self.assertEqual(encoded["signal"].shape, (3572, 2))
        channel = channel_apply(encoded, "cpu-test-image", 2001, 7)
        self.assertEqual(channel["base_received"].shape, base.shape)
        self.assertEqual(channel["enhancement_received"].shape, enhancement.shape)
        self.assertTrue(np.array_equal(channel["received"], np.concatenate((channel["base_received"], channel["enhancement_received"]))))

    def test_header_failure_uses_registered_shape(self):
        def failed_receive(_observation, _snr):
            return {"header": {"accepted": False, "mode": 0}, "label": None, "events": []}

        result = rx_base(np.zeros((3060, 2), dtype=np.float32), 1, receive_fn=failed_receive)
        self.assertEqual(tuple(result["latent"].shape), DEFAULT_LATENT_SHAPE)
        self.assertFalse(result["header_ok"])

    def test_nonfinite_snr_and_waveform_rejected(self):
        with self.assertRaises(FloatingPointError):
            tx_encode(np.full((3060, 2), np.nan, dtype=np.float32))
        with self.assertRaises(ValueError):
            channel_apply(tx_encode(np.zeros((3060, 2), dtype=np.float32)), "x", 1, np.nan)


if __name__ == "__main__":
    unittest.main()

