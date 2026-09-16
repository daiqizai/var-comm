#!/usr/bin/env python3
"""Protocol, observable-information, power, and gradient regression checks."""

import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from var_comm.hybrid_correction import HybridBudget, ResidualLink, encode_digital, normalize_analog, receive_digital, receiver_features
from var_comm.hybrid_training import choose_checkpoint, training_draw
from var_comm.progressive import header_bits
from var_comm.scale_channel import encode_packet


class HybridTests(unittest.TestCase):
    def test_paired_draw_and_epoch_coverage(self):
        config = json.loads((ROOT / "configs/hybrid_source_correction.json").read_text())
        first = training_draw(1, 32, config)
        repeated = training_draw(1, 32, config)
        for original, replay in zip(first, repeated):
            np.testing.assert_array_equal(original, replay)
        second = training_draw(2, 32, config)
        np.testing.assert_array_equal(np.sort(np.concatenate((first[0], second[0]))), np.arange(32))
        self.assertFalse(np.array_equal(first[2], second[2]))

    def test_selection_never_relabels_an_infeasible_checkpoint(self):
        config = json.loads((ROOT / "configs/hybrid_source_correction.json").read_text())
        references = {"raw": {"psnr_db": 20., "lpips": .18}, "arithmetic": {"psnr_db": 20., "lpips": .179}}
        summaries = {"5000": {}, "10000": {}}
        for arm in config["arms"]:
            summaries["5000"][arm] = {"primary": {"psnr_db": 22., "lpips": .2}}
            summaries["10000"][arm] = {"primary": {"psnr_db": 23., "lpips": .19}}
        chosen = choose_checkpoint(summaries, references, config)
        self.assertFalse(chosen["reliability_gain"]["feasible_on_calibration"])
        self.assertEqual(chosen["reliability_gain"]["step"], 10000)
        summaries["5000"]["reliability_gain"]["primary"]["lpips"] = .18
        self.assertEqual(choose_checkpoint(summaries, references, config)["reliability_gain"]["step"], 5000)

    def test_fixed_budget(self):
        ledger = HybridBudget().ledger()
        self.assertEqual(ledger["source_bits"], 1860)
        self.assertEqual(ledger["header_uses"] + ledger["digital_uses"] + ledger["analog_uses"], 3060)
        self.assertEqual(ledger["header_energy"] + ledger["digital_energy"] + ledger["analog_energy"], 6120)
        self.assertEqual(ledger["body_information_code_rate"], 0.5)
        with self.assertRaises(ValueError):
            HybridBudget(digital_uses=2992)

    def test_noiseless_actual_FEC_roundtrip(self):
        tokens = np.random.default_rng(15).integers(0, 4096, 155)
        waveform = encode_digital(tokens, 991)
        result = receive_digital(waveform, 19)
        self.assertEqual(waveform.shape, (1950, 2))
        self.assertEqual(np.sum(waveform ** 2), 3900)
        self.assertTrue(result["header_usable"] and result["crc_accepted"])
        self.assertEqual(result["label"], 991)
        np.testing.assert_array_equal(result["tokens"], tokens)
        self.assertEqual(result["received_distance"], 0)

    def test_failed_body_keeps_actual_candidate(self):
        tokens = np.random.default_rng(16).integers(0, 4096, 155)
        waveform = encode_digital(tokens, 4)
        waveform[68:] += np.random.default_rng(17).normal(0, 4, waveform[68:].shape)
        result = receive_digital(waveform, 1)
        self.assertTrue(result["header_usable"])
        self.assertFalse(result["crc_accepted"])
        self.assertEqual(len(result["prefix"]), 7)
        self.assertEqual(result["tokens"].shape, (155,))
        self.assertGreater(result["received_distance"], 0)
        self.assertEqual(receiver_features(result, 1).shape, (3,))

    def test_unusable_mode_has_no_free_class(self):
        waveform = encode_digital(np.zeros(155, dtype=int), 23)
        waveform[:68] = encode_packet(header_bits(23, 8), 68)["symbols"]
        result = receive_digital(waveform, 19)
        self.assertFalse(result["header_usable"])
        self.assertIsNone(result["label"])
        self.assertIsNone(result["tokens"])

    def test_channel_energy_and_gradient(self):
        torch.manual_seed(18)
        value = torch.randn(3, 2220, requires_grad=True)
        waveform = normalize_analog(value)
        torch.testing.assert_close(waveform.square().sum(1), torch.full((3,), 2220.0), rtol=1e-6, atol=0.001)
        waveform[:, :5].sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(value.grad.norm()), 0)
        torch.testing.assert_close(normalize_analog(torch.zeros(1, 2220)).square().sum(), torch.tensor(2220.0))

    def test_equal_gain_parameter_counts_and_header_gray(self):
        specification = json.loads((ROOT / "configs/hybrid_source_correction.json").read_text())["model"]
        torch.manual_seed(19)
        blind = ResidualLink(specification, "snr_gain")
        torch.manual_seed(19)
        aware = ResidualLink(specification, "reliability_gain")
        self.assertEqual(sum(value.numel() for value in blind.parameters()), sum(value.numel() for value in aware.parameters()))
        for name, value in blind.state_dict().items():
            torch.testing.assert_close(value, aware.state_dict()[name], rtol=0, atol=0)
        base = torch.rand(2, 3, 8, 8)
        correction = torch.randn_like(base) * 0.01
        features = torch.tensor([[0.05, 1., 0.7], [0.35, 0., 0.8]])
        usable = torch.tensor([True, False])
        image, gain = aware.fuse(base, correction, features, usable)
        torch.testing.assert_close(gain, torch.ones_like(gain), rtol=0, atol=0)
        torch.testing.assert_close(image[0], (base + correction).clamp(0, 1)[0])
        torch.testing.assert_close(image[1], torch.full_like(image[1], 0.5), rtol=0, atol=0)
        with torch.no_grad():
            blind.gain[-1].weight.fill_(0.1)
        first = blind.fuse(base, correction, features, usable)[0]
        features[:, 1:] += 100
        second = blind.fuse(base, correction, features, usable)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
