"""Focused CPU tests; run: python -m unittest hifi_swin_tests -v."""
import dataclasses
import math
import unittest

import torch
from torch import nn

from hifi_swin_operator import (DecodedSwinContext, SwinHiFiOperator,
    decode_with_image_gradient, encode_before_mask, qualify_native_parity,
    strict_decoded_context)


def native_gate(x, value, sm, bm, sigmoid):
    condition = x.new_tensor(float(value)).reshape(1, 1)
    hidden = sm[0](x.detach()) * bm[0](condition).unsqueeze(1)
    modulation = sigmoid(sm[-1](hidden))
    return x * modulation, modulation


class Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 8)

    def forward(self, x):
        return self.proj(x.permute(0, 2, 3, 1).reshape(1, 4, 2))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = Patch()
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh())])
        self.norm = nn.Identity()
        self.layer_num = 1
        self.sm_list1 = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.bm_list1 = nn.ModuleList([nn.Sequential(nn.Linear(1, 8), nn.Sigmoid())])
        self.sm_list = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.bm_list = nn.ModuleList([nn.Sequential(nn.Linear(1, 8), nn.Sigmoid())])
        self.sigmoid = self.sigmoid1 = nn.Sigmoid()

    def forward(self, image, snr, rate, model):
        x = self.patch_embed(image)
        for layer in self.layers:
            x = layer(x)
        x, _ = native_gate(x, snr, self.sm_list1, self.bm_list1, self.sigmoid1)
        x, mod = native_gate(x, rate, self.sm_list, self.bm_list, self.sigmoid)
        mask = torch.zeros_like(x[:, 0])
        mask.scatter_(1, mod.sum(1).sort(1, descending=True).indices[:, :rate], 1)
        mask = mask[:, None].expand_as(x)
        return x * mask, mask


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_num = 1
        self.sm_list = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.bm_list = nn.ModuleList([nn.Sequential(nn.Linear(1, 8), nn.Sigmoid())])
        self.sigmoid = nn.Sigmoid()
        self.layers = nn.ModuleList([nn.Linear(8, 3)])
        self.H = self.W = 2

    def forward(self, features, snr, model):
        x, _ = native_gate(features, snr, self.sm_list, self.bm_list, self.sigmoid)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(1, 2, 2, 3).permute(0, 3, 1, 2)


class TinySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = Encoder(), Decoder()
        self.model = "SwinJSCC_w/_SAandRA"


class HiFiSwinTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314)
        self.model = TinySwin().double().eval()
        self.context = DecodedSwinContext(36, 32, 2, (1, 6), .7, 7., True, 4, 8)
        self.operator = SwinHiFiOperator(self.model, self.context)
        self.image = torch.linspace(.1, .8, 8, dtype=torch.float64).reshape(1, 2, 2, 2)

    def test_registered_resource_ledgers(self):
        for n, c, h, mask_bits in [(1024, 6, 256, 41), (2048, 13, 384, 76)]:
            ctx = strict_decoded_context(n, 7, {"indices": list(range(c)), "power": .5}, crc_accepted=True)
            self.assertEqual(ctx.data_n + h, n)
            self.assertEqual(ctx.mask_bits, mask_bits)
            self.assertEqual(ctx.ledger()["E_total"], 2 * n)
            self.assertEqual(ctx.ledger()["header_coded_bits"], 2 * h)
            self.assertEqual(ctx.ledger()["project_noise_variance_per_real"],
                             2 * ctx.ledger()["author_noise_variance_per_real"])

    def test_invalid_headers_never_reach_posterior(self):
        for changes in [{"crc_accepted": False}, {"indices": (1, 1)},
                        {"indices": (6, 1)}, {"power": float("nan")}, {"power": 0},
                        {"total_n": 37}, {"indices": (1, 8)}]:
            with self.assertRaises(ValueError):
                dataclasses.replace(self.context, **changes)

    def test_native_forward_values_and_mask_are_unchanged(self):
        result = qualify_native_parity(self.model, self.image, 7, 2)
        self.assertTrue(result["mask_exact"])
        self.assertEqual(result["encoder_max_abs"], 0)
        self.assertEqual(result["decoder_max_abs"], 0)

    def test_fixed_mask_survives_other_candidate_channel_preferences(self):
        with torch.no_grad():
            head = self.model.encoder.sm_list[-1]
            head.weight.zero_()
            head.bias.fill_(-2)
            head.bias[[0, 7]] = 2
        features, _ = encode_before_mask(self.model.encoder, self.image, 7, 2)
        expected = torch.stack((features[0, :2, [1, 6]].reshape(-1),
                                features[0, 2:, [1, 6]].reshape(-1)), -1).reshape(1, -1)
        actual = self.operator.encode(self.image)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        native, native_mask = self.model.encoder(self.image, 7, 2, self.model.model)
        self.assertFalse(torch.equal(native_mask[0, 0].nonzero().flatten(), torch.tensor([1, 6])))
        self.assertGreater(float(actual.abs().sum()), 0)

    def test_paid_gain_is_fixed_and_not_renormalized_per_candidate(self):
        raw = torch.arange(1, 9, dtype=torch.float64).reshape(1, -1)
        waveform = self.operator.forward(raw)
        torch.testing.assert_close(waveform, raw / math.sqrt(1.4))
        torch.testing.assert_close(self.operator.forward(2 * raw), 2 * waveform)
        torch.testing.assert_close(self.operator.transpose(waveform), raw)

    def test_observed_energy_and_iq_order_are_converted_once(self):
        observed = torch.arange(1, 9, dtype=torch.float64).reshape(4, 2)
        saved = observed.clone()
        measurement = self.operator.measurement(observed)
        torch.testing.assert_close(measurement["ofdm_sig"].square().sum(), observed.square().sum() / 2)
        torch.testing.assert_close(measurement["s_hat"], observed.reshape(1, -1) * math.sqrt(.7))
        torch.testing.assert_close(observed, saved, atol=0, rtol=0)
        self.assertFalse(measurement["x_mse"].requires_grad)
        with self.assertRaises(ValueError):
            self.operator.measurement(torch.zeros(36, 2))

    def test_joint_image_gradient_matches_finite_difference(self):
        def objective(image):
            latent = self.operator.encode(image)
            wave = self.operator.forward(latent)
            confirming = self.operator.decode(self.operator.transpose(wave))
            return (wave - .19).square().sum() + (confirming - .31).square().sum()
        image = self.image.clone().requires_grad_()
        direction = torch.linspace(-.8, .9, 8, dtype=image.dtype).reshape_as(image)
        derivative = (torch.autograd.grad(objective(image), image)[0] * direction).sum()
        epsilon = 1e-5
        numerical = (objective(image.detach() + epsilon * direction) -
                     objective(image.detach() - epsilon * direction)) / (2 * epsilon)
        torch.testing.assert_close(derivative, numerical, rtol=1e-6, atol=1e-8)
        self.assertGreater(float(derivative.abs()), 1e-6)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_measurement_base_matches_registered_clamped_swin_receiver(self):
        observed = torch.linspace(-2, 2, 8, dtype=torch.float64).reshape(4, 2)
        # Independent channel-major unpacking, as in SwinCodec.receive_data.
        selected = torch.cat((observed[:, 0], observed[:, 1])).reshape(1, 4, 2) * math.sqrt(.7)
        full = observed.new_zeros((1, 4, 8))
        full[:, :, [1, 6]] = selected
        with torch.no_grad():
            reference = self.model.decoder(full, 7, self.model.model).clamp(0, 1)
        measurement = self.operator.measurement(observed)
        torch.testing.assert_close(measurement["x_mse"], reference, rtol=1e-12, atol=1e-12)
        self.assertTrue(bool((measurement["x_mse"] >= 0).all()))
        self.assertTrue(bool((measurement["x_mse"] <= 1).all()))

    def test_detached_native_gate_is_not_full_image_jacobian(self):
        image = self.image.clone().requires_grad_()
        native, mask = self.model.encoder(image, 7, 2, self.model.model)
        full, _ = encode_before_mask(self.model.encoder, image, 7, 2)
        native_grad = torch.autograd.grad(native.square().sum(), image)[0]
        full_grad = torch.autograd.grad((full * mask).square().sum(), image)[0]
        self.assertGreater(float((native_grad - full_grad).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
