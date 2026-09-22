import copy
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from latent_enhancement.latent import enhancement_noise
from latent_enhancement_b.common import decoder_gate_path, validate_gate
from latent_enhancement_b.model import ConditionalReceiver, build_arms, latent_errors, render_received


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.recipe = {"initialization_seed": 2026091713, "model_hidden_channels": 64, "model_residual_blocks": 3}
        self.shape = (32, 16, 16)
        self.scale = torch.ones(32)

    def test_receiver_has_no_truth_or_transmitter_normalization_input(self):
        self.assertEqual(list(inspect.signature(ConditionalReceiver.forward).parameters),
                         ["self", "observation", "base_rx", "snr_db", "rx_status"])

    def test_initial_function_common_parameters_and_exact_energy(self):
        arms = build_arms(self.shape, self.scale, self.recipe)
        base = torch.randn(2, *self.shape)
        truth = torch.randn_like(base)
        status = torch.tensor([[1., 1., 8.], [1., 0., 7.]])
        snrs = torch.tensor([1., 19.])
        noise = torch.randn(2, 1024, 2)
        for model in arms.values():
            latent, symbols = model.receive_training_sample(truth, base + 0.1, base, snrs, status, noise)
            self.assertTrue(torch.equal(latent, base))
            if model.uses:
                self.assertEqual(symbols.shape, (2, model.uses, 2))
                torch.testing.assert_close(symbols.square().sum((1, 2)), torch.full((2,), 2.0 * model.uses))
        self.assertTrue(torch.equal(arms['enhancement512'].receiver.base_stem.weight, arms['receiver_only_refiner'].receiver.base_stem.weight))
        self.assertTrue(torch.equal(arms['enhancement512'].encoder.stem.weight, arms['enhancement1024'].encoder.stem.weight))
        with self.assertRaises(ValueError):
            arms['receiver_only_refiner'].receiver(noise, base, snrs, status)

    def test_frozen_decoder_input_gradient_reaches_transmitter(self):
        model = build_arms(self.shape, self.scale, self.recipe)['enhancement512']
        decoder = torch.nn.Sequential(torch.nn.Conv2d(32, 3, 1), torch.nn.Sigmoid()).requires_grad_(False)
        before = copy.deepcopy(decoder.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
        base, truth = torch.randn(2, *self.shape), torch.randn(2, *self.shape)
        status, snrs = torch.tensor([[1., 0., 8.], [0., 0., 0.]]), torch.tensor([1., 4.])
        target = torch.rand(2, 3, 16, 16)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            latent, _ = model.receive_training_sample(truth, base, base, snrs, status, torch.randn(2, 1024, 2))
            image = render_received(decoder, latent, status)
            self.assertTrue(torch.equal(image[1], torch.full_like(image[1], 0.5)))
            auxiliary = latent_errors(latent, truth, self.scale, status)
            self.assertEqual(float(auxiliary[1].detach()), 0)
            ((image - target).square().mean() + 0.01 * auxiliary.mean()).backward()
            optimizer.step()
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in model.encoder.parameters() if parameter.grad is not None), 0)
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))
        for name in before:
            self.assertTrue(torch.equal(before[name], decoder.state_dict()[name]))

    def test_extra_noise_is_nested_and_reproducible(self):
        first = enhancement_noise('source', 771, 512)
        np.testing.assert_array_equal(first, enhancement_noise('source', 771, 1024)[:512])
        self.assertFalse(np.array_equal(first, enhancement_noise('source', 772, 512)))


class GateTests(unittest.TestCase):
    def test_repaired_gate_context_is_explicit_and_default_is_historical(self):
        historical = decoder_gate_path()
        with patch.dict(os.environ, {'VAR_COMM_DECODER_GATE': '/tmp/repair-gate.json'}):
            self.assertEqual(str(decoder_gate_path()), '/tmp/repair-gate.json')
        self.assertEqual(decoder_gate_path(), historical)

    def records(self):
        selection = {'step': 38000, 'checkpoint': 'selected.pt', 'checkpoint_sha256': 'abc', 'utility': 0.01}
        completion = {'status': 'STAGE_A_COMPLETE_NOT_FULL_EXPERIMENT', 'frozen_vae_state_unchanged': True,
                      'stage_B_eligible': True, 'selection': selection}
        return completion, {'stage_B_eligible': True, 'selection': selection}, selection

    def test_positive_final_selection(self):
        self.assertTrue(validate_gate(*self.records()))

    def test_negative_gate_never_launches(self):
        completion, result, selection = self.records()
        result['stage_B_eligible'] = False
        self.assertFalse(validate_gate(completion, result, selection))

    def test_running_or_mismatched_selection_is_rejected(self):
        completion, result, selection = self.records()
        completion['status'] = 'STAGE_A_TRAINING'
        with self.assertRaises(RuntimeError):
            validate_gate(completion, result, selection)
        completion, result, selection = self.records()
        result['selection'] = {**selection, 'step': 40000}
        with self.assertRaises(RuntimeError):
            validate_gate(completion, result, selection)


if __name__ == '__main__':
    unittest.main()
