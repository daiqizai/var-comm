#!/usr/bin/env python3
"""CPU checks for matched initialization, legal context, Adam isolation, and stopping."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from var_comm.base_conditioning import BaseConditionedLink, restore_parent, select_checkpoints, stopping_decision
from var_comm.hybrid_correction import ResidualLink


class ConditioningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/hybrid_base_conditioning.json").read_text())
        cls.checkpoint = torch.load(ROOT / cls.config["initializer"], map_location="cpu", weights_only=False)

    def models(self):
        models, optimizers = {}, {}
        for arm in self.config["arms"]:
            torch.manual_seed(self.config["training"]["initialization_seed"])
            model = BaseConditionedLink(self.config["model"], arm)
            optimizer, _names = restore_parent(model, self.checkpoint, self.config)
            models[arm], optimizers[arm] = model, optimizer
        return models, optimizers

    def test_exact_counts_and_shared_states(self):
        models, _optimizers = self.models()
        self.assertEqual(sum(value.numel() for value in models["unconditioned"].parameters()), 1648125)
        self.assertEqual(sum(value.numel() for value in models["conditioned"].parameters()), 1648125)
        for name, value in models["unconditioned"].state_dict().items():
            torch.testing.assert_close(value, models["conditioned"].state_dict()[name], rtol=0, atol=0)
        self.assertFalse(any("gain" in name for name, _value in models["conditioned"].named_parameters()))

    def test_identity_function_with_real_parent(self):
        models, _optimizers = self.models()
        parent = ResidualLink(self.config["model"], "add").eval()
        parent.load_state_dict(self.checkpoint["models"]["add"])
        torch.manual_seed(82)
        observations, snr, base = torch.randn(1, 2220), torch.tensor([7.]), torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            expected = parent.decode(observations, snr)
            for model in models.values():
                torch.testing.assert_close(model.decode(observations, snr, base), expected, rtol=0, atol=1e-6)

    def test_control_cannot_read_base_but_condition_can(self):
        models, _optimizers = self.models()
        observations, snr = torch.randn(1, 2220), torch.tensor([13.])
        first, second = torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            for model in models.values():
                for projection in model.context_projections:
                    projection.weight.fill_(.02)
            control = models["unconditioned"]
            torch.testing.assert_close(control.decode(observations, snr, first), control.decode(observations, snr, second), rtol=0, atol=0)
            condition = models["conditioned"]
            self.assertGreater(float((condition.decode(observations, snr, first) - condition.decode(observations, snr, second)).abs().max()), 1e-6)
            with self.assertRaises(ValueError):
                condition.decode(observations, snr)

    def test_feature_weights_are_not_a_dead_control_branch(self):
        models, _optimizers = self.models()
        for model in models.values():
            with torch.no_grad():
                for projection in model.context_projections:
                    projection.weight.fill_(.01)
            observed = torch.randn(1, 2220, requires_grad=True)
            base = torch.rand(1, 3, 256, 256, requires_grad=True)
            output = model.decode(observed, torch.tensor([7.]), base)
            output.square().mean().backward()
            self.assertGreater(float(observed.grad.abs().sum()), 0)
            self.assertTrue(all(parameter.grad is not None for parameter in model.context_extractor.parameters()))
            self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in model.context_extractor.parameters()), 0)
            if model.variant == "unconditioned":
                self.assertIsNone(base.grad)
            else:
                self.assertGreater(float(base.grad.abs().sum()), 0)

    def test_adam_restored_but_not_shared(self):
        _models, optimizers = self.models()
        first = next(iter(optimizers["unconditioned"].state.values()))
        second = next(iter(optimizers["conditioned"].state.values()))
        self.assertEqual(int(first["step"]), 10000)
        self.assertNotEqual(first["exp_avg"].data_ptr(), second["exp_avg"].data_ptr())
        snapshot = second["exp_avg"].clone()
        first["exp_avg"].add_(1)
        torch.testing.assert_close(snapshot, second["exp_avg"], rtol=0, atol=0)
        for optimizer in optimizers.values():
            self.assertTrue(all(parameter not in optimizer.state for parameter in optimizer.param_groups[1]["params"]))

    def test_stop_is_paired_and_capped(self):
        summary = {arm: {"mechanism": {"psnr_db": 21., "lpips": .2}} for arm in self.config["arms"]}
        records = {str(step): copy.deepcopy(summary) for step in (0, 5000, 10000)}
        self.assertTrue(stopping_decision(records, 10000, self.config)["stop"])
        records["10000"]["unconditioned"]["mechanism"]["lpips"] = .19
        self.assertFalse(stopping_decision(records, 10000, self.config)["stop"])
        records["20000"] = copy.deepcopy(records["10000"])
        self.assertTrue(stopping_decision(records, 20000, self.config)["stop"])

    def test_selection_does_not_use_between_arm_gap_or_fallback_to_parent(self):
        summary = {arm: {"mechanism": {"psnr_db": 21., "lpips": .2}} for arm in self.config["arms"]}
        records = {str(step): copy.deepcopy(summary) for step in (0, 5000, 10000)}
        records["5000"]["conditioned"]["mechanism"] = {"psnr_db": 20., "lpips": .10}
        records["10000"]["conditioned"]["mechanism"] = {"psnr_db": 21.1, "lpips": .18}
        choice = select_checkpoints(records, self.config)
        self.assertEqual(choice["conditioned"]["step"], 10000)
        records["10000"]["conditioned"]["mechanism"]["psnr_db"] = 20.5
        choice = select_checkpoints(records, self.config)
        self.assertFalse(choice["conditioned"]["selection_guard_passed"])
        self.assertNotEqual(choice["conditioned"]["step"], 0)

    def test_spatial_ablation_preserves_values(self):
        model = BaseConditionedLink(self.config["model"], "conditioned")
        value = torch.arange(3 * 64 * 64).float().reshape(1, 3, 64, 64)
        permuted = model.spatial_permutation(value)
        torch.testing.assert_close(value.flatten().sort().values, permuted.flatten().sort().values)
        self.assertFalse(torch.equal(value, permuted))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
