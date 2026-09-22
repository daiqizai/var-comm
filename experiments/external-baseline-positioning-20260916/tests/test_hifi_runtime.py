import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "scripts"), str(EXPERIMENT / "performance_branch")]
from prepare_hifi_stage20 import account_frame
from instrumentation import AttentionExecution


class HiFiAccountingTests(unittest.TestCase):
    def record(self, usable=True, reused=True):
        common = {"method": "hifi_diffcom_c2", "protocol": "common_paid_information", "image_index": 0,
                  "snr_db": 1., "seed": 2001, "metadata_usable": usable, "NFE": 251 if usable else 0,
                  "RX_seconds": 85. if usable else .02, "TX_seconds": .01, "metadata_crc_accepted": usable,
                  "offline_metadata_exact": reused, "fallback": "" if usable else "invalid_metadata_same_observation_ADJSCC"}
        native = {**common, "protocol": "author_assumed_information", "NFE": 251, "same_receive_inputs_reused": reused}
        base = {**common, "method": "adjscc_c2", "NFE": 0, "RX_seconds": .02}
        return {"frame_key": "example", "rows": [common, base, native]}

    def test_three_views_are_one_sampler(self):
        row = account_frame(self.record())
        self.assertEqual(row["record_views"], 3)
        self.assertEqual(row["actual_sampler_calls"], 1)
        self.assertEqual(row["total_actual_reverse_steps"], 251)

    def test_different_metadata_adds_one_not_three(self):
        row = account_frame(self.record(reused=False))
        self.assertEqual(row["actual_sampler_calls"], 2)
        self.assertEqual(row["metadata_extra_author_sampler_calls"], 1)
        self.assertEqual(row["extra_author_RX_seconds"], "not_recorded")

    def test_common_fallback_still_has_one_native_sampler(self):
        row = account_frame(self.record(usable=False, reused=False))
        self.assertEqual(row["common_sampler_calls"], 0)
        self.assertEqual(row["actual_sampler_calls"], 1)


class AttentionBlock(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.]))
        self.checkpoint = checkpoint

    def _forward(self, values):
        return (values * self.weight).square()

    def forward(self, values):
        return self.checkpoint(self._forward, (values,), self.parameters(), True)


class IsolatedAttentionTests(unittest.TestCase):
    def test_original_custom_checkpoint_input_gradient_and_restoration(self):
        path = EXPERIMENT / "vendor/diffcom_code/guided_diffusion/nn.py"
        if not path.exists():
            self.skipTest("pinned vendor is required for the original custom checkpoint test")
        spec = importlib.util.spec_from_file_location("original_checkpoint_nn", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        block = AttentionBlock(module.checkpoint)
        model = types.SimpleNamespace(unet=torch.nn.Sequential(block))
        original_forward = block.forward
        values = torch.tensor([3.], requires_grad=True)
        expected = model.unet(values)
        expected_grad = torch.autograd.grad(expected.sum(), values)[0]
        with AttentionExecution(model, "attention_direct"):
            direct = model.unet(values)
            actual_grad = torch.autograd.grad(direct.sum(), values)[0]
        self.assertTrue(torch.equal(expected, direct))
        self.assertTrue(torch.equal(expected_grad, actual_grad))
        self.assertEqual(block.forward, original_forward)
        self.assertIsNone(block.weight.grad)
        self.assertTrue(block.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
