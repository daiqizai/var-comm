#!/usr/bin/env python3
"""Reuse audit and tests for non-interpolated matching and honest expectations."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np

from var_comm.base_conditioning import BaseConditionedLink, restore_parent
from var_comm.hybrid_training import training_draw
from var_comm.study import sha256, verify_snapshot, write_json
from var_comm.weight_closure import expected_metric, fit_time_sharing, load_configs, select_matched_controls, specifications


def audit_reuse(output):
    closure, base, runtime = load_configs()
    old = ROOT / closure["base_run"] / "training_001"
    metadata = json.loads((old / "metadata.json").read_text())
    verify_snapshot(metadata["bindings"])
    completion = json.loads((old / "completion.json").read_text())
    if completion["new_updates_per_arm"] != 20000 or completion["shared_parameter_updates"] != 30000:
        raise RuntimeError("0.01 did not actually receive the same planned new update count")
    initial = torch.load(old / "checkpoints/step_0000000.pt", map_location="cpu", weights_only=False)
    parent = torch.load(ROOT / base["initializer"], map_location="cpu", weights_only=False)
    for variant in closure["variants"]:
        torch.manual_seed(base["training"]["initialization_seed"])
        model = BaseConditionedLink(base["model"], variant, base["mechanism"]["context_ablation_seed"])
        optimizer, _names = restore_parent(model, parent, runtime)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, initial["models"][variant][name], rtol=0, atol=0)
        actual, expected = optimizer.state_dict(), initial["optimizers"][variant]
        if actual["param_groups"] != expected["param_groups"] or actual["state"].keys() != expected["state"].keys():
            raise RuntimeError("original optimizer initialization differs")
        for parameter, state in actual["state"].items():
            for field, value in state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, expected["state"][parameter][field], rtol=0, atol=0)
                elif value != expected["state"][parameter][field]:
                    raise RuntimeError("optimizer scalar initialization differs")
    sampled = {1, 2, 1000, 5000, 10000, 15000, 19999, 20000}
    steps = []
    for path in sorted(old.glob("updates_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                steps.append(row["new_step"])
                if row["new_step"] in sampled:
                    indices, snrs, noise = training_draw(row["new_step"], 20000, runtime)
                    if indices.tolist() != row["indices"] or snrs.tolist() != row["snrs_db"] or hashlib.sha256(noise.tobytes()).hexdigest() != row["noise_sha256"]:
                        raise RuntimeError("0.01 training data/SNR/noise differs from the new schedule")
    if steps != list(range(1, 20001)):
        raise RuntimeError("original update history incomplete or duplicated")
    for step in range(0, 20001, 1000):
        kind = "full" if step in closure["calibration_full_steps"] else "subset"
        directory = old / "calibration" / f"step_{step:07d}_{kind}"
        receipt = json.loads((directory / "completion.json").read_text())
        expected_sources = 1000 if kind == "full" else 100
        if receipt["sources"] != expected_sources or sha256(directory / "per_frame.csv") != receipt["per_frame_sha256"]:
            raise RuntimeError("original calibration schedule or bytes differ")
    output.mkdir(parents=True, exist_ok=True)
    record = {"status": "PASS", "reuse_lambda": .01, "new_updates_each": 20000, "shared_updates_each": 30000,
              "same_initial_model_and_Adam": True, "same_actual_training_and_calibration_schedule": True,
              "sampled_data_noise_replay_steps": sorted(sampled), "base_metadata_sha256": sha256(old / "metadata.json"),
              "original_completion_sha256": sha256(old / "completion.json"), "new_training_of_0p01": False,
              "new_checkpoint_selection_will_not_rewrite_old_report": True}
    write_json(output / "reuse_audit.json", record)
    print(json.dumps(record, indent=2))


class ClosureChecks(unittest.TestCase):
    def test_expectation_not_image_blending(self):
        actual = expected_metric(np.array([10., 20.]), np.array([30., 40.]), .25)
        np.testing.assert_array_equal(actual, [15., 25.])
        with self.assertRaises(ValueError):
            expected_metric([1.], [2.], 1.1)

    def test_no_interpolated_learned_match(self):
        closure, _base, _runtime = load_configs()
        calibration = {}
        for arm, item in specifications(closure).items():
            metrics = {"lpips": .10 if item["variant"] == "conditioned" else .20, "psnr_db": 22. if item["variant"] == "conditioned" else 20.}
            calibration[arm] = {region: copy.deepcopy(metrics) for region in ("primary", "mechanism")}
        self.assertTrue(all(not match["coverage"] for match in select_matched_controls(calibration, closure)))

    def test_time_share_proportions_fit_calibration(self):
        closure, _base, _runtime = load_configs()
        refs = {"raw_adaptive": {"psnr_db": 20., "lpips": .1, "mse": .01, "dino": .9},
                "arithmetic_adaptive": {"psnr_db": 20., "lpips": .1, "mse": .01, "dino": .9},
                "perceptual_deepjscc": {"psnr_db": 24., "lpips": .2, "mse": .004, "dino": .6}}
        calibration = {"example": {"primary": {"psnr_db": 22., "lpips": .15}}}
        reference = fit_time_sharing(calibration, refs, closure)
        self.assertAlmostEqual(reference[0]["p_deep"], .5)
        self.assertAlmostEqual(reference[1]["p_deep"], .5)
        self.assertAlmostEqual(reference[-1]["p_deep"], .05)
        self.assertTrue(all(not entry["actual_deployment_tested"] for entry in reference))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-output", type=Path)
    args, remainder = parser.parse_known_args()
    torch.set_num_threads(2)
    if args.audit_output:
        audit_reuse(args.audit_output)
    unittest.main(argv=[sys.argv[0], *remainder])
