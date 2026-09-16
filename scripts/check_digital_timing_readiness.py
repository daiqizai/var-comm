#!/usr/bin/env python3
"""CPU regression checks for reference timing-input validation."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

import numpy as np

from audit_digital_timing_readiness import (
    PATCH_SIZES, array_digest, bound_file, check_frame, check_source_image,
    index_rows, require_matrix, seeded_noise, sha256,
)


def example_frame():
    signal = np.ones((3060, 2), dtype=np.float64)
    received = signal + seeded_noise("development-example", 2001, signal.shape) / np.sqrt(10 ** 0.1)
    digest = array_digest(received)
    row = {"image_index": 0, "image_id": "development-example", "snr_db": 1.0, "seed": 2001,
           "arm": "whole_adaptive", "complex_uses": 3060, "received_sha256": digest,
           "reconstruction_index": 0, "header_accepted": 1, "last_accepted_scale": 7,
           "header_false_acceptance": 0, "source_crc_false_acceptance_count": 0}
    receiver = {"arm": "whole_adaptive", "snr_db": 1.0, "seed": 2001, "received_sha256": digest,
                "reconstruction_index": 0, "header": {"accepted": True, "mode": 7, "label": 5},
                "label": 5, "output_prefix": [[0] * (size ** 2) for size in PATCH_SIZES[:7]],
                "events": [{"accepted": True}], "last_accepted_scale": 7}
    return row, receiver, signal, np.full((3, 256, 256), 0.75, dtype=np.float32)


class TimingReadinessChecks(unittest.TestCase):
    def test_complete_matrix_and_duplicates(self):
        row, unused_receiver, unused_signal, unused_image = example_frame()
        require_matrix([row], [0], [1.0], 2001, ["whole_adaptive"])
        with self.assertRaises(ValueError):
            index_rows([row, copy.deepcopy(row)])
        with self.assertRaises(ValueError):
            require_matrix([row], [0, 1], [1.0], 2001, ["whole_adaptive"])

    def test_original_observation_and_energy(self):
        row, receiver, signal, image = example_frame()
        result = check_frame(row, receiver, signal, image)
        self.assertEqual(result["total_energy"], 6120.0)
        self.assertEqual(result["transmitted_mode"], 7)
        signal[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "observation"):
            check_frame(row, receiver, signal, image)
        signal[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "QPSK"):
            check_frame(row, receiver, signal, image)

    def test_noise_and_dtype_cannot_change(self):
        row, receiver, signal, image = example_frame()
        with self.assertRaisesRegex(ValueError, "dtype"):
            check_frame(row, receiver, signal.astype(np.float32), image)
        row["seed"] = 2002
        with self.assertRaisesRegex(ValueError, "observation"):
            check_frame(row, receiver, signal, image)

    def test_crc_failure_retains_candidate_without_trust(self):
        row, receiver, signal, image = example_frame()
        receiver["events"][0]["accepted"] = False
        receiver["last_accepted_scale"] = row["last_accepted_scale"] = 0
        result = check_frame(row, receiver, signal, image)
        self.assertEqual(result["failure_stratum"], "body_crc")
        receiver["last_accepted_scale"] = 7
        with self.assertRaisesRegex(ValueError, "reliable"):
            check_frame(row, receiver, signal, image)

    def test_header_failure_is_not_removed_or_given_a_class(self):
        row, receiver, signal, image = example_frame()
        row["header_accepted"] = receiver["header"]["accepted"] = False
        receiver["label"] = None
        receiver["output_prefix"] = []
        receiver["events"] = []
        row["last_accepted_scale"] = receiver["last_accepted_scale"] = 0
        with self.assertRaisesRegex(ValueError, "gray"):
            check_frame(row, receiver, signal, image)
        image.fill(0.5)
        self.assertEqual(check_frame(row, receiver, signal, image)["failure_stratum"], "header")
        receiver["label"] = 5
        with self.assertRaisesRegex(ValueError, "free information"):
            check_frame(row, receiver, signal, image)

    def test_unchecked_header_false_acceptance_is_retained(self):
        row, receiver, signal, image = example_frame()
        row["header_false_acceptance"] = 1
        receiver["header"]["label"] = receiver["label"] = 6
        self.assertEqual(check_frame(row, receiver, signal, image)["header_false_acceptance"], 1)

    def test_receiver_tokens_and_class_must_be_legal(self):
        row, receiver, signal, image = example_frame()
        receiver["output_prefix"][0][0] = 4096
        with self.assertRaisesRegex(ValueError, "token"):
            check_frame(row, receiver, signal, image)
        receiver["output_prefix"][0][0] = 0
        receiver["label"] = 7
        with self.assertRaisesRegex(ValueError, "header"):
            check_frame(row, receiver, signal, image)

    def test_source_pixels_are_bound_to_existing_preprocessing(self):
        pixels = np.full((3, 256, 256), 128, dtype=np.uint8)
        source = pixels.astype(np.float32) / 255
        check_source_image(source, array_digest(pixels))
        source[0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "preprocessed"):
            check_source_image(source, array_digest(pixels))

    def test_receipt_rejects_tampering_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "reference.txt"
            artifact.write_text("original")
            receipt = {"output_hashes": {"reference.txt": sha256(artifact)}}
            inspected = {}
            bound_file(root, receipt, "reference.txt", inspected)
            self.assertEqual(len(inspected), 1)
            artifact.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                bound_file(root, receipt, "reference.txt", inspected)
            with self.assertRaisesRegex(ValueError, "escaped"):
                bound_file(root, receipt, "../outside.txt", inspected)

    def test_no_model_framework_import(self):
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
