#!/usr/bin/env python3
"""Offline success accounting must not turn a useful failed candidate into reliable recovery."""

from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from evaluate_communication_modes import compare_source, header_fields_correct
from var_comm.next_scale_prior import PATCH_NUMS


class MatrixAccountingTests(unittest.TestCase):
    def setUp(self):
        self.source = [np.zeros(size ** 2, dtype=np.int64) for size in PATCH_NUMS]
        self.target = {"class_index": 5}

    def test_success_has_correct_prefix_and_protocol(self):
        correct, reliable, error = compare_source(self.source[:7], 5, 7, self.source, self.target, 7, True, True, True)
        self.assertTrue(correct)
        self.assertTrue(reliable)
        self.assertEqual(error, 0.)

    def test_correct_failed_candidate_is_not_reliable(self):
        correct, reliable, error = compare_source(self.source[:7], 5, 7, self.source, self.target, 7, True, True, False)
        self.assertTrue(correct)
        self.assertFalse(reliable)
        self.assertEqual(error, 0.)

    def test_wrong_label_or_mode_is_not_source_success(self):
        for label, mode in ((6, 7), (5, 8)):
            correct, reliable, error = compare_source(self.source[:7], label, mode, self.source, self.target, 7, True, True, True)
            self.assertFalse(correct)
            self.assertFalse(reliable)

    def test_incomplete_prefix_has_no_fabricated_BER(self):
        correct, reliable, error = compare_source(self.source[:6], 5, 7, self.source, self.target, 7, False, True, False)
        self.assertFalse(correct)
        self.assertFalse(reliable)
        self.assertEqual(error, "")

    def test_arithmetic_header_length_is_part_of_header_integrity(self):
        header = {"accepted": True, "label": 5, "mode": 8, "length_field": 2300}
        self.assertTrue(header_fields_correct("arithmetic", header, 5, 8, 2300))
        self.assertFalse(header_fields_correct("arithmetic", header, 5, 8, 2301))
        self.assertTrue(header_fields_correct("raw", header, 5, 8))
        header["accepted"] = False
        self.assertFalse(header_fields_correct("raw", header, 5, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
