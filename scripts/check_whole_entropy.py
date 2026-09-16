#!/usr/bin/env python3
"""CPU reference roundtrips, one-block PHY budgets, overflow and failure-output invariants."""

from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from var_comm.entropy import arithmetic_decode, arithmetic_encode, probability_cdf
from var_comm.next_scale_prior import PATCH_NUMS
from var_comm.scale_channel import indices_to_bits
from var_comm.whole_entropy import ArithmeticDecoder, ArithmeticEncoder, choose_payload, decode_phy, decode_source, payload_key, transmit


class WholeEntropyTests(unittest.TestCase):
    def test_streaming_matches_existing_complete_coder(self):
        generator = np.random.default_rng(731)
        scores = generator.normal(size=(31, 4096))
        tables = probability_cdf(scores)
        tokens = generator.integers(0, 4096, 31)
        encoder = ArithmeticEncoder()
        encoder.encode(tokens[:7], tables[:7])
        prefix = encoder.finish()
        np.testing.assert_array_equal(prefix, arithmetic_encode(tokens[:7], tables[:7]))
        encoder.encode(tokens[7:], tables[7:])
        bits = encoder.finish()
        np.testing.assert_array_equal(bits, arithmetic_encode(tokens, tables))
        np.testing.assert_array_equal(arithmetic_decode(bits, tables), tokens)
        decoder = ArithmeticDecoder(bits)
        actual = np.concatenate((decoder.decode(tables[:7]), decoder.decode(tables[7:])))
        np.testing.assert_array_equal(actual, tokens)

    def test_finish_does_not_change_shared_prefix_stream(self):
        tables = probability_cdf(np.zeros((4, 4096)))
        encoder = ArithmeticEncoder()
        encoder.encode([2, 7], tables[:2])
        first = encoder.finish()
        np.testing.assert_array_equal(first, encoder.finish())
        encoder.encode([5, 9], tables[2:])
        np.testing.assert_array_equal(encoder.finish(), arithmetic_encode([2, 7, 5, 9], tables))

    def test_expansion_has_paid_raw_fallback(self):
        raw = np.zeros(1860, dtype=np.uint8)
        payload, length = choose_payload(raw, np.ones(1900, dtype=np.uint8))
        self.assertEqual(length, 0)
        np.testing.assert_array_equal(payload, raw)
        signal, ledger = transmit({"payload": payload, "length_field": length, "attempted_arithmetic_bits": 1900}, 23, 7)
        self.assertEqual(signal.shape, (3060, 2))
        self.assertEqual(ledger["total_energy"], 6120)
        self.assertEqual(ledger["header_uses"], 94)
        self.assertEqual(ledger["data_coded_bits"], 5932)
        self.assertEqual(ledger["data_mother_coded_bits"], 2 * (1860 + 22))

    def test_raw_and_compressed_payload_noiseless_PHY(self):
        generator = np.random.default_rng(3)
        for mode in (7, 8, 9):
            raw = generator.integers(0, 2, 12 * sum(size ** 2 for size in PATCH_NUMS[:mode]), dtype=np.uint8)
            for bits, length in ((raw, 0), (raw[:401], 401)):
                signal, ledger = transmit({"payload": bits, "length_field": length, "attempted_arithmetic_bits": 401}, 999, mode)
                decoded = decode_phy(signal, 19.)
                self.assertTrue(decoded["header"]["accepted"])
                self.assertTrue(decoded["body_crc_accepted"])
                self.assertEqual((decoded["label"], decoded["mode"], decoded["header"]["length_field"]), (999, mode, length))
                np.testing.assert_array_equal(decoded["payload"], bits)

    def test_header_erasure_has_no_free_class(self):
        result = decode_source({"label": None}, None, None, "cpu")
        self.assertFalse(result["source_complete"])
        self.assertEqual(result["prefix"], [])
        self.assertTrue(np.all(result["image"] == .5))

    def test_failed_CRC_does_not_erase_legal_raw_candidate(self):
        count = sum(size ** 2 for size in PATCH_NUMS[:7])
        phy = {"label": 5, "mode": 7, "header": {"length_field": 0},
               "payload": indices_to_bits(np.arange(count)), "body_crc_accepted": False}
        result = decode_source(phy, None, None, "cpu", render=False)
        self.assertTrue(result["source_complete"])
        np.testing.assert_array_equal(np.concatenate(result["prefix"]), np.arange(count))
        checksum_key = payload_key(phy)
        phy["body_crc_accepted"] = True
        self.assertEqual(payload_key(phy), checksum_key)
        phy["payload"][0] ^= 1
        self.assertNotEqual(payload_key(phy), checksum_key)

    def test_probability_floor_keeps_every_symbol_possible(self):
        scores = np.full((1, 4096), -1000.)
        scores[0, 7] = 0
        cdf = probability_cdf(scores)
        self.assertTrue(np.all(np.diff(cdf[0]) >= 1))
        bits = arithmetic_encode([4095], cdf)
        self.assertEqual(int(ArithmeticDecoder(bits).decode(cdf)[0]), 4095)

    def test_corrupt_stream_remains_a_bounded_candidate(self):
        tables = probability_cdf(np.zeros((155, 4096)))
        bits = np.ones(17, dtype=np.uint8)
        decoded = ArithmeticDecoder(bits).decode(tables)
        self.assertEqual(decoded.shape, (155,))
        self.assertTrue(np.all((decoded >= 0) & (decoded < 4096)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
