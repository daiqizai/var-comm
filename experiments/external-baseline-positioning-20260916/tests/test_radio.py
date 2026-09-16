import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from external_positioning.radio import decode_metadata, encode_metadata, metadata_layout, rank_subset, unrank_subset


class MetadataRadioTests(unittest.TestCase):
    def test_compact_subset_bijection(self):
        seen = set()
        for indices in itertools.combinations(range(8), 3):
            rank = rank_subset(indices, 8)
            seen.add(rank)
            self.assertEqual(unrank_subset(rank, 8, 3), list(indices))
        self.assertEqual(seen, set(range(56)))

    def test_noiseless_power_and_actual_mask_roundtrip(self):
        generator = np.random.default_rng(901)
        for count in (32, 64, 96):
            indices = sorted(generator.choice(320, count, replace=False).tolist())
            power = np.float32(.123456)
            packet = encode_metadata(power, indices, 320)
            decoded = decode_metadata(packet["symbols"], 7, 320, count)
            self.assertTrue(decoded["crc_accepted"])
            self.assertTrue(decoded["usable"])
            self.assertEqual(decoded["indices"], indices)
            self.assertEqual(decoded["power"], power)
            self.assertEqual(float(np.square(packet["symbols"]).sum()), 2 * packet["complex_uses"])

    def test_only_required_side_information_is_charged(self):
        self.assertEqual(metadata_layout()["complex_uses"], 108)
        self.assertEqual(metadata_layout(320, 32)["complex_uses"], 402)
        self.assertEqual(metadata_layout(320, 64)["complex_uses"], 562)
        self.assertEqual(metadata_layout(320, 96)["complex_uses"], 664)

    def test_illegal_subset_does_not_receive_source_assistance(self):
        with self.assertRaises(ValueError):
            unrank_subset(56, 8, 3)


if __name__ == "__main__":
    unittest.main()
