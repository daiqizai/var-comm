import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wetok_comm.digital import WeTokDigital


class DigitalTests(unittest.TestCase):
    def test_native_payload_roundtrip_and_exact_energy(self):
        modem = WeTokDigital()
        random = np.random.default_rng(619)
        for codes in (np.zeros((16, 16, 4), dtype=np.uint8), random.integers(0, 256, (16, 16, 4), dtype=np.uint8)):
            symbols = modem.transmit(codes)
            self.assertEqual(symbols.shape, (3060, 2))
            np.testing.assert_allclose(np.square(symbols).sum(-1), 2, rtol=0, atol=1e-12)
            received = modem.receive(symbols, 19.)
            np.testing.assert_array_equal(received['indices'], codes)
            self.assertTrue(received['crc_accepted'])
        self.assertEqual(modem.ledger()['transmitted_coded_bits'], 9180)
        self.assertEqual(modem.ledger()['raw_source_bits'], 8192)

    def test_failed_candidate_is_not_repaired_from_truth(self):
        modem = WeTokDigital()
        waveform = np.random.default_rng(81).normal(size=(3060, 2))
        decoded = modem.receive(waveform, 1.)
        self.assertEqual(decoded['indices'].shape, (16, 16, 4))
        self.assertTrue(decoded['candidate_retained_for_image'])


if __name__ == '__main__':
    unittest.main()
