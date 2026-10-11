"""No-science tests for crash replay and finite timing admission."""
import tempfile
from pathlib import Path
import unittest
from timing_contract import CaseReservations, fixed_cases


class TestTimingReservations(unittest.TestCase):
    def test_crashed_reserved_case_blocks_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'journal.json'
            ledger = CaseReservations(path); ledger.begin(fixed_cases()[0])
            with self.assertRaises(RuntimeError):
                CaseReservations(path)
            with self.assertRaises(AssertionError):
                ledger.begin(fixed_cases()[0])

    def test_actual_failed_decode_is_charged_and_third_decode_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = CaseReservations(Path(temporary) / 'journal.json')
            ledger.begin(fixed_cases()[0])
            def fail():
                raise ValueError('synthetic callback failure, no physical decoder')
            with self.assertRaises(ValueError):
                ledger.packet_call(fail)
            self.assertEqual(ledger.current['actual_packet_decodes'], 1)
            self.assertEqual(ledger.packet_call(lambda: 'synthetic result'), 'synthetic result')
            with self.assertRaises(AssertionError):
                ledger.packet_call(lambda: None)

    def test_cached_source_cannot_be_completed_as_fresh_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = CaseReservations(Path(temporary) / 'journal.json')
            ledger.begin(fixed_cases()[0])
            with self.assertRaises(AssertionError):
                ledger.finish(dict(fresh_visual_encoder=True, cached_tokens_or_streams=True), {})


if __name__ == '__main__':
    unittest.main()
