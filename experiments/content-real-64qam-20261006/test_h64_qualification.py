from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from h64_catalog import all_buckets
from h64_qualification import mathematical_qualification,physical_qualification,wilson
from test_h64_core import FakeBackend,RecordingLedger
import numpy as np


class PureHeader:
    def transmit(self,pid): return np.full((68,2),float(pid))
    def receive(self,y,snr,book):
        pid=int(y[0,0]);ok=str(pid) in book
        return dict(profile_id=pid if ok else None,header_ok=ok,header_crc_ok=True,header_fields_legal=ok)


class QualificationTests(unittest.TestCase):
    def test_fixed_100k_uncoded_ser_qualification(self):
        rows=mathematical_qualification(100000)
        self.assertEqual(len(rows),6)
        for row in rows:
            self.assertEqual(row['result'],'PASS')
            self.assertLessEqual(abs(row['observed_SER']-row['exact_SER']),row['absolute_tolerance'])
    def test_real_packet_hooks_are_all_metered_and_weak_link_is_not_failure(self):
        buckets=all_buckets()
        for b in buckets:
            b.update(admission='ADMITTED',layout={'layout_id':'TEST_ONLY'})
        ledger=RecordingLedger()
        result=physical_qualification(FakeBackend(),PureHeader(),ledger,buckets,
                                      noiseless_per_layout=2,blocks_per_point=2)
        self.assertEqual(result['actual_decode_calls'],8+16+48)
        self.assertEqual(len(ledger.events),result['actual_decode_calls'])
        self.assertEqual(sum(e['kind']=='header' for e in ledger.events),8)
        self.assertFalse(result['source_model_executed'])
        self.assertTrue(all(r['weak_performance_is_implementation_failure'] is False for r in result['coded_curve']))
    def test_wilson_zero_errors_does_not_claim_perfect_probability(self):
        low,high=wilson(64,64)
        self.assertLess(low,1);self.assertAlmostEqual(high,1)
        low,high=wilson(0,64)
        self.assertGreater(high,0);self.assertAlmostEqual(low,0)
    def test_boundary_precedes_every_metered_packet(self):
        ledger=RecordingLedger();calls=[]
        def stop():
            calls.append(1)
            raise RuntimeError('STOP')
        with self.assertRaisesRegex(RuntimeError,'STOP'):
            physical_qualification(FakeBackend(),PureHeader(),ledger,all_buckets(),stop)
        self.assertEqual(len(ledger.events),0);self.assertEqual(calls,[1])


if __name__=='__main__':unittest.main()
