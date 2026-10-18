import pathlib,unittest
from population import evaluate_sources,policies
import native
class Evaluation(unittest.TestCase):
    def sources(self):return [dict(rank=i,source_id=str(i),source_index=499-i) for i in range(100)]
    def test_registered_population_order(self):
        s=evaluate_sources(self.sources(),[dict(source_id='cal')]);self.assertEqual(s[0]['rank'],499);self.assertEqual(len(s),100)
    def test_calibration_overlap_rejected(self):
        with self.assertRaises(AssertionError):evaluate_sources(self.sources(),[dict(source_id='5')])
    def test_only_full_calibration_frozen_gate(self):
        r=dict(status='FROZEN_FROM_FULL_CAL1000',sources=1000,logical_frames=18000,policies={'4':'a','10':'b','19':'c'},A3_logical_cap=900)
        self.assertEqual(len(policies(r)),3);r['sources']=100
        with self.assertRaises(AssertionError):policies(r)
    def test_native_math_population_only(self):
        s=native.adapt((pathlib.Path(__file__).parents[1]/'a1_native.py').read_bytes());compile(s,'adapted','exec');self.assertIn("'tx_probability':100",s);self.assertIn('if False:',s)
if __name__=='__main__':unittest.main(verbosity=2)
