import unittest,numpy as np
from statistics_schema_v3 import interval
class PairedStatistics(unittest.TestCase):
 def test_paired_constant_difference(self):
  bank=np.array([[0,0,1],[2,2,1],[0,1,2]],np.int64);left=np.array([2.,6.,12.]);right=left-1.25
  self.assertEqual(interval(left-right,bank),dict(mean=1.25,ci_low=1.25,ci_high=1.25))
  self.assertGreater(interval(left,bank)['ci_high']-interval(right,bank)['ci_low'],1.25)
 def test_source_order_controls_pairing(self):
  order=['c','a','b'];left={'a':10.,'b':20.,'c':30.};right={'b':19.,'c':29.,'a':9.}
  bank=np.array([[0,0,1],[2,2,1],[0,1,2]],np.int64)
  self.assertEqual(interval([left[s]-right[s]for s in order],bank)['ci_low'],1.)
if __name__=='__main__':unittest.main()
