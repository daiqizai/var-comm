import unittest
import numpy as np
from analysis_data import *

def fixture():
 return [dict(method=m,source_id=s,snr_db=n,noise_seed=z,**{k:(i+1)*10+(z-6201)+(0 if m==METHODS[0]else i-1)for k in METRICS})for m in METHODS for i,s in enumerate(['a','b','c'])for n in [4,10,19]for z in [6201,6202,6203]]

class Tests(unittest.TestCase):
 def test_exact_denominator(self):
  rows=fixture();out=source_means(rows,['a','b','c']);self.assertEqual(len(out),126);self.assertEqual(out[0]['source_mean'],11)
 def test_duplicate_rejected(self):
  a=fixture();a[-1]=a[0]
  with self.assertRaises(AssertionError):source_means(a,['a','b','c'])
 def test_missing_rejected(self):
  with self.assertRaises(AssertionError):source_means(fixture()[:-1],['a','b','c'])
 def test_direct_paired_not_marginal_subtraction(self):
  x=source_means(fixture(),['a','b','c']);d=np.array([[0,0,0],[1,1,1],[2,2,2],[0,1,2]])
  s,p=summarize(x,['a','b','c'],d);self.assertEqual(len(s),42);self.assertEqual(len(p),21)
  self.assertEqual(p[0]['mean'],0);self.assertTrue(p[0]['interval_crosses_or_touches_zero']);self.assertAlmostEqual(p[0]['ci_low'],-.925);self.assertAlmostEqual(p[0]['ci_high'],.925)
 def test_failure_gray_not_dropped(self):
  a=fixture()
  for r in a:
   if r['source_id']=='a':r.update({k:0 for k in METRICS})
  v=source_means(a,['a','b','c']);self.assertEqual(v[0]['source_mean'],0);self.assertEqual(v[0]['noise_count'],3)
 def test_negative_preserved(self):
  a=fixture()
  for r in a:
   if r['method']==METHODS[0]:r.update({k:-10 for k in METRICS})
  _,p=summarize(source_means(a,['a','b','c']),['a','b','c'],np.array([[0,1,2]]));self.assertLess(p[0]['mean'],0)
if __name__=='__main__':unittest.main()
