"""Synthetic CPU tests only: denominator, pairing, hash and interval contracts."""
import unittest
from common import *
class TestProtocol(unittest.TestCase):
 def test_cosine_is_not_percentage(self):
  a=np.array([1.,0.],dtype=np.float32);b=np.array([-.5,np.sqrt(.75)],dtype=np.float32)
  self.assertAlmostEqual(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),-.5)
 def test_pixels_preserve_floats_and_negative_reject(self):
  a=np.full((3,256,256),.123456,np.float32);self.assertEqual(ash(pixels(a)),ash(a))
  with self.assertRaises(AssertionError):pixels(a-.5)
 def test_same_pixels_dedup_not_source_pairs(self):
  gray=np.full((3,256,256),.5,np.float32)
  self.assertEqual(ash(gray),ash(gray.copy()))
  self.assertNotEqual(('source1',ash(gray)),('source2',ash(gray)))
 def test_grid_and_failure_denominator(self):
  src=[dict(source_index=i,source_id=str(i),pixel_sha256=str(i))for i in range(500)]
  rows=[dict(method=m,source_index=i,source_id=str(i),reference_sha256=str(i),N=1024,snr_db=s,noise_seed=n,fixed_gray=True)
        for m in METHODS for i in range(500)for s in SNRS for n in SEEDS[m]]
  grid(rows,src)
  with self.assertRaises(AssertionError):grid(rows[:-1],src)
  with self.assertRaises(AssertionError):grid(rows[:-1]+[rows[0]],src)
 def test_direct_pairing_retains_negative(self):
  bank=np.tile(np.arange(500,dtype=np.int64),(10000,1))
  a=np.arange(500,dtype=np.float64)/1000;b=a+.2
  d=interval(a-b,bank)
  for v in d.values():self.assertAlmostEqual(v,-.2)
 def test_marginal_intervals_not_subtracted(self):
  bank=np.zeros((10000,500),dtype=np.int64);bank[5000:]=499
  a=np.arange(500,dtype=np.float64);b=a+1
  self.assertEqual(interval(a-b,bank)['ci_high'],-1)
  self.assertNotEqual(interval(a,bank)['ci_high']-interval(b,bank)['ci_low'],-1)
if __name__=='__main__':unittest.main()
