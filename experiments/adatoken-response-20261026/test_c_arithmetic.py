import unittest,sys
from pathlib import Path
import numpy as np
from c_arithmetic import *
class Arithmetic(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.enc,cls.dec,cdf,cls.proof=load_codec(Path(sys.argv[1]));cls.cdf=staticmethod(cdf)
 def test_vocab_and_mass(self):
  a=self.cdf(np.zeros((2,16384),np.float32));self.assertEqual(a.shape,(2,16385));self.assertTrue(np.all(np.diff(a,axis=1)>0));self.assertTrue(np.all(a[:,-1]==1<<24))
 def test_extreme_finite(self):
  x=np.full((1,16384),-10000.,np.float32);x[0,-1]=0;a=self.cdf(x);self.assertTrue(np.all(np.diff(a)>0));self.assertGreater(a[0,-1]-a[0,-2],16000000)
 def test_independent_all_prefixes(self):
  t=np.random.default_rng(26).integers(0,16384,256,dtype=np.int64);t[-1]=16383;cdf=self.cdf(np.zeros((1,16384),np.float32));e=self.enc()
  for i in range(256):
   e.encode(t[i:i+1],cdf);before=(e.low,e.high,e.pending,len(e.bits));bits=e.finish();self.assertEqual(before,(e.low,e.high,e.pending,len(e.bits)));d=self.dec(bits);c=self.enc();v=d.decode(np.repeat(cdf,i+1,axis=0));c.encode(v,np.repeat(cdf,i+1,axis=0));canonical(d,c,bits);np.testing.assert_array_equal(v,t[:i+1])
 def test_reject_bad_tokens(self):
  c=self.cdf(np.zeros((1,16384),np.float32))
  for token in(-1,16384):
   with self.assertRaises(ValueError):self.enc().encode(np.array([token]),c)
 def test_noncanonical_extra_bit(self):
  c=self.cdf(np.zeros((1,16384),np.float32));e=self.enc();e.encode(np.array([1000]),c);b=np.r_[e.finish(),np.uint8(0)];d=self.dec(b);v=d.decode(c);r=self.enc();r.encode(v,c)
  with self.assertRaises(AssertionError):canonical(d,r,b)
 def test_seed_uses_only_received(self):
  a=np.array([0,1,0,1],np.uint8);self.assertEqual(generation_seed(a,32),generation_seed(a.copy(),32));self.assertNotEqual(generation_seed(a,32),generation_seed(a,128))
 def test_inverse_proof(self):self.assertTrue(self.proof['inverse_AST_equal']);self.assertEqual(self.proof['vocab'],16384)
if __name__=='__main__':unittest.main(argv=[sys.argv[0]],verbosity=2)
