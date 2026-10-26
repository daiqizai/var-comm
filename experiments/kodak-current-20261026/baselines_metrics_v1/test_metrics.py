"""Synthetic-only guards: identity, population denominator, call durability, no models."""
import unittest,tempfile,sys,math
from pathlib import Path
import numpy as np
from common import *
class MetricsTest(unittest.TestCase):
 def grid(self):return [dict(method=m,source_index=i,snr_db=s,noise_seed=n) for m in METHODS for i in range(24) for s in (4,10,19) for n in (2001,2002,2003)]
 def test_full_denominator(self):assert_grid(self.grid())
 def test_missing_gray_not_dropped(self):
  with self.assertRaises(AssertionError):assert_grid(self.grid()[:-1])
 def test_duplicate_displaces_source(self):
  rows=self.grid();rows[-1]=rows[0]
  with self.assertRaises(AssertionError):assert_grid(rows)
 def test_alias(self):self.assertEqual(framekey(dict(method='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024',source_index='1',snr_db='4',noise_seed='2001')),('BPG_ADAPTIVE',1,4,2001))
 def test_array_hash(self):
  x=np.full((3,256,256),.5,dtype=np.float32);self.assertEqual(assert_array(x,ah(x),'dtype_shape_cbytes'),ah(x));self.assertEqual(assert_array(x,rawah(x),'raw_bytes'),ah(x))
 def test_hash_mismatch(self):
  x=np.zeros((3,256,256),np.float32)
  with self.assertRaises(AssertionError):assert_array(x,'0'*64,'dtype_shape_cbytes')
 def test_unknown_convention(self):
  x=np.zeros((3,256,256),np.float32)
  with self.assertRaises(AssertionError):assert_array(x,ah(x),'guess')
 def test_native_dtype_required(self):
  x=np.zeros((3,256,256),np.float64)
  with self.assertRaises(AssertionError):assert_array(x,ah(x),'dtype_shape_cbytes')
 def test_no_negative_zero_normalization(self):
  x=np.zeros((3,256,256),np.float32);y=x.copy();y[0,0,0]=-0.;self.assertNotEqual(ah(x),ah(y));assert_array(y,ah(y),'dtype_shape_cbytes')
 def test_scalar_completion_reuse(self):
  with tempfile.TemporaryDirectory() as t:
   l=Ledger(t,{'score':1});self.assertEqual(l.call('score','a',lambda:.123),.123)
   self.assertEqual(l.call('score','a',lambda:1/0),.123);self.assertEqual(l.counts(),[('score','COMPLETE',1)]);l.db.close()
   l=Ledger(t,{'score':1});self.assertEqual(l.call('score','a',lambda:1/0),.123);l.db.close()
 def test_feature_reuse(self):
  with tempfile.TemporaryDirectory() as t:
   l=Ledger(t,{'feature':1});a=np.arange(8,dtype=np.float32);l.call('feature','a',lambda:a,True);b=l.call('feature','a',lambda:1/0,True);self.assertTrue(np.array_equal(a,b));l.db.close()
 def test_cap(self):
  with tempfile.TemporaryDirectory() as t:
   l=Ledger(t,{'score':1});l.call('score','a',lambda:1.)
   with self.assertRaises(AssertionError):l.call('score','b',lambda:2.)
   l.db.close()
 def test_failed_call_preserved(self):
  with tempfile.TemporaryDirectory() as t:
   l=Ledger(t,{'score':2})
   with self.assertRaises(ZeroDivisionError):l.call('score','a',lambda:1/0)
   with self.assertRaises(AssertionError):l.call('score','a',lambda:2.)
   self.assertEqual(l.counts(),[('score','FAILED',1)]);l.db.close()
 def test_scalar_nonfinite_fails(self):
  with tempfile.TemporaryDirectory() as t:
   l=Ledger(t,{'score':1})
   with self.assertRaises(ValueError):l.call('score','a',lambda:float('nan'))
   self.assertEqual(l.counts(),[('score','FAILED',1)]);l.db.close()
 def test_receipt_corruption_not_recomputed(self):
  with tempfile.TemporaryDirectory() as t:
   l=Ledger(t,{'score':1});l.call('score','a',lambda:1.);p=next((Path(t)/'call_results').glob('*.json'));p.write_text('{}')
   with self.assertRaises(AssertionError):l.call('score','a',lambda:2.)
   l.db.close()
 def test_no_model_imports(self):self.assertFalse(set(sys.modules)&{'torch','lpips','dreamsim','clip','transformers'})
if __name__=='__main__':unittest.main()
