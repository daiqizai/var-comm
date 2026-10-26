"""Synthetic CPU context only; no assets, model, or actual source traversal."""
import unittest,sys,numpy as np
from contextlib import nullcontext
from types import SimpleNamespace
from common import *
from a_codec import receiver_class,endpoint
class FakeProvider:
 def __init__(self):self.scale=0;self.history=[]
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def cdf(self):
  counts=np.full(4096,4096,np.int64);k=(self.scale+sum(int(x.sum())for x in self.history))%4096;counts[k]+=2048;counts[(k+1)%4096]-=2048
  return np.broadcast_to(np.r_[0,counts.cumsum()],(SIZES[self.scale]**2,4097))
 def advance(self,v):
  assert v.shape==(SIZES[self.scale]**2,);self.history.append(v.copy());self.scale+=1
class Tests(unittest.TestCase):
 def setUp(self):
  C,self.proof=receiver_class();self.c=C(R);self.c.native=SimpleNamespace(torch=SimpleNamespace(no_grad=nullcontext));self.c.provider_factory=FakeProvider
  self.tokens=(np.arange(680,dtype=np.int64)*61+17)%4096
 def encode(self,m,K):
  p=FakeProvider();e=self.c.Encoder()
  for scale in range(m+int(K>0)):
   v=self.tokens[OFFSETS[scale]:OFFSETS[scale+1]];table=p.cdf();n=len(v)if scale<m else K;e.encode(v[:n],table[:n])
   if scale<m:p.advance(v.copy())
  return e.finish()
 def test_all49_roundtrip(self):
  for m,K in grids()['8']:
   r=self.c.decode('EC_VAR_PARTIAL'if K else'EC_VAR_WHOLE',self.encode(m,K),m,K)
   np.testing.assert_array_equal(r['received_tokens'],self.tokens[:OFFSETS[m]+K]);self.assertTrue(r['canonical']);self.assertEqual(r['zero_extension_reads'],30)
 def test_original_body(self):self.assertTrue(self.proof['original_decode_body_unchanged'])
 def test_bad_endpoint(self):
  for v in[(4,1),(3,0),(10,1),(11,0),(True,0)]:
   with self.assertRaises(ValueError):endpoint(*v)
 def test_noncanonical_rejected(self):
  with self.assertRaises(ValueError):self.c.decode('EC_VAR_PARTIAL',np.r_[self.encode(7,37),np.uint8(0)],7,37)
 def test_whole_cannot_claim_boundary(self):
  with self.assertRaises(ValueError):self.c.decode('EC_VAR_WHOLE',self.encode(7,37),7,37)
if __name__=='__main__':unittest.main()
