"""CPU-only engineering tests for the m10 catalog extension."""
import contextlib,unittest
import numpy as np
from seed_runtime import EndpointCodec,endpoint,ep,SourceCodec10,R
EPS=((7,0),(7,25),(8,0),(8,84),(8,126),(9,0),(9,128),(10,0))
class Uniform(EndpointCodec):
 def __init__(self):super().__init__(R);self.contexts=0
 def _inference(self,f):return contextlib.nullcontext()
 def _context(self,f):
  self.contexts+=1
  class P:
   scale=0
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def cdf(self):return np.broadcast_to(np.arange(4097,dtype=np.int64)*4096,(ep.SIZES[self.scale]**2,4097))
   def advance(self,values):self.scale+=1
  return P()
class TestCodec(unittest.TestCase):
 def test_one_pass_and_eight_independent_roundtrips(self):
  c=Uniform();tokens=np.arange(680,dtype=np.int64)%4096;v=c.encode_endpoints(tokens,EPS)
  self.assertEqual(c.contexts,1)
  for m,k in EPS:
   rx=c.decode('EC_VAR_PARTIAL' if k else 'EC_VAR_WHOLE',v[m,k]['bits'],m,k)
   np.testing.assert_array_equal(rx['received_tokens'],tokens[:int(ep.OFFSETS[m])+k])
   self.assertTrue(rx['canonical']);self.assertEqual(rx['zero_extension_reads'],30)
 def test_existing_streams_unchanged(self):
  c=Uniform();tokens=np.random.default_rng(12).integers(4096,size=680);v=c.encode_endpoints(tokens,EPS)
  old=ep.PartialSourceCodec.encode_endpoints(c,tokens,EPS[:-1])
  for e in EPS[:-1]:np.testing.assert_array_equal(v[e]['bits'],old[e]['bits'])
  whole=SourceCodec10.encode(c,'EC_VAR_WHOLE',tokens,(10,))
  np.testing.assert_array_equal(v[10,0]['bits'],whole[10]['bits'])
 def test_no_extra_catalog_entries(self):
  for e in [(10,1),(7,26),(8,85),(9,127),(11,0),(True,0)]:
   with self.assertRaises(ValueError):endpoint(*e)
if __name__=='__main__':unittest.main()
