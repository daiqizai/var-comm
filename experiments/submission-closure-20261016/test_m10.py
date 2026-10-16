import unittest,contextlib
import numpy as np
from common import R,SIZES,ENDS,OFFSETS
from m10_codec import AuditCodec,PartialSourceCodec,WHOLE_FAMILY
class Provider:
 def __init__(self):self.scale=0;self.sum=0
 def advance(self,a):self.sum+=int(a.sum());self.scale+=1
class Fake:
 def _inference(self,f):return contextlib.nullcontext()
 def _context(self,f):return contextlib.nullcontext(Provider())
 def _cdf(self,f,p,scale):
  assert scale==p.scale and 0<=scale<10
  # Nonuniform, received-prefix-dependent frequencies; no TX table supplied to RX.
  freqs=np.full(4096,4095,np.int64);freqs[(scale+p.sum)%4096]+=4096
  table=np.concatenate(([0],np.cumsum(freqs)))
  return np.broadcast_to(table,(SIZES[scale]**2,4097))
class New(Fake,AuditCodec):pass
class Old(Fake,PartialSourceCodec):pass
class Tests(unittest.TestCase):
 def test_regression_and_roundtrip(self):
  tokens=np.arange(680,dtype=np.int64)*5%4096;n=New(R);o=Old(R)
  streams,trace=n.encode_all(tokens);old=o.encode_endpoints(tokens,ENDS)
  self.assertEqual([x['scale']for x in trace],list(range(10)))
  for e in ENDS:self.assertTrue(np.array_equal(streams[e]['bits'],old[e]['bits']),e)
  rx=n.decode_m10(streams[10,0]['bits']);self.assertTrue(np.array_equal(rx['received_tokens'],tokens));self.assertEqual(rx['zero_extension_reads'],30)
  self.assertGreater(len(streams[10,0]['bits']),4751) # audit retains above-capacity streams
 def test_noncanonical_rejected(self):
  n=New(R);a,_=n.encode_all(np.zeros(680,np.int64))
  with self.assertRaises(Exception):n.decode_m10(np.r_[a[10,0]['bits'],np.uint8(0)])
if __name__=='__main__':unittest.main()
