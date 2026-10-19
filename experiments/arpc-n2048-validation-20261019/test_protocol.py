import unittest,random
import pair_protocol as p
class ProtocolTests(unittest.TestCase):
 def setUp(self):
  self.data=bytes(range(256))*4;self.length=8192;self.fs=p.fragment(self.data,self.length,0,123);self.ps=p.profiles()
 def pairs(self,fs=None):return [(v,self.ps[str(i)]) for i,v in enumerate(fs or self.fs)]
 def test_capacity(self):self.assertEqual((p.FRAGMENT_BYTES,p.MAX_BYTES,8*p.MAX_BYTES),(581,1162,9296));self.assertEqual(2*p.K-(26+32+192)-9296,14)
 def test_complete(self):self.assertEqual(p.reassemble(self.pairs())['data'],self.data)
 def test_reversed(self):self.assertEqual(p.reassemble(self.pairs()[::-1])['data'],self.data)
 def test_missing(self):
  with self.assertRaises(p.Reject):p.reassemble(self.pairs()[:1])
 def test_duplicate(self):
  with self.assertRaises(p.Reject):p.reassemble([self.pairs()[0]]*2)
 def test_mixed_pair(self):
  f=p.fragment(self.data,self.length,0,124)
  with self.assertRaises(p.Reject):p.reassemble(self.pairs([self.fs[0],f[1]]))
 def test_wrong_profile(self):
  with self.assertRaises(p.Reject):p.reassemble([(self.fs[0],self.ps['2']),(self.fs[1],self.ps['3'])])
 def test_whole_crc_after_body_crc(self):
  b=bytearray(self.fs[1]);b[-1]^=128;f=[self.fs[0],bytes(b)]
  clean=[p.parse_body(p.pack_body(v,self.ps[str(i)]),self.ps[str(i)]) for i,v in enumerate(f)]
  with self.assertRaises(p.Reject):p.reassemble(self.pairs(clean))
 def test_metadata_reserved(self):
  b=bytearray(self.fs[0]);b[7]^=1
  with self.assertRaises(p.Reject):p.reassemble(self.pairs([bytes(b),self.fs[1]]))
 def test_body_crc(self):
  b=p.pack_body(self.fs[0],self.ps['0']);b[150]^=1
  with self.assertRaises(p.Reject):p.parse_body(b,self.ps['0'])
 def test_body_padding(self):
  b=p.pack_body(self.fs[0],self.ps['0']);b[-17]=1;b[-16:]=p.integer_bits(p.crc16(b[:-16]),16)
  with self.assertRaises(p.Reject):p.parse_body(b,self.ps['0'])
 def test_null(self):
  self.assertEqual(p.parse_body(p.pack_body(b'',self.ps['4']),self.ps['4']),b'')
  with self.assertRaises(p.Reject):p.reassemble([(b'',self.ps['4'])]*2)
 def test_bits(self):
  for n in range(1,50):
   b=[random.Random(n+i).randrange(2) for i in range(n)];self.assertEqual(p.unpack_bits(p.pack_bits(b))[:n],b)
 def test_nonbyte(self):
  data=b'\xff\x80';f=p.fragment(data,9,0,0);self.assertEqual(p.reassemble([(f[i],self.ps[str(i)]) for i in (0,1)])['data'],data)
  with self.assertRaises(p.Reject):p.validate_object(b'\xff\x81',9,0)
 def test_arpc_alignment(self):
  with self.assertRaises(p.Reject):p.validate_object(b'\x80',1,1)
 def test_maximum(self):
  f=p.fragment(b'\0'*1162,9296,0,0);self.assertEqual([len(x) for x in f],[593,593])
  for i,x in enumerate(f):self.assertEqual(len(p.pack_body(x,self.ps[str(i)])),4780)
 def test_oversized(self):
  with self.assertRaises(p.Reject):p.fragment(b'\0'*1163,9304,0,0)
 def test_all_codecs(self):
  for codec in (0,1):
   f=p.fragment(self.data,self.length,codec,2**32-1);self.assertEqual(p.reassemble([(f[i],self.ps[str(2*codec+i)]) for i in (0,1)])['codec'],codec)
 def test_accounting(self):self.assertEqual(2*(68+956),2048);self.assertEqual(100*3*3*2*2+6*2*2*2,3648)
 def test_no_source_interface(self):
  import inspect
  self.assertEqual(list(inspect.signature(p.reassemble).parameters),['received'])
if __name__=='__main__':unittest.main(verbosity=2)
