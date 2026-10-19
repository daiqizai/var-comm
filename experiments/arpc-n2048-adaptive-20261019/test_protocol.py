import unittest,hashlib
import pair_protocol as p
from adaptive_plan import layouts,fallback,choose,NEST_ENDPOINTS
class ProtocolTests(unittest.TestCase):
 def data(self,n):return bytes((i*17+53)%256 for i in range(n))
 def test_public_directory_charged_range(self):
  d=p.profiles();self.assertEqual(len(d),798);self.assertEqual(list(map(int,d)),list(range(798)))
  for k,v in p.legacy().profiles().items():self.assertEqual(d[k],v)
 def test_capacities_and_every_actual_endpoint(self):
  caps={g['layout_id']:g['object_bits'] for g in layouts()}
  self.assertEqual([caps[k] for k in ('q2_r2-3','q4_r2-3','q6_r5-6')],[2288,4848,9296])
  for codec in (0,1):
   for end in (range(25) if codec==0 else range(1,8)):
    for g in layouts():
     n=2*g['fragment_bytes'];raw=self.data(n);fs=p.fragment(raw,n*8,codec,end,343,g['layout_id'])
     received=[(p.parse_body(p.pack_body(d,q),q),q) for d,q in fs]
     self.assertEqual(p.reassemble(received)['data'],raw)
     with self.assertRaises(p.Reject):p.fragment(raw+b'\0',n*8+8,codec,end,343,g['layout_id'])
 def test_nest_bit_length_and_zero_alignment(self):
  raw=b'\xfa\x80';fs=p.fragment(raw,9,0,0,4,'q2_r1-2');self.assertEqual(p.reassemble(fs)['bit_length'],9)
  with self.assertRaises(p.Reject):p.fragment(b'\xfa\x81',9,0,0,4,'q2_r1-2')
  with self.assertRaises(p.Reject):p.fragment(raw,9,1,1,4,'q2_r1-2')
 def test_crc_reject_and_canonical_padding(self):
  fs=p.fragment(self.data(20),160,0,2,4,'q2_r1-2');raw,q=fs[0];bits=p.pack_body(raw,q);bits[40]^=1
  with self.assertRaises(p.Reject):p.parse_body(bits,q)
  bits=p.pack_body(raw,q);bits[-17]=1;bits[-16:]=p.integer_bits(p.crc16(bits[:-16]),16)
  with self.assertRaises(p.Reject):p.parse_body(bits,q)
 def test_actual_wrong_but_legal_profile_rejected_without_truth(self):
  fs=p.fragment(self.data(20),160,0,2,4,'q2_r1-2');data,q=fs[0];wrong=p.profile(0,3,'q2_r1-2',0)
  bits=p.pack_body(data,wrong);parsed=p.parse_body(bits,wrong)
  with self.assertRaises(p.Reject):p.reassemble([(parsed,wrong),fs[1]])
 def test_mixed_geometry_rejected_even_if_source_metadata_same(self):
  a=p.fragment(self.data(20),160,1,4,4,'q2_r1-2');b=p.fragment(self.data(20),160,1,4,4,'q4_r1-2')
  with self.assertRaises(p.Reject):p.reassemble([a[0],b[1]])
 def test_truncated_duplicate_reversed(self):
  fs=p.fragment(self.data(21),168,1,5,99,'q2_r1-2')
  self.assertEqual(p.reassemble(list(reversed(fs)))['data'],self.data(21))
  for x in ([fs[0]],[fs[0],fs[0]],[(fs[0][0][:-1],fs[0][1]),fs[1]]):
   with self.assertRaises(p.Reject):p.reassemble(x)
 def test_whole_crc_detects_changed_data_with_valid_outer_crc(self):
  fs=p.fragment(self.data(21),168,1,3,99,'q2_r1-2');d,q=fs[1];d=d[:-1]+bytes([d[-1]^1]);fs[1]=(p.parse_body(p.pack_body(d,q),q),q)
  with self.assertRaises(p.Reject):p.reassemble(fs)
 def test_null_profile_still_has_full_k_and_crc(self):
  for q in p.directory().values():
   if q['mode']=='null':
    bits=p.pack_body(b'',q);self.assertEqual(len(bits),q['k']);self.assertEqual(p.parse_body(bits,q),b'')
    with self.assertRaises(p.Reject):p.pack_body(b'x',q)
 def test_old_anchor_body_and_reassembly_exact(self):
  old=p.legacy()
  for codec in (0,1):
   raw=self.data(100);fs=old.fragment(raw,800,codec,19);qs=[old.profiles()[str(2*codec+i)] for i in (0,1)]
   for data,q in zip(fs,qs):self.assertEqual(p.pack_body(data,q),old.pack_body(data,q));self.assertEqual(p.parse_body(old.pack_body(data,q),q),data)
   self.assertEqual(p.reassemble(list(zip(fs,qs)))['data'],raw)
 def test_reassembly_metadata_reserved_version_pair_id(self):
  original=p.fragment(self.data(20),160,0,5,44,'q4_r2-3')
  for bit in (0,4,6,11,12,13,45,59,64):
   fs=list(original);data,q=fs[0];b=p.unpack_bits(data);b[bit]^=1;fs[0]=(p.pack_bits(b),q)
   with self.assertRaises(p.Reject):p.reassemble(fs)
 def test_paid_profile_not_mutable_config(self):
  q=dict(p.profile(0,0,'q2_r1-2',0));q['q']=6
  with self.assertRaises(p.Reject):p.pack_body(b'x'*20,q)
 def test_fallback_does_not_use_quality(self):
  self.assertEqual(fallback(0,12),(12,8,4,0));self.assertEqual(fallback(0,13),tuple(range(13,-1,-1)));self.assertEqual(fallback(0,24),tuple(range(24,-1,-1)))
  g=layouts()[0];ss={i:dict(bit_length=1700 if i>=8 else 800) for i in range(25)}
  self.assertEqual(choose(ss,0,24,g),7);self.assertEqual(choose(ss,0,12,g),4)
 def test_input_key_binds_actual_endpoint(self):
  self.assertNotEqual(p.input_key(0,0,160,self.data(20)),p.input_key(0,1,160,self.data(20)))
if __name__=='__main__':unittest.main(verbosity=2)
