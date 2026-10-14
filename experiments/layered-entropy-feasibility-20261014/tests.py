"""Synthetic conditional-CDF/source/packet tests; no Torch or real assets loaded."""
import contextlib,inspect,types,unittest
import numpy as np
from common import R,ENDS,OFFSETS,SIZES
from source_split_codec import SplitSourceCodec
from packet_two_group import *
class Provider:
 def __init__(self):self.scale=0;self.history=[]
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def cdf(self):
  # Context depends only on explicit received advance(); no original source.
  weights=np.ones(4096,np.int64);weights[(sum(self.history)+self.scale*7)%4096]=4097
  weights*=2048;row=np.r_[0,np.cumsum(weights)]
  assert row[-1]==1<<24
  return np.broadcast_to(row,(SIZES[self.scale]**2,4097)).copy()
 def advance(self,values):self.history.extend(map(int,values));self.scale+=1
def codec():
 c=SplitSourceCodec(R);c.native=types.SimpleNamespace(torch=types.SimpleNamespace(no_grad=contextlib.nullcontext))
 c.provider_factory=Provider;return c
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=codec();cls.tokens=np.arange(680,dtype=np.int64)%4096;cls.encoded=cls.c.encode_all(cls.tokens)
  # Uniform synthetic data can require more bits than the real proof layout.
  # This is a source-syntax fixture, never a legal wireless-capacity claim.
  cls.cat={i:dict(p,k_B=8191,k_E=8191,status='SYNTHETIC_SOURCE_SYNTAX_NOT_A_PHYSICAL_FRAME') if not p['null_profile'] else p for i,p in proof_catalogue().items()}
 def received(self,j=6,m=8,K=42):
  p=next(x for x in self.cat.values() if not x['null_profile'] and (x['j'],x['m'],x['K'])==(j,m,K))
  B=pack_group(self.encoded['base'][j]['bits'],p['k_B'])
  E=pack_group(self.encoded['enhancement'][(j,m,K)]['bits'],p['k_E'])
  return p,intbits(p['profile_id'],12),B,E
 def test_counts_and_finish(self):
  self.assertEqual(len(self.encoded['single']),24);self.assertEqual(len(self.encoded['enhancement']),68)
  for x in (*self.encoded['single'].values(),*self.encoded['enhancement'].values()):self.assertGreaterEqual(x['finish_bits'],2)
 def test_all_68_conditional_split_roundtrips(self):
  for j,m,K in self.encoded['enhancement']:
   p=next(x for x in self.cat.values() if not x['null_profile'] and (x['j'],x['m'],x['K'])==(j,m,K))
   B=pack_group(self.encoded['base'][j]['bits'],p['k_B']);E=pack_group(self.encoded['enhancement'][(j,m,K)]['bits'],p['k_E'])
   rx=self.c.decode_received(intbits(p['profile_id'],12),B,E,self.cat)
   self.assertEqual(rx['status'],'FULL_RECEIVED_PREFIX');np.testing.assert_array_equal(rx['received_tokens'],self.tokens[:OFFSETS[m]+K])
 def test_base_failure_does_not_compute_any_CDF(self):
  p,h,B,E=self.received();B[20]^=1;rx=self.c.decode_received(h,B,E,self.cat)
  self.assertEqual(rx['status'],'GRAY');self.assertEqual(rx['cdf_trace'],[])
 def test_enhancement_crc_failure_preserves_only_base(self):
  p,h,B,E=self.received();E[20]^=1;rx=self.c.decode_received(h,B,E,self.cat)
  self.assertEqual(rx['status'],'BASE_ONLY');np.testing.assert_array_equal(rx['received_tokens'],self.tokens[:OFFSETS[6]])
  self.assertEqual(len(rx['cdf_trace']),6)
 def test_explicit_empty_profile(self):
  p=next(x for x in self.cat.values() if not x['null_profile'] and (x['j'],x['m'],x['K'])==(6,6,0))
  rx=self.c.decode_received(intbits(p['profile_id'],12),pack_group(self.encoded['base'][6]['bits'],p['k_B']),pack_group([],p['k_E'],True),self.cat)
  self.assertEqual(rx['reason'],'EXPLICIT_PAID_EMPTY_ENHANCEMENT')
 def test_zero_length_for_nonempty_rejected(self):self.assertFalse(parse_group(pack_group([],64,True),64)['accepted'])
 def test_nonzero_length_for_empty_rejected(self):self.assertFalse(parse_group(pack_group([0,1],64),64,True)['accepted'])
 def test_length_overflow(self):
  with self.assertRaises(AssertionError):pack_group(np.zeros(36,np.uint8),64)
 def test_known_padding_rejected_even_valid_crc(self):
  a=pack_group([0,1],64);a[30]=1;a[-16:]=intbits(crc16(a[:-16]),16);self.assertEqual(parse_group(a,64)['status'],'KNOWN_PADDING_REJECT')
 def test_noncanonical_enhancement_preserves_base(self):
  p,h,B,E=self.received();bad=np.r_[self.encoded['enhancement'][(6,8,42)]['bits'],0];E=pack_group(bad,p['k_E'])
  self.assertEqual(self.c.decode_received(h,B,E,self.cat)['status'],'BASE_ONLY')
 def test_null(self):
  p=next(x for x in self.cat.values() if x['null_profile']);rx=self.c.decode_received(intbits(p['profile_id'],12),[],[],self.cat)
  self.assertEqual(rx['status'],'GRAY');self.assertEqual(rx['cdf_trace'],[])
 def test_unknown_paid_header(self):self.assertEqual(self.c.decode_received(intbits(4095,12),[],[],self.cat)['status'],'GRAY')
 def test_public_directory_contains_all_fallbacks_without_collision(self):self.assertEqual(len(self.cat),73);self.assertEqual(len({x['profile_id'] for x in self.cat.values()}),73)
 def test_base_priority_fallback(self):
  b={4:50,5:80,6:100,7:110};e={(j,m,K):100 for j,m,K in self.encoded['enhancement']}
  self.assertEqual(choose_fallback(b,e,7,(8,42),105,101)['j'],6)
 def test_null_keeps_failure(self):
  b={j:100 for j in (4,5,6,7)};e={(j,m,K):100 for j,m,K in self.encoded['enhancement']}
  self.assertEqual(choose_fallback(b,e,7,(8,42),99,100)['status'],'TX_SOURCE_UNENCODABLE')
 def test_no_truth_or_TX_CDF_RX_parameters(self):
  self.assertEqual(list(inspect.signature(self.c.decode_received).parameters),['received_header_bits','base_information','enhancement_information','catalogue'])
 def test_failed_enhancement_image_renderer_receives_actual_base(self):
  p,h,B,E=self.received();E[20]^=1;seen=[]
  def render(native,tokens,m,K):seen.append((tokens.copy(),m,K));return np.full((3,256,256),.25,np.float32)
  im,rx=self.c.receive_image(h,B,E,self.cat,render);self.assertEqual(seen[0][1:],(6,0));np.testing.assert_array_equal(seen[0][0],self.tokens[:91])
 def test_gray_has_no_renderer_call(self):
  im,rx=self.c.receive_image(intbits(4095,12),[],[],self.cat,lambda *a:self.fail('gray must not render'))
  self.assertTrue(np.all(im==.5))
if __name__=='__main__':unittest.main()
