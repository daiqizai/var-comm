"""Finite engineering checks for paid budgets, fallbacks and receiver parsing."""
import unittest,inspect
from n512_phy import *
class ContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.phy,cls.proof=engine()
 def test_inverse_ast(self):self.assertTrue(self.proof['reverse_AST_equal']);self.assertEqual(len(self.proof['changes']),8)
 def test_candidates(self):
  a=candidates(MCS);self.assertEqual(len(a),396);self.assertEqual(len({v['candidate_id'] for v in a}),396)
  self.assertEqual(sum(v['method']=='NEST' for v in a),300);self.assertEqual(sum(v['method']=='WHOLE' for v in a),84)
 def test_budget(self):
  for q,r,nest,ada in ((2,'2/3',563,553),(4,'2/3',1155,1145),(6,'5/6',2191,2181)):
   k=int(444*q*Fraction(r));self.assertEqual(k-29,nest);self.assertEqual(k-39,ada)
 def test_noise_shared_dimensions(self):
  a=self.phy.standard_noise('source',6201);self.assertEqual(a.shape,(512,2));np.testing.assert_array_equal(a,self.phy.standard_noise('source',6201));self.assertFalse(np.array_equal(a,self.phy.standard_noise('source',6202)))
 def test_full_paid_body(self):
  p=dict(k=592,mode='arithmetic',family=NEST_W);b=np.zeros(563,np.uint8);v=self.phy.pack_body(b,p);self.assertEqual(len(v),592);o=self.phy.parse_body(v,p);self.assertTrue(o['parser_accepted']);self.assertEqual(o['payload'],b.tolist())
 def test_overflow(self):
  with self.assertRaises(RuntimeError):self.phy.pack_body(np.zeros(564,np.uint8),dict(k=592,mode='arithmetic',family=NEST_W))
 def test_crc_reject(self):
  p=dict(k=592,mode='arithmetic',family=NEST_W);v=self.phy.pack_body(np.zeros(2,np.uint8),p);v[15]^=1;self.assertEqual(self.phy.parse_body(v,p)['status'],'CRC_REJECT')
 def test_padding_reject_after_crc(self):
  p=dict(k=592,mode='arithmetic',family=NEST_W);v=self.phy.pack_body(np.zeros(2,np.uint8),p);v[16]=1;v=self.phy.append_crc(v[:-16]);self.assertEqual(self.phy.parse_body(v,p)['invalid_reason'],'NONZERO_KNOWN_PADDING')
 def test_length_reject_after_crc(self):
  p=dict(k=592,mode='arithmetic',family=NEST_W);v=self.phy.pack_body(np.zeros(2,np.uint8),p);v[:13]=0;v=self.phy.append_crc(v[:-16]);self.assertEqual(self.phy.parse_body(v,p)['invalid_reason'],'L_OUTSIDE_PROFILE_CAPACITY')
 def test_paid_null_projection(self):
  rx=dict(header=dict(header_ok=True),rx_profile=dict(family=NULL),body=dict(crc_accepted=True,parser_accepted=True,payload=[0,0]));self.assertEqual(received_projection(rx)['reason'],'EXPLICIT_NULL_PROFILE');rx['body']['payload']=[1,0];self.assertEqual(received_projection(rx)['reason'],'INVALID_NULL_PAYLOAD')
 def test_paid_class_invalid(self):
  rx=dict(header=dict(header_ok=True),rx_profile=dict(family=ADA,m=1),body=dict(crc_accepted=True,parser_accepted=True,payload=[1]*10+[0,0]));self.assertEqual(received_projection(rx)['reason'],'CLASS_CONTAINER_REJECT')
 def test_received_only(self):
  self.assertEqual(list(inspect.signature(received_projection).parameters),['rx'])
  self.assertNotIn('payload',inspect.signature(self.phy.receive_frame).parameters)
 def test_complete_scale_target_partial_fallback(self):
  # A NeST target at K=0 still allows all shorter quarter endpoints.
  streams={f'm{m}_K{K}':np.zeros(1000,np.uint8) for m,K in ENDS};streams['m4_K18']=np.zeros(400,np.uint8)
  profiles=[dict(family=NEST_P,m=4,K=18,q=2,nominal_rate='1/2',k=444),dict(family=NULL,q=2,nominal_rate='1/2',k=444)]
  p,_,null=choose(streams,profiles,dict(method='NEST',target=[5,0],q=2,rate='1/2'));self.assertFalse(null);self.assertEqual(p['K'],18)
  _,_,null=choose(streams,profiles,dict(method='WHOLE',target=[5,0],q=2,rate='1/2'));self.assertTrue(null)
if __name__=='__main__':unittest.main(verbosity=2)
