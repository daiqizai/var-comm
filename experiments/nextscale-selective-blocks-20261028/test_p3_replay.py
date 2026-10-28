"""Protocol-cache adversarial tests, all synthetic and CPU-only."""
import unittest
from block_common import digest
from protocol import state_identity
from p3_rx_replay import replay

class ReplayTests(unittest.TestCase):
 def bundle(self):
  C=list(range(30));pos=list(range(6,12));B=list(range(200,206));cp=dict(type='C',m=4,bits=[0,1]);bp=dict(type='BOUNDARY',m=4,C_bits=[0,1],positions=pos,bits=[1,0])
  cache={digest(cp):dict(projection=cp,value=C),digest(bp):dict(projection=bp,value=B)};native={}
  for positions,values,px in [([],[],'c'),(pos,B,'cb')]:
   s=state_identity(4,C,positions,values,'fixed',{});native[digest(s)]=dict(state=s,pixel_sha256=px)
  return dict(cache=cache,native=native,route={},preprocessing='fixed')
 def event(self):
  return dict(header=dict(header_ok=True),profile=dict(kind='THREE',m=4,K=12,null=False,positions_A=list(range(6)),positions_B=list(range(6,12))),blocks=[dict(accepted=True,status='PARSED',bits=[0,1]),dict(accepted=False,status='CRC_REJECT'),dict(accepted=True,status='PARSED',bits=[1,0])])
 def test_B_survives_missing_A_and_replay_shares_received_bits(self):
  r=replay(self.event(),self.bundle());self.assertEqual(r['missing'],[])
  self.assertEqual([r['modes'][x]['state']for x in ('ALL','SEQ','SEL')],['GRAY','C','CB'])
  self.assertEqual([r['modes'][x]['pixel_sha256']for x in ('SEQ','SEL')],['c','cb'])
 def test_unknown_accepted_bits_are_not_replaced_by_expected(self):
  e=self.event();e['blocks'][2]['bits']=[1,1]
  r=replay(e,self.bundle());self.assertEqual(len(r['missing']),1);self.assertNotIn('modes',r)
 def test_missing_C_does_not_use_B_cache(self):
  e=self.event();e['blocks'][0]=dict(accepted=False,status='CRC_REJECT')
  r=replay(e,self.bundle());self.assertEqual(r['reused_decode_keys'],[]);self.assertEqual(r['modes']['SEL']['state'],'GRAY')
if __name__=='__main__':unittest.main()
