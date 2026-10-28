"""CPU-only adversarial framing/proxy tests; no real PHY or scientific calls."""
import unittest
import numpy as np
from unittest.mock import patch
from protocol import catalogue
from nominate_p2 import probabilities
from p3_phy import receive_frames

class LedgerStub:
 def call(self,k,key,fn):return fn()

class P3Tests(unittest.TestCase):
 def test_received_header_controls_actual_geometry(self):
  profiles=catalogue()['profiles']
  p=next(v for v in profiles.values()if v['kind']=='THREE'and not v['null']and v['K']>0)
  class Header:
   def receive(self,y,snr,catalog):return dict(header_ok=True,profile_id=p['profile_id'])
  seen=[]
  def rx(b,phy,packet,y,g,snr,ctrs,group,empty,session):
   seen.append((group,g['N'],g['k'],y.shape,tuple(empty)))
   return np.zeros((len(y),g['k']),np.uint8),[dict(accepted=False,status='CRC_REJECT')for _ in y]
  budget=dict(decodes=0,decode_limit=3)
  with patch('p3_phy.receive',rx):
   records,_=receive_frames(np.ones((1,1024,2),np.float32),2,[23],profiles,None,None,Header(),None,LedgerStub(),'wrong_legal',budget)
  self.assertEqual([v[1]for v in seen],[g['N']for g in p['layouts']]);self.assertEqual(budget['decodes'],3)
  self.assertEqual(records[0]['profile'],p)

 def test_rejected_header_does_not_decode_body(self):
  class Header:
   def receive(self,*a):return dict(header_ok=False,profile_id=None)
  with patch('p3_phy.receive',side_effect=RuntimeError('must not decode')):
   r,_=receive_frames(np.ones((1,1024,2),np.float32),0,[1],{},None,None,Header(),None,LedgerStub(),'bad',dict(decodes=0,decode_limit=0))
  self.assertEqual(r[0]['blocks'],[])

 def test_proxy_disabled_AB_and_missing_A(self):
  p=next(v for v in catalogue()['profiles'].values()if v['kind']=='THREE'and not v['null']and v['K']==0)
  b={(g['N'],g['rate']):0.0 for g in p['layouts']};b[p['layouts'][0]['N'],p['layouts'][0]['rate']]=1
  self.assertEqual(probabilities(p,1,b),dict(C=1,GRAY=0))
  p=dict(p,K=12);v=probabilities(p,1,b);self.assertEqual(v['C'],1)
  self.assertAlmostEqual(sum(v.values()),1)

if __name__=='__main__':unittest.main()
