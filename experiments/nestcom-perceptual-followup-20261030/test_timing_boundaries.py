"""CPU adapter tests, explicitly not real-weight latency/quality acceptance."""
import unittest,types,sys
from timing_end_to_end import MemoryCalls,counters,rx,R,np

class BoundaryTests(unittest.TestCase):
 def test_counter_does_not_need_intended_profile_at_receiver(self):
  g=dict(N=1024,snr_db=7)
  a,base=counters(g,17,100);b,base2=counters(g,17,300)
  self.assertEqual(base,base2);self.assertEqual(a-base,100);self.assertEqual(b-base,300)
 def test_packet_callback_variants(self):
  c=MemoryCalls();self.assertEqual(c.call({'kind':'header'}, {},lambda:7),7)
  self.assertEqual(c.decode('x','y','body',{},lambda:8),8)
  self.assertEqual(c.counts,dict(header=1,body=1))
 def test_rejection_stops_before_source_decoder(self):
  sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'))
  event=dict(header=dict(header_ok=False),body=None,rx_profile=None,status='HEADER_REJECT')
  seen=[]
  class RT:
   profiles={'public':'entire catalogue'}
   def receive(self,*args,**kwargs):seen.append((args,kwargs));return event
  class Forbidden:
   def __getattr__(self,name):raise AssertionError('Source/model touched on rejected reception')
  ctx=dict(g=dict(method='Token_NeST',N=1024,snr_db=10),n=Forbidden(),rt=RT(),phy=None,calls=MemoryCalls(),worker=types.SimpleNamespace(Clock=lambda _:None),c=Forbidden())
  image,status=rx(np.zeros((1024,2)),dict(counter=4,public_base=None),ctx,'test')
  self.assertTrue(np.array_equal(image,np.full((3,256,256),.5,np.float32)))
  self.assertFalse(status['success']);self.assertEqual(len(seen),1)
  self.assertEqual(seen[0][0][3],{'public':'entire catalogue'})

if __name__=='__main__':unittest.main()
