"""Additional synthetic guards for multi-host process closure only."""
import unittest,copy
from common import *
class HostClosureTest(unittest.TestCase):
 def setUp(self):
  self.ids=[dict(pid=i,start_ticks=i*100) for i in (1,2,3)]
  self.proof=dict(hostname='actual-pro',registration_sha256='abc',role='clip_metrics',observed_unix=123.,processes=[dict(identity=p,gone=True) for p in self.ids])
 def test_same_host_exact_ids(self):verify_host_closure(self.proof,'actual-pro',self.ids,'abc','clip_metrics')
 def test_wrong_host_rejected(self):
  with self.assertRaises(AssertionError):verify_host_closure(self.proof,'other-host',self.ids,'abc','clip_metrics')
 def test_missing_score_process(self):
  self.proof['processes'].pop()
  with self.assertRaises(AssertionError):verify_host_closure(self.proof,'actual-pro',self.ids,'abc','clip_metrics')
 def test_live_or_wrong_generation(self):
  self.proof['processes'][0]['gone']=False
  with self.assertRaises(AssertionError):verify_host_closure(self.proof,'actual-pro',self.ids,'abc','clip_metrics')
if __name__=='__main__':unittest.main()
