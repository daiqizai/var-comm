"""CPU-only tests of new population/control glue; no model or scientific calls."""
import ast,tempfile,unittest
from kn_common import *
import numpy as np
from kn_native import choose
class Tests(unittest.TestCase):
 def streams(self):return {e:np.zeros(10+OFFSETS[e[0]]+e[1],dtype=np.uint8)for e in ENDS+((10,0),)}
 def test_six_frozen_targets(self):
  x=self.streams()
  expected={('WHOLE',4):(7,0),('WHOLE',10):(9,0),('PARTIAL',4):(7,25),('PARTIAL',10):(8,126),('WHOLE',19):(10,0),('PARTIAL',19):(10,0)}
  for key,target in expected.items():self.assertEqual(choose(x,*key)[0],target)
 def test_actual_finished_length_fallback(self):
  x=self.streams();x[10,0]=np.zeros(4752,dtype=np.uint8)
  self.assertEqual(choose(x,'WHOLE',19)[0],(9,0));self.assertEqual(choose(x,'PARTIAL',19)[0],(9,192))
 def test_inclusive_capacity(self):
  x=self.streams();x[10,0]=np.zeros(4751,dtype=np.uint8);self.assertEqual(choose(x,'WHOLE',19)[0],(10,0))
 def test_no_fake_null(self):
  x={e:np.zeros(9000,dtype=np.uint8)for e in ENDS+((10,0),)}
  with self.assertRaises(AssertionError):choose(x,'PARTIAL',4)
 def test_counter_216_and_family_shared(self):
  cs=[counter(i,s,n)for i in range(24)for s in (4,10,19)for n in (2001,2002,2003)]
  self.assertEqual(set(cs),set(range(216)))
 def test_rx_key_only_received_information(self):
  x=dict(header=dict(header_ok=True),body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0]),rx_profile=dict(family='EC_VAR_PARTIAL',m=7,K=25),TX_truth=[999])
  a=source_key(x);x['TX_truth']=[888];self.assertEqual(source_key(x),a)
  x['body']['payload']=[0,1];self.assertNotEqual(source_key(x),a)
 def test_gray_key_has_no_source_identity(self):
  x=dict(header=dict(header_ok=False),body=None,rx_profile=None,source='A');y=dict(x,source='B')
  self.assertEqual(source_key(x),source_key(y))
 def test_ledger_rejects_duplicate_and_cap(self):
  with tempfile.TemporaryDirectory()as d:
   l=Ledger(Path(d)/'calls.sqlite',dict(foo=1));self.assertEqual(l.call('foo','a',lambda:5),5)
   with self.assertRaises(AssertionError):l.call('foo','b',lambda:6)
   self.assertEqual(l.summary()['unresolved'],0);l.db.close()
 def test_received_tokens_not_input_to_source_key(self):
  tree=ast.parse((Path(__file__).parent/'kn_receiver.py').read_text());text=ast.dump(tree)
  self.assertNotIn('tokenizer',text);self.assertNotIn('source_archive',text);self.assertNotIn('source_tx',text)
if __name__=='__main__':unittest.main()
