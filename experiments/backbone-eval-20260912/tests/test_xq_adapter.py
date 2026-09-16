import sys, pathlib, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import torch
from xq_adapter import SCALES, token_metadata, token_hash, ModelAdapter, compare_states
class TestBudget(unittest.TestCase):
 def setUp(self): self.tokens=[[torch.zeros(1,p*p,dtype=torch.long) for p in SCALES] for _ in range(2)]
 def test_full_and_prefix(self):
  for m,b,n in [(8,2424,202),(9,3960,330),(10,6864,572)]:
   d=token_metadata(self.tokens,m);self.assertEqual(d['raw_bits'],b);self.assertEqual(d['index_count'],n)
 def test_branch_is_not_optional(self):
  with self.assertRaises(ValueError):token_metadata(self.tokens[:1],10)
 def test_no_joint_vocabulary_accounting(self):
  self.tokens[1][-1][0,0]=4096
  with self.assertRaises(ValueError):token_metadata(self.tokens,10)
 def test_no_space_position_clipping(self):
  self.tokens[1][7]=self.tokens[1][7][:,:-1]
  with self.assertRaises(ValueError):token_metadata(self.tokens,8)
 def test_truncated_metadata(self):self.assertEqual(token_metadata([x[:8] for x in self.tokens],8)['raw_bits'],2424)
 def test_receiver_true_suffix_forbidden(self):
  adapter=ModelAdapter.__new__(ModelAdapter)
  with self.assertRaisesRegex(ValueError,'ONLY the prefix'):adapter.complete(self.tokens,torch.tensor([0]),8)
 def test_each_branch_hash_covered(self):
  a=token_hash(self.tokens);self.tokens[1][-1][0,0]=1;self.assertNotEqual(a,token_hash(self.tokens))
 def test_mismatched_checkpoint_not_silently_accepted(self):
  a={'x':torch.ones(2)};b={'x':torch.zeros(2)}
  self.assertFalse(compare_states(a,b)['exactly_equal']);self.assertTrue(compare_states(a,a)['exactly_equal'])
if __name__=='__main__':unittest.main()
