import pathlib,unittest
import native
from cache import match,extend_cache
class EvalRX(unittest.TestCase):
 def test_evaluation_cap(self):
  jobs=[{'key':str(i),'generation_seed':0,'paid_profile':4,'input':{'sha256':str(i),'bytes':20}} for i in range(701)]
  with self.assertRaises(AssertionError):match(jobs,[])
 def test_closed_actual_cache_only(self):
  pin={'sha256':'a','bytes':10}
  states=extend_cache([], [{'status':'IMAGE_READY','endpoint':4,'input':pin,'image':{'sha256':'im','bytes':90}}, {'status':'SOURCE_PARSE_FAIL','image':None}])
  a,b=match([{'key':'j','generation_seed':0,'paid_profile':4,'input':pin}],states)
  self.assertEqual(len(a),1);self.assertFalse(b)
 def test_wrong_received_input_never_replaced_by_truth(self):
  states=extend_cache([], [{'status':'IMAGE_READY','endpoint':4,'input':{'sha256':'truth','bytes':10},'image':{'sha256':'im','bytes':90}}])
  a,b=match([{'key':'j','generation_seed':0,'paid_profile':4,'input':{'sha256':'wrong','bytes':10}}],states)
  self.assertFalse(a);self.assertEqual(len(b),1)
 def test_native_cap_and_math(self):
  text=native.adapt((pathlib.Path(__file__).parents[1]/'a1_native.py').read_bytes())
  self.assertIn('len(jobs)<=700',text);self.assertIn('except SourceParseError',text)
if __name__=='__main__':unittest.main()
