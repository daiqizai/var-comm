import ast,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from rx_crosshost_finalize import matches_closed,failure_source,main_source,original
class Tests(unittest.TestCase):
 def fixture(self):
  who={'pid':1,'start_ticks':10,'host':'test-main'}
  return who,{'stage':'phy','status':'PASS','host':'test-main','observed_unix':100,'processes':[{**who,'verified_gone':True}]}
 def test_exact_identity(self):
  who,e=self.fixture();self.assertTrue(matches_closed(who,e,110));self.assertFalse(matches_closed({**who,'start_ticks':11},e,110))
 def test_stale_and_foreign_rejected(self):
  who,e=self.fixture()
  for now in [99,401]:
   with self.assertRaises(AssertionError):matches_closed(who,e,now)
  with self.assertRaises(AssertionError):matches_closed({**who,'host':'other'},e,110)
 def test_no_gone_false_reuse(self):
  who,e=self.fixture();e['processes'][0]['verified_gone']=False;self.assertFalse(matches_closed(who,e,110))
 def test_exact_reversible_guards(self):
  for module,name,adapt,old,new in [(original.previous,'failure',failure_source,'c.gone(','cross_gone('),(original,'main',main_source,'previous.failure()','crosshost_failure()')]:
   source=pathlib.Path(module.__file__).read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name)
   restored=ast.parse(adapt(source).replace(new,old)).body[0]
   self.assertEqual(ast.dump(fn,include_attributes=False),ast.dump(restored,include_attributes=False))
if __name__=='__main__':unittest.main()
