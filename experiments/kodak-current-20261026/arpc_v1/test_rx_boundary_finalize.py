import ast,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from rx_boundary_finalize import validate_denied_code_reads,adapted_finalizer,c

class Tests(unittest.TestCase):
 def example(self):return {'status':'COMPLETE','counts':{'model_load':1,'rx_prefix':72,'rx_render':72},'blocked_reads':['/fixture/native.py']*100}
 def test_exact(self):self.assertTrue(validate_denied_code_reads(self.example(),'/fixture/native.py'))
 def test_different_or_truth_path(self):
  for p in ['/fixture/source.npz','/fixture/tx/truth.npy','/fixture/native.py.bak']:
   d=self.example();d['blocked_reads'][99]=p
   with self.assertRaises(AssertionError):validate_denied_code_reads(d,'/fixture/native.py')
 def test_wrong_count(self):
  for n in [0,99,101]:
   d=self.example();d['blocked_reads']=['/fixture/native.py']*n
   with self.assertRaises(AssertionError):validate_denied_code_reads(d,'/fixture/native.py')
 def test_incomplete_calls(self):
  d=self.example();d['counts']['rx_render']=71
  with self.assertRaises(AssertionError):validate_denied_code_reads(d,'/fixture/native.py')
 def test_output_math_unchanged(self):
  s=(c.E/'control.py').read_text();old=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='finalize');new=ast.parse(adapted_finalizer(s)).body[0]
  # Everything from outputs={} onwards, including all arrays/hash/frame logic,
  # is the byte-identical original AST; only preceding lineage guards differ.
  def tail(n):
   pos=next(i for i,x in enumerate(n.body)if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='outputs'for t in x.targets))
   return ast.dump(ast.Module(body=n.body[pos:],type_ignores=[]),include_attributes=False)
  self.assertEqual(tail(old),tail(new))

if __name__=='__main__':unittest.main()
