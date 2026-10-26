import ast,importlib.util,pathlib,unittest,hashlib
E=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('resource',E/'resource_continuation.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class TestResource(unittest.TestCase):
 def test_exact_resource_only(self):
  raw=(E/'control.py').read_bytes();text=m.adapt_worker(raw);self.assertIn("gpu['utilization']<=80",text);self.assertIn("gpu['free_mib']>=65536",text);self.assertNotIn("gpu['utilization']<=25",text)
 def test_native_guard(self):
  text=m.adapt_worker((E/'control.py').read_bytes());self.assertIn("not(run/'native.launch.json').exists()",text);self.assertIn("not(run/'tx').exists()",text)
 def test_bad_pin_rejected(self):
  with self.assertRaises(AssertionError):m.adapt_worker((E/'control.py').read_bytes()+b' ')
 def test_native_calls_unchanged(self):
  raw=(E/'control.py').read_text();old=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name=='worker');new=ast.parse(m.adapt_worker(raw.encode())).body[0]
  calls=lambda t:[ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in ['stage_wait','audit_source']]
  self.assertEqual(calls(old),calls(new))
if __name__=='__main__':unittest.main()
