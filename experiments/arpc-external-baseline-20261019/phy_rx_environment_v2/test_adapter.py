import ast,unittest
from adapter import ORIGINAL,EDITS,adapt
class ZipRevision(unittest.TestCase):
 def test_only_exact_interpreter_zip_added(self):
  self.assertEqual(EDITS[1][1].count("rtroot/'envs/frozen_python_base_v1/lib/python311.zip'"),1)
  self.assertNotIn("rtroot/'envs'",EDITS[1][1])
 def test_zip_does_not_change_any_calls(self):
  text=adapt(ORIGINAL.read_bytes());self.assertIn('python311.zip',text)
  before=ast.parse(ORIGINAL.read_text());after=ast.parse(text)
  def calls(t):
   return sorted(ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['transmit','receive','standard_noise','unpack'])
  self.assertEqual(calls(before),calls(after))
if __name__=='__main__':unittest.main(verbosity=2)
