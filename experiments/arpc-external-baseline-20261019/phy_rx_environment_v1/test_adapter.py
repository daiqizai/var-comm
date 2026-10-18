import ast,unittest
from adapter import ORIGINAL,EDITS,adapt
class Revision(unittest.TestCase):
    def test_exact_reversible(self):
        data=ORIGINAL.read_bytes();text=adapt(data)
        for a,b in reversed(EDITS):text=text.replace(b,a)
        self.assertEqual(text.encode(),data)
    def test_science_calls_unchanged(self):
        a=ast.parse(ORIGINAL.read_text());b=ast.parse(adapt(ORIGINAL.read_bytes()))
        def call_signatures(t):
            return sorted(ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['call','transmit','receive','standard_noise','unpack'])
        self.assertEqual(call_signatures(a),call_signatures(b))
    def test_source_rejected(self):
        with self.assertRaises(AssertionError):adapt(ORIGINAL.read_bytes()+b'\n')
    def test_exact_roots_only(self):
        changes=EDITS[1][1];self.assertIn("frozen_python_base_v1/lib/python3.11",changes)
        self.assertNotIn("rtroot/'assets'",changes);self.assertNotIn("link.R,",changes)
if __name__=='__main__':unittest.main(verbosity=2)
