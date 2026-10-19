"""Synthetic closure tests: no models, PHY kernels, real packets or probes."""
import json,pathlib,tempfile,unittest
from link_finalize_v1 import ledger,boundary_proof,audit_source

class Tests(unittest.TestCase):
    def check_ledger(self,rows,counts):
        with tempfile.TemporaryDirectory()as t:
            p=pathlib.Path(t)/'calls.jsonl';p.write_text('\n'.join(json.dumps(r)for r in rows));return ledger(p,counts)
    def rows(self):return [{'event':1,'state':'RESERVED','kind':'header'},{'event':1,'state':'COMPLETE'}]
    def test_complete(self):self.assertEqual(self.check_ledger(self.rows(),{'header':1}),{'header':1})
    def test_unresolved(self):
        with self.assertRaises(AssertionError):self.check_ledger(self.rows()[:1],{'header':1})
    def test_failed(self):
        a=self.rows();a[1]['state']='FAILED'
        with self.assertRaises(AssertionError):self.check_ledger(a,{'header':1})
    def test_duplicate(self):
        with self.assertRaises(AssertionError):self.check_ledger(self.rows()*2,{'header':2})
    def test_count(self):
        with self.assertRaises(AssertionError):self.check_ledger(self.rows(),{'header':2})
    def test_boundary(self):
        text="""def install_rx_boundary(allowed):
 def audit(event,args):
  if not any(x for x in allowed):
   blocked.append(str(p))
   raise PermissionError('denied')
 sys.addaudithook(audit)
 return blocked
"""
        a=boundary_proof(text);self.assertIsNone(a['original_blocked_paths']);self.assertFalse(a['attempt_free_gate_passed'])
        with self.assertRaises(AssertionError):boundary_proof(text.replace("raise PermissionError('denied')","return None"))
    def test_original_audit_adapter(self):
        from common import PE
        s=audit_source((PE/'a2b_link_v1/worker.py').read_bytes())
        self.assertIn('rx_receipt.json',s);self.assertIn('out_of_allowlist_open_raises_before_returning_data',s)
        self.assertNotIn('rt.receive',s)

if __name__=='__main__':unittest.main()
