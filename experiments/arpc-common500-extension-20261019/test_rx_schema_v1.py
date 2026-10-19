import unittest
from rx_prepare_v2 import normalized_clean_review,adapt
from execute_rx_v2 import adapt as adapt_execute
class Tests(unittest.TestCase):
    def fixture(self):
        p={'path':'completion','sha256':'x','bytes':1}
        r={'status':'PASS','all_processes_gone':True,'new_failure_files':[],'registration_sha256':'r','receipts':[p],'process_checks':[{'observed_gone':True}]}
        c={'status':'CLEAN_STATES_AUDITED_NO_METRICS_OR_WIRELESS','unresolved':0,'total_clean_states':700,'registration_sha256':'r','outputs':[]}
        return r,c,p
    def test_schema(self):
        r,c,p=self.fixture();n=normalized_clean_review(r,c,p);self.assertEqual(n['unresolved'],0);self.assertNotIn('unresolved',r)
    def test_missing_completion_pin(self):
        r,c,p=self.fixture();r['receipts']=[]
        with self.assertRaises(AssertionError):normalized_clean_review(r,c,p)
    def test_unresolved(self):
        r,c,p=self.fixture();c['unresolved']=1
        with self.assertRaises(AssertionError):normalized_clean_review(r,c,p)
    def test_live(self):
        r,c,p=self.fixture();r['process_checks'][0]['observed_gone']=False
        with self.assertRaises(AssertionError):normalized_clean_review(r,c,p)
    def test_pinned_adapter(self):
        from common import E
        self.assertIn('normalized_clean_review',adapt((E/'rx_control.py').read_bytes()))
        self.assertIn('rx_prepare_control_v2',adapt_execute((E/'execute_stage_v1.py').read_bytes()))
if __name__=='__main__':unittest.main()
