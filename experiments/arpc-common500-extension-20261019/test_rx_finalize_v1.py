import unittest
from rx_finalize_v1 import validate_denied,audit_source,boundary_proof,WRAPPER_SHA
from common import E,PE
class Tests(unittest.TestCase):
    def pin(self):return {'path':str(E/'rx_native.py'),'sha256':WRAPPER_SHA}
    def test_exact_public_file(self):self.assertEqual(sum(validate_denied([str(E/'rx_native.py')]*100,E/'rx_native.py',self.pin()).values()),100)
    def test_private_path_rejected(self):
        with self.assertRaises(AssertionError):validate_denied([str(E/'rx_native.py')]*99+[str(E/'source_native.py')],E/'rx_native.py',self.pin())
    def test_empty_rejected(self):
        with self.assertRaises(AssertionError):validate_denied([],E/'rx_native.py',self.pin())
    def test_changed_count_rejected(self):
        with self.assertRaises(AssertionError):validate_denied([str(E/'rx_native.py')]*101,E/'rx_native.py',self.pin())
    def test_changed_source_rejected(self):
        p=self.pin();p['sha256']='0'*64
        with self.assertRaises(AssertionError):validate_denied([str(E/'rx_native.py')]*100,E/'rx_native.py',p)
    def test_exact_audit_edit(self):self.assertIn('approved_public_source_denials',audit_source((PE/'a3_rx_v1/audit.py').read_bytes()))
    def test_boundary_raises(self):self.assertTrue(boundary_proof((PE/'a1_native.py').read_text())['out_of_allowlist_open_rejected_before_data'])
if __name__=='__main__':unittest.main()
