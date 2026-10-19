import unittest
from rx_prepare_v3 import choose_gpu,adapt
from execute_rx_v3 import adapt as execute_adapt
from common import E
class Tests(unittest.TestCase):
    def row(self,i,free,util):return dict(index=i,free_mib=free,utilization=util,uuid=str(i))
    def test_best_eligible(self):
        self.assertEqual(choose_gpu([self.row(1,90000,26),self.row(2,70000,1),self.row(6,72489,2)])['index'],6)
    def test_boundary(self):self.assertEqual(choose_gpu([self.row(0,65536,25)])['index'],0)
    def test_reject_low_memory(self):
        with self.assertRaises(AssertionError):choose_gpu([self.row(0,65535,0)])
    def test_reject_busy(self):
        with self.assertRaises(AssertionError):choose_gpu([self.row(0,90000,26)])
    def test_tie(self):self.assertEqual(choose_gpu([self.row(7,70000,0),self.row(2,70000,0)])['index'],2)
    def test_source_and_run_preservation(self):self.assertIn('gpu=select_device(cfg)',adapt((E/'rx_control.py').read_bytes()))
    def test_dispatch_preservation(self):
        s=execute_adapt((E/'execute_stage_v1.py').read_bytes());self.assertIn('rx_dispatch_control_v1',s);self.assertIn('rx_prepare_control_v3',s);self.assertIn("else'rx_control.py'",s)
if __name__=='__main__':unittest.main()
