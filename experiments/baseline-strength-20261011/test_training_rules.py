import unittest
from training_rules import learning_rate,stability

class Rules(unittest.TestCase):
    def test_p_boundaries(self):
        for step,lr in [(40001,2e-4),(60000,2e-4),(60001,1e-4),(70000,1e-4),(70001,3e-5),(100000,3e-5)]:
            self.assertEqual(learning_rate('P',step,40000,2e-4),lr)
        for step in [40000,100001]:
            with self.assertRaises(ValueError):learning_rate('P',step,40000,2e-4)
    def test_swin_added_budget_and_no_lr_raise(self):
        for u,lr in [(1,1e-4),(40000,1e-4),(40001,3e-5),(60000,3e-5),(60001,1e-5),(120000,1e-5)]:
            self.assertEqual(learning_rate('Swin',80000+u,80000,1e-4),lr)
            self.assertEqual(learning_rate('Swin',80000+u,80000,1e-5),1e-5)
        with self.assertRaises(ValueError):learning_rate('Swin',200001,80000,1e-4)
    def test_deterioration_does_not_count_as_plateau(self):
        h=[dict(step=t,objective=j,cells={'N1024_snr1':j}) for t,j in [(70000,1.),(75000,1.01),(80000,1.02)]]
        self.assertFalse(stability(h,80000,5000)['stable'])
    def test_every_cell_and_both_windows_required(self):
        h=[dict(step=t,objective=1.,cells={'bad':j,'good':1.}) for t,j in [(70000,1.),(75000,1.01),(80000,1.01)]]
        self.assertFalse(stability(h,80000,5000)['stable'])
        with self.assertRaises(ValueError):stability(h[1:],80000,5000)
    def test_exact_plateau_and_earliest_selection(self):
        h=[dict(step=t,objective=1.,cells={'N1024_snr1':1.}) for t in [70000,75000,80000]]
        self.assertTrue(stability(h,80000,5000)['stable'])
        self.assertEqual(min(h,key=lambda r:(r['objective'],r['step']))['step'],70000)

if __name__=='__main__':unittest.main()
