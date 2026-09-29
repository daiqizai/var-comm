"""CPU tests for the revised statistical design; no GPU quality claims."""
import unittest, math
import torch
import probe_v2 as p
class RevisionTests(unittest.TestCase):
    def test_noiseless_limit_and_nonzero_noise_scaling(self):
        stats={'unit_noise_var':torch.tensor([.1,.3]),'s2':torch.tensor([.2,.9])}
        self.assertTrue(torch.equal(p.variance(stats,0,0),torch.zeros(2,dtype=torch.float64)))
        torch.testing.assert_close(p.variance(stats,.1,0),stats['unit_noise_var'].double()*.01)
        torch.testing.assert_close(p.variance(stats,.1,.2),stats['unit_noise_var'].double()*.01+.2*stats['s2'].double())
    def test_grid_fairness_and_deterministic_ties(self):
        self.assertEqual(len(p.parameters()),20)
        self.assertEqual(p.parameters()[p.choose([0.]*20)],(0,.5))
        scores=[-3.]*20;scores[7]=-1.
        self.assertEqual(p.choose(scores),7)
        with self.assertRaises(AssertionError):p.choose([float('nan')]*20)
    def test_scalar_variance_cannot_change_uniform_prior_map(self):
        d=torch.tensor([[0.,1.,2.],[3.,1.,0.]],dtype=torch.float64)
        for v in [.00001,.01,1.,100.]:
            for lam in [.5,.75,1.,1.5]:
                self.assertTrue(torch.equal((-d/(2*v)+lam*math.log(1/3)).argmax(-1),d.argmin(-1)))
    def test_proper_score_rewards_true_distribution(self):
        truth=torch.tensor([.7,.2,.1],dtype=torch.float64)
        calibrated=(truth*truth.log()).sum()
        overconfident=(truth*torch.tensor([.98,.01,.01],dtype=torch.float64).log()).sum()
        uniform=(truth*torch.full((3,),1/3,dtype=torch.float64).log()).sum()
        self.assertGreater(float(calibrated),float(overconfident))
        self.assertGreater(float(calibrated),float(uniform))
    def test_new_design_preserves_original_thresholds_and_noise_targets(self):
        new=p.read(p.DESIGN);old=p.read(p.base.CONFIG)
        for key in ['gates','difficulty_targets','oracle_snrs_db','development_noise_seeds','calibration_noise_seeds']:
            self.assertEqual(new[key],old[key])
        self.assertEqual(new['beta_grid'],[0,.05,.1,.2,.5])
        self.assertEqual(new['lambda_grid'],[.5,.75,1,1.5])
        self.assertEqual(new['engineering_modes'],['TF','CL'])
        self.assertGreaterEqual(new['engineering_snr_db'],30)
if __name__=='__main__':unittest.main()
