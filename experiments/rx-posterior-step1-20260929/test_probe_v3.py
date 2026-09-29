"""CPU regression checks for v3 selection; synthetic arrays are not quality evidence."""
import unittest
import numpy as np
import rx_v3_b_helpers as h
class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg=h.algorithm();self.g=np.zeros((2,8,3,3,4,10,2));self.p=np.zeros((2,8,3,2,4,4))
        self.g[:,:,:,0,:,:,0]=.1;self.g[:,:,:,0,:,2:5,0]=.5
    def test_accuracy_not_log_probability_selects_MAP(self):
        self.g[:,:,:,2,1,2:5,0]=.9;self.g[:,:,:,2,3,2:5,1]=100
        s=h.select_arrays(self.g,self.p,self.cfg)[0]
        self.assertEqual(s['decision']['V']['lambda'],.75)
        self.assertEqual(s['ambiguity_scales_1based'],[3,4,5])
    def test_same_selection_for_static_and_VAR(self):
        self.g[:,:,:,1:3,3,2:5,0]=.8
        s=h.select_arrays(self.g,self.p,self.cfg)[0]
        self.assertEqual(s['decision']['V'],s['decision']['A2'])
    def test_empty_band_is_untuned(self):
        self.g[:,:,:,0,:,:,0]=.01
        s=h.select_arrays(self.g,self.p,self.cfg)[0]
        self.assertEqual(s['ambiguity_scales_1based'],[])
        self.assertEqual(s['decision']['V']['lambda'],1)
    def test_probability_grid_cannot_change_decisions(self):
        self.g[:,:,:,2,1,2:5,0]=.9
        a=h.select_arrays(self.g,self.p,self.cfg)
        self.p[:,:,:,1,3,3]=100
        b=h.select_arrays(self.g,self.p,self.cfg)
        self.assertEqual(a[0]['decision'],b[0]['decision'])
        self.assertNotEqual(a[0]['probability_only']['V'],b[0]['probability_only']['V'])
    def test_source_and_noise_pairing_not_flattened(self):
        a=h.select_arrays(self.g,self.p,self.cfg)
        b=h.select_arrays(self.g[::-1,:,::-1],self.p[::-1,:,::-1],self.cfg)
        self.assertEqual(a,b)
    def test_band_endpoints_and_tie(self):
        self.g[:,:,:,0,:,0,0]=.2;self.g[:,:,:,0,:,1,0]=.95
        s=h.select_arrays(self.g,self.p,self.cfg)[0]
        self.assertEqual(s['ambiguity_scales_1based'],[1,2,3,4,5])
        self.assertEqual(s['decision']['V']['lambda'],.5)
    def test_noise_conversion_all_levels(self):
        for x in self.cfg['levels']:
            self.assertAlmostEqual(x['eta']**2,10**(-x['snr_db']/10),places=14)
        self.assertEqual([x['snr_db'] for x in self.cfg['levels'] if x['name'].startswith('target')],[4,1,-2])
    def test_diagnostic_30_excluded_from_decision_levels(self):
        self.assertEqual(sum(x['role']=='decision' for x in self.cfg['levels']),7)
        self.assertEqual(self.cfg['levels'][-1]['role'],'TF_diagnostic_only')
if __name__=='__main__':unittest.main()
