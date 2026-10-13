import unittest
import contract as c


class DesignTests(unittest.TestCase):
    def test_selection_is_source_id_only_and_permutation_invariant(self):
        rows = [dict(source_id=f's{i:04}', bogus_quality=-i) for i in range(1000)]
        first = c.select_cal200(rows)
        second = c.select_cal200(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 200)
        with self.assertRaises(ValueError): c.select_cal200(rows[:200])

    def test_noise_pairing_has_no_true_snr_or_configuration_argument(self):
        import numpy as np
        a = c.standard_noise('source', 7, 4101, 'cal200')
        self.assertEqual(a.shape, (1024, 2))
        self.assertEqual(a.dtype, np.float64)
        np.testing.assert_array_equal(a, c.standard_noise('source', 7, 4101, 'cal200'))
        self.assertFalse(np.array_equal(a, c.standard_noise('source', 10, 4101, 'cal200')))
        self.assertFalse(np.array_equal(a, c.standard_noise('another', 7, 4101, 'cal200')))
        with self.assertRaises(ValueError): c.standard_noise('source', 7, 2001, 'cal200')

    def test_counter_and_grid(self):
        self.assertEqual(c.true_snrs(7), (4, 7, 10))
        self.assertEqual(c.true_snrs(10), (7, 10, 13))
        values = {c.public_counter(i, h, seed, 'cal200')
                  for i in range(200) for h in c.HATS for seed in c.CAL_SEEDS}
        self.assertEqual(values, set(range(1200)))

    def test_gate_cannot_hide_negative_hat_or_replace_strong_reference(self):
        scores = {'U_two': {7: .55, 10: .65}, 'S_robust': {7: .52, 10: .64},
                  'S_back3': {7: .52, 10: .62}}
        self.assertTrue(c.calibration_gate(scores)['r2_authorized'])
        scores['S_robust'][10] = .65
        self.assertFalse(c.calibration_gate(scores)['r2_authorized'])
        scores['S_robust'][10] = .66
        self.assertEqual(c.calibration_gate(scores)['status'], 'ONE_POINT_PROMISING')

    def test_mixture_and_lpips_interaction(self):
        self.assertAlmostEqual(c.design_score(.8, .4), .7)
        self.assertAlmostEqual(c.interaction(.3, .4, .5, .55, 'lpips_alex'), .05)
        self.assertAlmostEqual(c.interaction(.3, .4, .5, .55, 'psnr_db'), -.05)
        with self.assertRaises(ValueError): c.design_score(float('nan'), .3)

    def test_payload_budgets_derive_from_fixed_grid(self):
        self.assertEqual(c.R1_CAPS['physical_frames'], 200*2*3*3*10)
        self.assertEqual(c.R1_CAPS['logical_var_images'], 200*2*3*3*13)
        self.assertEqual(c.R2_CAPS['physical_frames'], 100*6*3*6)
        self.assertEqual(c.R2_CAPS['logical_images'], 100*6*3*8)


if __name__ == '__main__': unittest.main()
