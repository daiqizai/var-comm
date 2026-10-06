import copy
import unittest
import numpy as np
import h_development_statistics as s


def fixture():
    ids = ['original-' + str(i) for i in range(100)]
    points = {'a': dict(snr_db=13, role='H_FIXED_M7_ATTRIBUTION', candidate_id='frozen-a'),
              'b': dict(snr_db=13, role='H_FIXED_M7_ATTRIBUTION', candidate_id='frozen-b')}
    rows = []
    for p, spec in points.items():
        for i in range(100):
            for ni, n in enumerate(s.SEEDS):
                rows.append(dict(point_id=p, **spec, source_id=ids[i], source_index=i, noise_seed=n,
                    N=1024, used_for_selection=False, holdout_used=False,
                    value=i + ni / 2 + (2 if p == 'a' else 0), convnext_source_prediction=i))
    return ids, points, rows


class StatisticsTests(unittest.TestCase):
    def test_pairing_cancels_source_and_noise_variation(self):
        ids, points, rows = fixture()
        out = s.summarize(rows[::-1], ids, points, ['value'], [('a', 'b')])
        paired = out['paired'][0]
        self.assertEqual((paired['mean'], paired['ci_low'], paired['ci_high']), (2, 2, 2))
        self.assertEqual(paired['bootstrap_seed'], 2026100605)
        self.assertEqual(paired['frame_count'], 300)
        self.assertEqual(len(out['source_means']), 200)
        self.assertFalse(out['policy_selection'])

    def test_frozen_bootstrap_exact_source_resamples(self):
        boot = s.draws()
        np.testing.assert_array_equal(boot, np.random.default_rng(2026100605).integers(0, 100, (10000, 100)))
        self.assertFalse(np.array_equal(boot, np.random.default_rng(20261002).integers(0, 100, (10000, 100))))

    def test_missing_duplicate_or_old_noise_rejected(self):
        ids, points, rows = fixture()
        for altered in (rows[:-1], rows + [rows[0]], [dict(rows[0], noise_seed=2001)] + rows[1:]):
            with self.assertRaises(ValueError):
                s.validate(altered, ids, points, ['value'])

    def test_mismatched_identity_or_nonindependent_rows_rejected(self):
        ids, points, rows = fixture()
        for change in (dict(source_id='wrong'), dict(candidate_id='reselected'), dict(used_for_selection=True),
                       dict(holdout_used=True), dict(convnext_source_prediction=100), dict(value=float('nan'))):
            with self.assertRaises(ValueError):
                s.validate([dict(rows[0], **change)] + rows[1:], ids, points, ['value'])

    def test_cross_snr_or_self_pair_not_computed(self):
        ids, points, rows = fixture()
        with self.assertRaises(ValueError):
            s.summarize(rows, ids, points, ['value'], [('a', 'a')])
        points['b']['snr_db'] = 19
        rows = [dict(r, snr_db=19) if r['point_id'] == 'b' else r for r in rows]
        with self.assertRaises(ValueError):
            s.summarize(rows, ids, points, ['value'], [('a', 'b')])

    def test_success_needs_two_snrs_and_all_independent_intervals(self):
        rows = []
        for snr in (13, 19):
            for metric, lo, hi in [('psnr_db', .1, .2), ('lpips_alex', -.02, 0),
                                  ('clip_image_cosine', 0, .02), ('convnext_source_prediction_agreement', 0, .03)]:
                rows.append(dict(snr_db=snr, metric=metric, method='frozenH', reference='frozenMAIN', ci_low=lo, ci_high=hi))
        self.assertEqual(s.independent_success(rows)['overall'], 'PASS')
        self.assertEqual(s.independent_success(rows[:-1])['overall'], 'INCOMPLETE')
        changed = copy.deepcopy(rows)
        changed[3]['ci_low'] = -.001
        self.assertEqual(s.independent_success(changed)['overall'], 'NOT_ESTABLISHED')
        changed = copy.deepcopy(rows)
        changed[0]['ci_low'] = 0
        self.assertEqual(s.independent_success(changed)['overall'], 'NOT_ESTABLISHED')

    def test_dino_only_never_meets_image_content_rule(self):
        rows = [dict(snr_db=n, metric='dinov2_vitl14_cosine', method='h', reference='main', ci_low=.1, ci_high=.2)
                for n in (13, 19)]
        self.assertEqual(s.independent_success(rows)['overall'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
