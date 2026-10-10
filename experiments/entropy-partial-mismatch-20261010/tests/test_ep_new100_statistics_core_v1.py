"""CPU engineering with fake intervals only; no experiment or bootstrap is run."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ep_new100_statistics_core_v1 as core
c = core.confirmation


def fixture():
    rows = []
    for i in range(100):
        for s in c.SNRS:
            for ni, n in enumerate(c.SEEDS):
                for m in c.METHODS:
                    partial = m.endswith('PARTIAL')
                    entropy = m.startswith('EC_')
                    delta = (0.2 if entropy else 0.1) if partial else 0.
                    rows.append(dict(frame_index=len(rows), N=1024, source_index=i,
                        source_id=f'source-{i:03d}', snr_db=s, noise_seed=n, method=m,
                        public_frame_counter=c.counter(i, s, n), receiver_contract=c.method_contract(m),
                        psnr_db=10. + i / 100 + ni / 10 + s / 10 + delta,
                        lpips_alex=.9 - i / 1000 - ni / 100 - delta,
                        dinov2_vitl14_cosine=.3 + i / 1000 + ni / 100 + delta,
                        convnext_top1_source_prediction=int(i < (70 if partial else 50))))
    return rows


class FakePublished:
    def __init__(self):
        self.draw_calls = 0
        self.interval_vectors = []

    def draws(self):
        self.draw_calls += 1
        return np.zeros((10000, 100), dtype=np.int64)

    def interval(self, vector, samples):
        self.interval_vectors.append(vector.copy())
        value = float(np.mean(vector, dtype=np.float64))
        return dict(mean=value, ci_low=value - .0123, ci_high=value + .0123,
                    source_count=100, noise_count=3, frame_count=1500,
                    bootstrap_seed=core.SEED, bootstrap_replicates=core.REPLICATES)


class StatisticsTests(unittest.TestCase):
    def test_fixed_scientific_definitions(self):
        summaries, pairs = core.definitions()
        self.assertEqual((len(summaries), len(pairs)), (48, 24))
        self.assertEqual({(p['method'], p['reference']) for p in pairs}, set(c.CONTRASTS))
        self.assertEqual(set(core.CAPS), {'draw_matrix', 'summary_intervals', 'paired_intervals'})

    def test_exact_original_mean_operation(self):
        rows = fixture()
        # This sum distinguishes np.mean from compensated summation.
        for r in rows:
            if r['method'] == 'RAW_WHOLE' and r['source_index'] == 0 and r['snr_db'] == 4:
                r['psnr_db'] = [1e16, 1., -1e16][c.SEEDS.index(r['noise_seed'])]
        vectors, table, ids = core.source_vectors(rows, rows)
        self.assertEqual(vectors['RAW_WHOLE', 4, 'psnr_db'][0], 0.)
        self.assertEqual(len(table), 4800)
        self.assertEqual(len(ids), 100)

    def test_one_draw_and_unchanged_paired_sign_units(self):
        rows = fixture(); stat = FakePublished(); events = []
        result = core.summarize(rows, rows, stat, lambda *x: events.append(x))
        self.assertEqual(stat.draw_calls, 1)
        self.assertEqual((len(result['summary']), len(result['paired'])), (48, 24))
        self.assertEqual(len(result['source_paired_differences']), 2400)
        lpips = next(r for r in result['paired'] if r['metric'] == 'lpips_alex')
        self.assertLess(lpips['mean'], 0.)
        self.assertEqual(lpips['improvement_direction'], 'negative')
        agree = next(r for r in result['paired'] if r['metric'] == c.METRICS[-1])
        self.assertAlmostEqual(agree['mean'], .2)
        self.assertAlmostEqual(agree['display_mean'], 20.)
        self.assertEqual(agree['display_unit'], 'percentage_points')
        self.assertAlmostEqual(agree['display_ci_low'], 100 * agree['ci_low'])
        self.assertTrue(any(np.count_nonzero(v == 1.) == 20 and np.count_nonzero(v == 0.) == 80
                            for v in stat.interval_vectors))
        reserved = {(k, i) for phase, k, i, detail in events if phase == 'reserved'}
        completed = {(k, i) for phase, k, i, detail in events if phase == 'completed'}
        self.assertEqual(reserved, completed)
        self.assertEqual(len(reserved), 1 + len(stat.interval_vectors))

    def test_zero_deltas_retained_without_interval_calls(self):
        rows = fixture()
        for offset in range(0, len(rows), 4):
            for base in (0, 2):
                for metric in c.METRICS:
                    rows[offset + base + 1][metric] = rows[offset + base][metric]
        stat = FakePublished()
        result = core.summarize(rows, rows, stat, lambda *x: None)
        self.assertEqual(result['actual_calls']['paired_intervals'], 0)
        for row in result['paired']:
            self.assertEqual((row['mean'], row['ci_low'], row['ci_high']), (0., 0., 0.))
        self.assertEqual(len(result['paired']), 24)

    def test_missing_duplicate_reordered_nonfinite_and_nonbinary_rejected(self):
        expected = fixture()
        cases = [expected[:-1], [expected[1], expected[0]] + expected[2:]]
        for field, value in [('lpips_alex', float('nan')),
                             ('convnext_top1_source_prediction', .7),
                             ('noise_seed', 6201), ('source_id', 'foreign')]:
            altered = copy.deepcopy(expected); altered[0][field] = value; cases.append(altered)
        altered = copy.deepcopy(expected); altered[1] = altered[0]; cases.append(altered)
        for rows in cases:
            stat = FakePublished()
            with self.assertRaises((RuntimeError, ValueError)):
                core.summarize(rows, expected, stat, lambda *x: None)
            self.assertEqual(stat.draw_calls, 0)

    def test_wrong_published_metadata_is_rejected(self):
        class Wrong(FakePublished):
            def interval(self, vector, samples):
                return dict(super().interval(vector, samples), bootstrap_seed=123)
        rows = fixture()
        with self.assertRaises((RuntimeError, ValueError)):
            core.summarize(rows, rows, Wrong(), lambda *x: None)


if __name__ == '__main__':
    unittest.main()
