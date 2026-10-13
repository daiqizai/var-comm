"""Synthetic engineering checks; no real images, models, channels, or statistics."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ENTRY = Path(__file__).with_name('postprocess.py')
spec = importlib.util.spec_from_file_location('kodak_final_postprocess_tested', ENTRY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def synthetic_rows():
    ids = [f'synthetic/source{i:02d}' for i in range(24)]
    old = [dict(source_index=i, reference_sha256=f'synthetic_rgb_{i}', source_prediction='3',
                preprocessing_id='synthetic_256', metric_backend_identity='synthetic_metric_backend')
           for i in range(24)]
    rows = []
    for method in module.NEW:
        for snr in module.SNRS:
            for index in range(24):
                for noise in module.NOISES:
                    for metric, value in zip(module.METRICS, (20., .2, .6, 1.)):
                        rows.append(dict(method=method, source_index=index, source_id=ids[index],
                                         snr_db=snr, noise_seed=noise,
                                         status='FIXED_FAILURE_OUTPUT' if index == 0 else 'RECEIVED_OK',
                                         N=1024, population='Kodak24', reference_sha256=f'synthetic_rgb_{index}',
                                         reconstruction_sha256=f'synthetic_rx_{method}_{index}_{noise}_{snr}',
                                         reconstruction_archive_sha256=f'synthetic_npz_{method}_{index}_{noise}_{snr}',
                                         preprocessing_id='synthetic_256', metric_backend_identity='synthetic_current_runtime',
                                         original_metric_backend_identity='synthetic_metric_backend',
                                         true_class_label_available='False', source_prediction='3',
                                         reconstruction_prediction='3', frame_level_noise_pairing_claimed='False',
                                         metric=metric, value=value))
    done = dict(status='KODAK24_FINAL_BASELINES_FOUR_METRICS_COMPLETE_V1', source_count=24,
                frame_count=432, metric_rows=1728, N=1024, source_ids=ids,
                metrics=list(module.METRICS), methods=list(module.NEW),
                metric_backend_identity='synthetic_current_runtime',
                original_metric_backend_identity='synthetic_metric_backend')
    return ids, old, rows, done


class SyntheticEngineeringTests(unittest.TestCase):
    def test_failed_layout_recovers_completed_statistics_without_bootstrap(self):
        with tempfile.TemporaryDirectory(dir=ENTRY.parent) as directory:
            out = Path(directory)
            summary = [dict(method=m, snr_db=s, metric=k, mean=.2, ci_low=.1, ci_high=.3,
                            source_count=24, noise_count=3, frame_count=72,
                            bootstrap_seed=module.SEED, bootstrap_replicates=module.REPLICATES,
                            bootstrap_unit='source after3-noise mean')
                       for m in module.METHODS for s in module.SNRS for k in module.METRICS]
            paired = [dict(method='VAR_UNCONDITIONAL', reference=m, snr_db=s, metric=k,
                           mean=.1, ci_low=-.1, ci_high=.3)
                      for m in module.NEW for s in module.SNRS for k in module.METRICS]
            module.write_csv(out / 'summary.csv', summary)
            module.write_csv(out / 'paired_new.csv', paired)
            source = out / 'synthetic_input.txt'
            source.write_text('Synthetic engineering input only', encoding='utf8')
            checkpoint = dict(status='KODAK24_FINAL_BASELINE_SOURCE_STATISTICS_COMPLETE_V1',
                              script_sha256=module.sha(ENTRY), inputs={str(source): module.sha(source)},
                              outputs={str(out / name): module.sha(out / name)
                                       for name in ('summary.csv', 'paired_new.csv')},
                              interval_function_calls=48, old_interval_recomputations=0)
            module.write_json(out / 'statistics_completion.json', checkpoint)
            initial = {name: (out / name).read_bytes() for name in ('summary.csv', 'paired_new.csv')}

            def synthetic_render(target, table):
                artifact = target / 'synthetic_layout_check.txt'
                artifact.write_text('A placeholder used only in an engineering test', encoding='utf8')
                return [artifact]

            with patch.object(module, 'new_statistics', side_effect=AssertionError('Bootstrap must not run')):
                with patch.object(module, 'plot', side_effect=ValueError('Synthetic layout failure')):
                    with self.assertRaisesRegex(ValueError, 'Synthetic layout failure'):
                        module.render_existing(out)
                self.assertFalse((out / 'completion.json').exists())
                with patch.object(module, 'plot', side_effect=synthetic_render):
                    result = module.render_existing(out)
            self.assertEqual(result['layout_new_bootstrap_calls'], 0)
            self.assertEqual(result['interval_function_calls'], 48)
            self.assertTrue((out / 'completion.json').is_file())
            self.assertEqual({name: (out / name).read_bytes() for name in initial}, initial)
            with self.assertRaisesRegex(ValueError, 'Already completed'):
                module.render_existing(out)

    def test_complete_grid_keeps_failure_outputs_and_rejects_duplicate_or_foreign_source(self):
        ids, old, rows, done = synthetic_rows()
        arrays, frames = module.validate_new(rows, done, ids, old)
        self.assertEqual(len(arrays), 24)
        self.assertEqual(len(frames), 432)
        self.assertEqual(sum(v[1] == 'FIXED_FAILURE_OUTPUT' for v in frames.values()), 18)
        duplicated = copy.deepcopy(rows)
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate or unexpected'):
            module.validate_new(duplicated, done, ids, old)
        foreign = copy.deepcopy(rows)
        foreign[-1]['reference_sha256'] = 'wrong_source_pixels'
        with self.assertRaisesRegex(ValueError, 'identities must match'):
            module.validate_new(foreign, done, ids, old)

    def test_known_signed_source_effect_uses_only_new_intervals(self):
        original = module.load_original()
        base = np.linspace(.1, .4, 24)
        arrays = {}
        for snr in module.SNRS:
            for metric in module.METRICS:
                arrays['VAR_UNCONDITIONAL', snr, metric] = base.copy()
                arrays['P100K', snr, metric] = base + .25
                arrays['SWIN200K', snr, metric] = base - .5
        calls = []
        native_interval = original.interval

        def tracked(values, draws):
            calls.append(values.copy())
            return native_interval(values, draws)

        original.interval = tracked
        summary, paired, means, draw_sha = module.new_statistics(original, list(range(24)), arrays)
        self.assertEqual((len(summary), len(paired), len(means), len(calls)), (24, 24, 576, 48))
        self.assertEqual({r['method'] for r in summary}, {'P100K', 'SWIN200K'})
        for row in paired:
            expected = -.25 if row['reference'] == 'P100K' else .5
            for field in ('mean', 'ci_low', 'ci_high'):
                self.assertAlmostEqual(row[field], expected, places=14)
        self.assertEqual(draw_sha, '552e5b5aebdfa237ac3a72f9c1329dbdcff67335f4f0ee1fa9d1dc0275ed42d8')

    def test_historical_number_text_and_line_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=ENTRY.parent) as directory:
            src, out = (Path(directory) / name for name in ('original.csv', 'combined.csv'))
            raw = (b'method,mean,ci_low,ci_high\r\n'
                   b'VAR_UNCONDITIONAL,0.12345678901234567,0.10000000000000001,0.2\r\n'
                   b'P1024,0.5,0.4,0.6\r\n'
                   b'BPG_ADAPTIVE,2.5e-05,1.2e-05,4.5e-05\r\n')
            src.write_bytes(raw)
            subset, count = module.csv_subset_bytes(src, lambda r: r['method'] in module.ANCHORS)
            self.assertEqual(count, 2)
            expected = raw.splitlines(keepends=True)
            self.assertEqual(subset, b''.join((expected[0], expected[1], expected[3])))
            module.append_csv_preserving_old(out, subset,
                                            [dict(method='P100K', mean=.6, ci_low=.4, ci_high=.7)])
            self.assertTrue(out.read_bytes().startswith(subset))
            self.assertEqual(src.read_bytes(), raw)


if __name__ == '__main__':
    unittest.main()
