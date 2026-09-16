import copy
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

from wetok_comm.interface_evaluation import METRICS, comparisons, feature_diagnostics, reference_names, statistics, validate_grid
from wetok_comm.interface_study import interface_definitions, receiver_features
from wetok_comm.native import hard_native_st


class InterfaceEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = yaml.safe_load((EXPERIMENT / 'configs/interface_study.yaml').read_text())
        cls.base = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        cls.base['evaluation']['bootstrap_resamples'] = 32
        definitions = interface_definitions(cls.study)
        names = list(definitions) + reference_names(cls.study)
        cls.rows = []
        for index in range(100):
            for snr in cls.base['evaluation']['snrs_db']:
                for seed in cls.base['evaluation']['noise_seeds']:
                    for position, name in enumerate(names):
                        header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                        interface = definitions[name]['interface'] if name in definitions else 'frozen_reference'
                        kind = ('continuous_bounded_features' if interface == 'continuous_mean' else 'native_signs') if name in definitions else 'historical_RGB_reference'
                        value = position * (1 + index / 100) + snr / 20 + (seed - 2000) / 10
                        cls.rows.append({'image_index': index, 'image_id': f'synthetic_{index:03d}', 'snr_db': snr,
                            'seed': seed, 'arm': name, 'noise_sha256': hashlib.sha256(f'{index}_{seed}'.encode()).hexdigest(),
                            'total_complex_uses': 3060, 'header_uses': header, 'data_uses': 3060 - header, 'total_energy': 6120,
                            'decoder_interface': interface, 'decoded_feature_kind': kind, 'raw_bits': 8192, 'coded_bits': '',
                            'online_TX_seconds': .03 if name in definitions else '', 'receiver_seconds': .05 if name in definitions else '',
                            'psnr_db': 20 + value, 'lpips': .01 * value, 'dino': .9 - .01 * value,
                            'ssim': .7 - .01 * value, 'LPIPS_excess_from_native': .01 * value - .1,
                            'severe_distortion': int(.01 * value - .1 >= .15)})

    def test_complete_16_method_grid_and_image_level_pairing(self):
        summary, paired = statistics(self.rows, self.study, self.base)
        self.assertEqual(len(self.rows), 33600)
        self.assertEqual(len(comparisons(self.study)), 63)
        self.assertEqual(len(summary), 128)
        self.assertEqual(len(paired), 3024)
        row = next(row for row in paired if row['method'] == 'hard_bounded__single_pass' and
                   row['control'] == 'hard_identity__single_pass' and row['metric'] == 'lpips' and row['snrs_db'] == '1.0+4.0+7.0')
        differences = .01 * (1 + np.arange(100) / 100)
        sampled = np.random.default_rng(self.base['evaluation']['bootstrap_seed']).integers(100, size=(32, 100))
        interval = np.percentile(differences[sampled].mean(1), [2.5, 97.5])
        self.assertAlmostEqual(row['delta'], differences.mean())
        self.assertAlmostEqual(row['ci_low'], interval[0])
        self.assertAlmostEqual(row['ci_high'], interval[1])

    def test_missing_duplicate_nonfinite_and_resource_drift_are_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(RuntimeError):
                validate_grid(rows, self.study, self.base)
        for key, value in (('total_complex_uses', 6120), ('total_energy', 6121), ('header_uses', 68), ('lpips', float('nan')),
                           ('noise_sha256', 'unpaired'), ('image_id', 'wrong_source'), ('dino', 1.1), ('severe_distortion', 7)):
            rows = [dict(self.rows[0], **{key: value}), *self.rows[1:]]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_grid(rows, self.study, self.base)

    def test_continuous_must_not_be_silently_projected_to_native(self):
        logits = torch.zeros(1, 32, 16, 16)
        truth = torch.ones_like(logits)
        result = {'logits': logits, 'native_fq': hard_native_st(logits),
                  'receiver_features': receiver_features(logits, 'continuous_mean')}
        diagnostic = feature_diagnostics(result, truth, 'continuous_mean')
        self.assertEqual(diagnostic['decoded_feature_kind'], 'continuous_bounded_features')
        self.assertEqual(diagnostic['feature_abs_mean'], 0)
        result['receiver_features'] = result['native_fq']
        with self.assertRaises(RuntimeError):
            feature_diagnostics(result, truth, 'continuous_mean')
        result['receiver_features'] = torch.ones_like(logits)
        with self.assertRaises(RuntimeError):
            feature_diagnostics(result, truth, 'hard_identity')
        with self.assertRaises(ValueError):
            feature_diagnostics(result, truth, 'unknown')
        wrong = copy.copy(self.rows)
        index = next(index for index, row in enumerate(wrong) if row['decoder_interface'] == 'continuous_mean')
        wrong[index] = dict(wrong[index], decoded_feature_kind='native_signs')
        with self.assertRaises(RuntimeError):
            validate_grid(wrong, self.study, self.base)

    def test_quality_plot_writer_with_synthetic_statistics(self):
        sys.path.insert(0, str(EXPERIMENT / 'scripts'))
        from analyze_interfaces import plot_quality
        from wetok_comm.deep_support import SUPPORT_NAME

        summary, unused = statistics(self.rows, self.study, self.base)
        support = [{**row, 'arm': SUPPORT_NAME} for row in summary if row['arm'] == 'perceptual_deepjscc' and row['snrs_db'] in ('5.0', '6.0')]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            plot_quality(output, summary, self.study, support)
            for metric in ('psnr_db', 'lpips', 'dino'):
                for suffix in ('png', 'pdf'):
                    self.assertGreater((output / f'quality_{metric}.{suffix}').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
