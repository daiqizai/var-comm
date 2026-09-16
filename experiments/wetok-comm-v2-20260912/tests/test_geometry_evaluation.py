from pathlib import Path
import sys
import unittest

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

from wetok_comm.geometry_evaluation import comparison_pairs, geometry_statistics, method_definitions, validate_geometry_rows
import test_interface_evaluation as interface_fixtures


class GeometryEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        interface_fixtures.InterfaceEvaluationTests.setUpClass()
        fixtures = interface_fixtures.InterfaceEvaluationTests
        cls.config = yaml.safe_load((EXPERIMENT / 'configs/geometry_study.yaml').read_text())
        cls.base = fixtures.base
        cls.rows = []
        mapping = {}
        for variant in cls.config['variants']:
            mapping['hard_identity__' + variant] = ('geometry153x40__' + variant, '153x40', variant)
            mapping['continuous_mean__' + variant] = ('geometry204x30__' + variant, '204x30', variant)
        for source in fixtures.rows:
            if source['arm'] in mapping:
                name, geometry, variant = mapping[source['arm']]
                cls.rows.append({**source, 'arm': name, 'geometry': geometry, 'variant': variant,
                    'decoder_interface': 'continuous_mean', 'decoded_feature_kind': 'continuous_bounded_features',
                    'available_image_updates': 5000, 'representation_updates': 2000})
            elif source['arm'] in ('wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc'):
                cls.rows.append(source.copy())

    def test_complete_geometry_grid_and_factorial_contrast(self):
        self.assertEqual(len(self.rows), 21000)
        self.assertEqual(len(method_definitions(self.config)), 6)
        self.assertEqual(len(comparison_pairs(self.config)), 21)
        summary, paired, interactions = geometry_statistics(self.rows, self.config, self.base)
        self.assertEqual(len(summary), 80)
        self.assertEqual(len(paired), 1008)
        self.assertEqual(len(interactions), 48)
        self.assertTrue(all(abs(row['delta']) < 1e-12 for row in interactions if row['metric'] != 'severe_distortion'))

    def test_unmatched_training_budget_and_source_noise_are_rejected(self):
        for key, value in (('available_image_updates', 1000), ('geometry', 'wrong'), ('noise_sha256', 'wrong'),
                           ('total_energy', 6121), ('decoder_interface', 'hard_identity')):
            rows = [dict(self.rows[0], **{key: value}), *self.rows[1:]]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_geometry_rows(rows, self.config, self.base)
        with self.assertRaises(RuntimeError):
            validate_geometry_rows(self.rows[:-1], self.config, self.base)


if __name__ == '__main__':
    unittest.main()
