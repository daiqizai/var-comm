import copy
from pathlib import Path
import sys
import unittest

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from joint_sender.evaluation_io import digest
from sufficiency.evaluation import all_names, comparison_pairs, previous_name, statistics, validate_rows, validate_selection
from sufficiency.references import project_reference
from wetok_comm.evaluation import raw_noise


def fixture():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    original = yaml.safe_load((JOINT / 'configs/study.yaml').read_text())
    grid = yaml.safe_load((GRID / 'configs/study.yaml').read_text())
    reference = yaml.safe_load((INNOVATION / 'configs/study.yaml').read_text())
    base = yaml.safe_load((BASE / 'configs/study.yaml').read_text())
    base['evaluation']['bootstrap_resamples'] = 20
    rows = []
    for index in range(2):
        identifier = f'synthetic_source_{index}'
        for snr in base['evaluation']['snrs_db']:
            for seed in base['evaluation']['noise_seeds']:
                for name in all_names(config, original, grid, reference):
                    fresh = name.startswith('r2__')
                    score = .2 + .01 * index - (.005 if fresh else 0.)
                    rows.append({'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed, 'arm': name,
                        'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': digest(raw_noise(identifier, seed)),
                        'psnr_db': 20., 'ssim': .8, 'lpips': score, 'dino': .7, 'LPIPS_excess_from_native': score - .1,
                        'severe_distortion': 0, 'online_TX_seconds': '', 'receiver_seconds': '',
                        'r2_quality_origin': 'new_r2_measurement' if fresh else 'sealed_5000_reference',
                        'available_updates': 10000 if fresh else 5000, 'new_update_opportunity': 5000, 'parent_updates': 7000,
                        'header_uses': 0, 'data_uses': 3060, 'raw_bits': 8192, 'decoder_interface': 'continuous_mean',
                        'transmitted_sha256': f'{name}_{index}_{snr}', 'selected_step': 5000,
                        'selected_global_data_step': 12000, 'checkpoint_sha256': 'old_best_checkpoint'})
    return config, original, grid, reference, base, rows


class QualityContractTests(unittest.TestCase):
    def test_all22_methods_and_correctly_labeled_training_opportunities(self):
        config, original, grid, reference, base, rows = fixture()
        lookup, names = validate_rows(rows, config, original, grid, reference, base, 2)
        self.assertEqual((len(names), len(rows)), (22, 924))
        self.assertEqual(len(comparison_pairs(config, original, grid, reference)), 78)
        summaries, paired = statistics(rows, config, original, grid, reference, base, 2)
        self.assertEqual(len(paired), 3744)
        training = [row for row in paired if row['comparison_scope'] == 'extra_training_effect_not_new_mechanism']
        structure = [row for row in paired if row['comparison_scope'] == 'matched10000_structure_comparison']
        self.assertEqual((len(training), len(structure)), (192, 288))
        self.assertTrue(all(row['control'] == previous_name(row['method'].split('__', 1)[1]) for row in training))
        self.assertTrue(all(row['control'].startswith('r2__') for row in structure))
        difference = next(row for row in training if row['metric'] == 'lpips')
        self.assertAlmostEqual(difference['delta'], -.005)
        self.assertIn('frozen__full_grid_innovation', names)
        self.assertIn('digital_adaptive', names)

    def test_old_checkpoint_may_be_selected_but_new_opportunity_cannot_be_faked(self):
        config, original, grid, reference, base, rows = fixture()
        milestone = {'selected': {name: {'step': 5000, 'checkpoint_sha256': 'old_best_checkpoint'} for name in config['variants']}}
        validate_selection(rows, milestone)
        fresh = next(row for row in rows if row['arm'].startswith('r2__'))
        fresh['selected_step'] = 10000
        with self.assertRaises(RuntimeError):
            validate_selection(rows, milestone)
        fresh['selected_step'] = 5000
        fresh['available_updates'] = 5000
        with self.assertRaisesRegex(RuntimeError, 'update, input'):
            validate_rows(rows, config, original, grid, reference, base, 2)

    def test_missing_rows_wrong_resources_noise_or_current_latency_are_rejected(self):
        config, original, grid, reference, base, rows = fixture()
        with self.assertRaises(RuntimeError):
            validate_rows(rows[:-1], config, original, grid, reference, base, 2)
        for key, value in [('total_complex_uses', 3061), ('total_energy', 6121), ('noise_sha256', 'wrong'), ('receiver_seconds', .04)]:
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.assertRaises(RuntimeError):
                validate_rows(changed, config, original, grid, reference, base, 2)

    def test_reference_projection_preserves_old_quality_and_archived_provenance(self):
        row = {'quality_origin': 'new_grid_measurement', 'lpips': '.2', 'available_updates': '5000',
               'image_archive': '/old/image.npz', 'receiver_seconds': '', 'online_TX_seconds': '',
               'archived_receiver_seconds': '.04', 'checkpoint_sha256': 'old_checkpoint'}
        before = copy.deepcopy(row)
        result = project_reference(row, 'sealed_grid_receipt')
        self.assertEqual(row, before)
        self.assertTrue(all(result[key] == value for key, value in before.items()))
        self.assertEqual(result['r2_quality_origin'], 'sealed_5000_reference')
        row['receiver_seconds'] = '.04'
        with self.assertRaises(RuntimeError):
            project_reference(row, 'sealed_grid_receipt')


if __name__ == '__main__':
    unittest.main()
