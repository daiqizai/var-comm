import copy
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src')]

from innovation_comm import evaluation
from innovation_comm.common import new_system
from wetok_comm.common import sha256
from wetok_comm.deep_support import SUPPORT_NAME
from wetok_comm.geometry_candidate import GeometryJSCC
from wetok_comm.training import module_sha256


def fixture_grid():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    base = yaml.safe_load((REFERENCE / 'configs/study.yaml').read_text())
    base['evaluation']['bootstrap_resamples'] = 200
    methods = [evaluation.receiver_name(variant) for variant in config['variants']]
    names = methods + evaluation.reference_names()
    rows, support = [], []
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    for index in range(100):
        for snr in base['evaluation']['snrs_db']:
            for seed in base['evaluation']['noise_seeds']:
                for method_index, name in enumerate(names):
                    quality = .2 + index / 10000 + snr / 1000 + method_index * (1 + index / 100) / 1000
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    row = {'image_index': index, 'image_id': f'synthetic_{index}', 'snr_db': snr, 'seed': seed, 'arm': name,
                        'noise_sha256': digest(f'n{index}/{seed}'), 'total_complex_uses': 3060, 'total_energy': 6120.,
                        'header_uses': header, 'data_uses': 3060 - header, 'psnr_db': 23 - quality, 'ssim': .9 - quality,
                        'lpips': quality, 'dino': 1 - quality, 'LPIPS_excess_from_native': quality - .08,
                        'severe_distortion': int(quality - .08 >= .15), 'online_TX_seconds': .01 if name in methods else '',
                        'receiver_seconds': .04 if name in methods else '', 'encoder_frozen': name in methods,
                        'decoder_interface': 'continuous_mean' if name in methods else 'frozen_reference',
                        'shared_transmitted_sha256': digest(f's{index}/{snr}'), 'shared_received_sha256': digest(f'y{index}/{snr}/{seed}'),
                        'shared_encoder_sha256': 'e' * 64, 'available_receiver_updates': 1000, 'selected_receiver_step': 1000,
                        'selected_global_data_step': 8000, 'selected_checkpoint_sha256': 'f' * 64}
                    rows.append(row)
                if snr in (5., 6.):
                    support.append({**row, 'arm': SUPPORT_NAME, 'condition_snr_db': {5.: 4., 6.: 7.}[snr],
                                    'header_uses': 0, 'data_uses': 3060})
    return config, base, rows, support


class PairedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.base, cls.rows, cls.support = fixture_grid()

    def test_full_grid_and_source_image_bootstrap(self):
        summary, paired = evaluation.statistics(self.rows, self.config, self.base)
        self.assertEqual(len(summary), 104)
        self.assertEqual(len(paired), 1968)
        self.assertEqual(len(evaluation.comparison_pairs(self.config)), 41)
        target = next(row for row in paired if row['method'] == 'receiver__multiscale_innovation' and
                      row['control'] == 'receiver__single_pass' and row['metric'] == 'lpips' and row['snrs_db'] == '1.0+4.0+7.0')
        differences = np.array([4 * (1 + index / 100) / 1000 for index in range(100)])
        draws = np.random.default_rng(self.base['evaluation']['bootstrap_seed']).integers(100, size=(200, 100))
        low, high = np.percentile(differences[draws].mean(1), [2.5, 97.5])
        self.assertAlmostEqual(target['delta'], float(differences.mean()), places=12)
        self.assertAlmostEqual(target['ci_low'], float(low), places=12)
        self.assertAlmostEqual(target['ci_high'], float(high), places=12)
        summaries, comparisons = evaluation.supplement_statistics(self.rows, self.support, self.config, self.base)
        self.assertEqual(len(summaries), 21)
        self.assertEqual(len(comparisons), 108)

    def test_missing_duplicate_nonfinite_budget_observation_and_update_errors(self):
        cases = [self.rows[:-1], self.rows + [self.rows[0]]]
        for key, value in [('total_energy', 6200), ('data_uses', 2992), ('lpips', float('nan')),
                           ('shared_received_sha256', '0' * 64), ('shared_encoder_sha256', '1' * 64),
                           ('available_receiver_updates', 2500), ('selected_receiver_step', 500),
                           ('decoder_interface', 'hard_native'), ('encoder_frozen', False), ('severe_distortion', 1)]:
            cases.append([{**self.rows[0], key: value}, *self.rows[1:]])
        for rows in cases:
            with self.subTest(first=rows[0]):
                with self.assertRaises(RuntimeError):
                    evaluation.validate_rows(rows, self.config, self.base)

    def test_repeated_source_and_receiver_dependent_transmitter_rejected(self):
        rows = [{**row, 'image_id': 'synthetic_0'} if row['image_index'] == 1 else row for row in self.rows]
        with self.assertRaisesRegex(RuntimeError, 'distinct-source'):
            evaluation.validate_rows(rows, self.config, self.base)
        rows = [{**row, 'shared_transmitted_sha256': '0' * 64} if row['image_index'] == 0 and
                row['seed'] == self.base['evaluation']['noise_seeds'][0] and row['arm'].startswith('receiver__') else row for row in self.rows]
        with self.assertRaisesRegex(RuntimeError, 'receiver noise seed'):
            evaluation.validate_rows(rows, self.config, self.base)

    def test_actual_selected_checkpoint_not_just_matching_grid(self):
        milestone = {'selected': {variant: {'step': 1000, 'checkpoint_sha256': 'f' * 64} for variant in self.config['variants']},
                     'receiver_updates_per_arm': 1000, 'frozen': {'encoder': 'e' * 64}}
        evaluation.validate_selection(self.rows, milestone)
        changed = [{**self.rows[0], 'selected_checkpoint_sha256': '0' * 64}, *self.rows[1:]]
        with self.assertRaisesRegex(RuntimeError, 'different selection'):
            evaluation.validate_selection(changed, milestone)

    def test_native_and_noiseless_diagnostics_are_complete_and_separate(self):
        clean = [{'image_index': index, 'image_id': f'synthetic_{index}', 'psnr_db': 25., 'ssim': .9, 'lpips': .08, 'dino': .95}
                 for index in range(100)]
        noiseless = [{**row, 'channel': 'noiseless_nominal19_not_wireless_ranking', 'complex_uses': 3060}
                     for row in self.rows if row['arm'].startswith('receiver__') and row['snr_db'] == 19. and
                     row['seed'] == self.base['evaluation']['noise_seeds'][0]]
        evaluation.validate_native_reference(self.rows + self.support, clean, noiseless, self.config, self.base)
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            evaluation.validate_native_reference(self.rows, clean, noiseless[:-1], self.config, self.base)
        changed = [{**clean[0], 'lpips': .1}, *clean[1:]]
        with self.assertRaisesRegex(RuntimeError, 'anchor'):
            evaluation.validate_native_reference(self.rows, changed, noiseless, self.config, self.base)

    def test_fixed_support_noise_and_actual_condition_are_not_free(self):
        for key, value in [('noise_sha256', '0' * 64), ('condition_snr_db', 5.), ('total_energy', 6200.)]:
            support = [{**self.support[0], key: value}, *self.support[1:]]
            with self.assertRaises(RuntimeError):
                evaluation.supplement_statistics(self.rows, support, self.config, self.base)


class SelectedRegistryTests(unittest.TestCase):
    def test_pinned_encoder_and_calibration_variant_checks(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        parent = GeometryJSCC('single_pass', dict(width=24, heads=2, encoder_layers=1, receiver_layers=1,
            channel_positions=204, channel_features=30)).eval().requires_grad_(False)
        encoder_hash = module_sha256(parent.encoder)
        with TemporaryDirectory() as temporary:
            training = Path(temporary)
            milestone = {'receiver_updates_per_arm': 1000, 'selected': {}, 'frozen': {'encoder': encoder_hash}}
            for variant in config['variants']:
                system = new_system(parent, variant, config, 'cpu')
                path = training / f'{variant}.pt'
                torch.save({'model': system.state_dict(), 'variant': variant, 'step': 0}, path)
                milestone['selected'][variant] = {'step': 0, 'scope': 'full', 'source_images': 1000,
                    'checkpoint': path.name, 'checkpoint_sha256': sha256(path)}
            with mock.patch.object(evaluation, 'load_parent', return_value=parent), mock.patch.object(evaluation, 'output_path', return_value=training):
                common, networks, choices = evaluation.model_registry(config, {}, milestone, 'cpu')
                self.assertIs(common, parent)
                self.assertEqual(len(networks), 6)
                self.assertTrue(all(not model.training and all(not value.requires_grad for value in model.parameters()) for model in networks.values()))
                self.assertEqual({module_sha256(model.encoder) for model in networks.values()}, {encoder_hash})
                self.assertTrue(all(choice['receiver_trainable_parameters'] > 0 for choice in choices.values()))
                bad_scope = copy.deepcopy(milestone)
                bad_scope['selected']['single_pass']['scope'] = 'monitor'
                with self.assertRaisesRegex(RuntimeError, 'eligible'):
                    evaluation.model_registry(config, {}, bad_scope, 'cpu')
                path = training / milestone['selected']['single_pass']['checkpoint']
                changed = torch.load(path, weights_only=True)
                key = next(key for key in changed['model'] if key.startswith('encoder.') and changed['model'][key].is_floating_point())
                changed['model'][key] = changed['model'][key] + .01
                torch.save(changed, path)
                milestone['selected']['single_pass']['checkpoint_sha256'] = sha256(path)
                with self.assertRaisesRegex(RuntimeError, 'different transmitter'):
                    evaluation.model_registry(config, {}, milestone, 'cpu')


if __name__ == '__main__':
    unittest.main()
