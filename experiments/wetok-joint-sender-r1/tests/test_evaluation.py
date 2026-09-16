import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from joint_sender import evaluation
from joint_sender.evaluation_io import SourceImages, digest
from wetok_comm.common import sha256
from wetok_comm.deep_support import SUPPORT_NAME


def fixture_grid():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    reference = yaml.safe_load((INNOVATION / 'configs/study.yaml').read_text())
    base = yaml.safe_load((BASE / 'configs/study.yaml').read_text())
    base['evaluation']['bootstrap_resamples'] = 200
    measured = evaluation.fresh_names(config, reference)
    names = evaluation.all_names(config, reference)
    shifts = {'joint__single_pass': -.01, 'joint__multiscale_no_history': -.008, 'joint__multiscale_state_history': -.015,
              'frozen__single_pass': 0., 'frozen__multiscale_no_history': .003, 'frozen__multiscale_state_history': .001}
    hash_text = lambda value: hashlib.sha256(value.encode()).hexdigest()
    rows, support = [], []
    for index in range(100):
        for snr in base['evaluation']['snrs_db']:
            for seed in base['evaluation']['noise_seeds']:
                for name in names:
                    quality = .2 + index / 10000 + snr / 1000 + shifts.get(name, .005) * (1 + index / 100)
                    policy = 'joint_E_R' if name.startswith('joint__') else 'receiver_only'
                    sender = name if name.startswith('joint__') else 'common_frozen'
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    row = {'image_index': index, 'image_id': f'synthetic_{index}', 'snr_db': snr, 'seed': seed, 'arm': name,
                        'total_complex_uses': 3060, 'header_uses': header, 'data_uses': 3060 - header, 'total_energy': 6120.,
                        'noise_sha256': hash_text(f'noise{index}/{seed}'), 'transmitted_sha256': hash_text(f's{index}/{snr}/{sender}'),
                        'received_sha256': hash_text(f'y{index}/{snr}/{seed}/{sender}'), 'encoder_sha256': hash_text(sender),
                        'checkpoint_sha256': hash_text(name), 'selected_step': 1000, 'selected_global_data_step': 8000,
                        'available_updates': 5000, 'decoder_interface': 'continuous_mean', 'training_policy': policy,
                        'receiver_forward_scope': 'fine_only_no_history_auxiliary_reads_pruned' if 'no_history' in name else 'full_declared_receiver',
                        'prune_feature_max_error': 0., 'online_TX_seconds': .01 if name in measured else '',
                        'receiver_seconds': .04 if name in measured else '', 'psnr_db': 25 - quality, 'ssim': .9 - quality,
                        'lpips': quality, 'dino': 1 - quality, 'LPIPS_excess_from_native': quality - .08,
                        'severe_distortion': int(quality - .08 >= .15)}
                    rows.append(row)
                if snr in (5., 6.):
                    support.append({**row, 'arm': SUPPORT_NAME, 'header_uses': 0, 'data_uses': 3060,
                                    'condition_snr_db': {5.: 4., 6.: 7.}[snr]})
    return config, reference, base, rows, support


class JointEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.reference, cls.base, cls.rows, cls.support = fixture_grid()

    def test_full_grid_accepts_distinct_joint_waveforms_and_retains_all_strong_controls(self):
        lookup, names = evaluation.validate_rows(self.rows, self.config, self.reference, self.base)
        self.assertEqual(len(lookup), 33600)
        self.assertEqual(len(names), 16)
        self.assertIn('frozen__full_grid_innovation', names)
        self.assertEqual(len(evaluation.comparison_pairs(self.config, self.reference)), 42)
        self.assertNotEqual(lookup[0, 1., 2001, 'joint__single_pass']['received_sha256'], lookup[0, 1., 2001, 'frozen__single_pass']['received_sha256'])

    def test_source_image_bootstrap_and_joint_history_interaction(self):
        summaries, paired, interactions = evaluation.statistics(self.rows, self.config, self.reference, self.base)
        self.assertEqual((len(summaries), len(paired), len(interactions)), (128, 2016, 144))
        target = next(row for row in interactions if row['snrs_db'] == '1.0+4.0+7.0' and row['metric'] == 'lpips' and
                      row['method_structure'] == 'multiscale_state_history' and row['control_structure'] == 'multiscale_no_history')
        differences = -.005 * (1 + np.arange(100) / 100)
        sampled = np.random.default_rng(self.base['evaluation']['bootstrap_seed']).integers(100, size=(200, 100))
        low, high = np.percentile(differences[sampled].mean(1), [2.5, 97.5])
        self.assertAlmostEqual(target['joint_minus_frozen_structure_contrast'], float(differences.mean()), places=12)
        self.assertAlmostEqual(target['ci_low'], float(low), places=12)
        self.assertAlmostEqual(target['ci_high'], float(high), places=12)
        support_summary, support_paired = evaluation.support_statistics(self.rows, self.support, self.config, self.reference, self.base)
        self.assertEqual((len(support_summary), len(support_paired)), (30, 162))

    def test_grid_budget_noise_policy_training_opportunity_and_failed_outcomes(self):
        candidates = [self.rows[:-1], self.rows + [self.rows[0]]]
        for field, value in [('total_energy', 6200), ('header_uses', 68), ('available_updates', 1000), ('selected_step', 3000),
                             ('lpips', float('nan')), ('receiver_seconds', -1), ('decoder_interface', 'hard_identity'),
                             ('training_policy', 'receiver_only'), ('noise_sha256', '0' * 64)]:
            candidates.append([{**self.rows[0], field: value}, *self.rows[1:]])
        for rows in candidates:
            with self.assertRaises((ValueError, RuntimeError)):
                evaluation.validate_rows(rows, self.config, self.reference, self.base)
        corrupted = [{**row, 'received_sha256': '0' * 64} if row['image_index'] == 0 and row['snr_db'] == 1. and
                     row['seed'] == 2001 and row['arm'] == 'frozen__single_pass' else row for row in self.rows]
        with self.assertRaisesRegex(RuntimeError, 'same actual y'):
            evaluation.validate_rows(corrupted, self.config, self.reference, self.base)

    def test_transmitter_cannot_depend_on_noise_seed_and_source_ids_cannot_repeat(self):
        altered = [{**row, 'transmitted_sha256': '0' * 64} if row['image_index'] == 0 and row['snr_db'] == 1. and
                   row['seed'] == 2001 and row['arm'] == 'joint__single_pass' else row for row in self.rows]
        with self.assertRaisesRegex(RuntimeError, 'transmitter reads'):
            evaluation.validate_rows(altered, self.config, self.reference, self.base)
        repeated = [{**row, 'image_id': 'synthetic_0'} if row['image_index'] == 1 else row for row in self.rows]
        with self.assertRaisesRegex(RuntimeError, '100 distinct'):
            evaluation.validate_rows(repeated, self.config, self.reference, self.base)

    def test_nine_model_timing_order_balanced_within_one(self):
        names = evaluation.fresh_names(self.config, self.reference)
        counts = {(name, position, snr): 0 for name in names for position in range(9) for snr in range(7)}
        for index in range(100):
            for snr in range(7):
                for seed in range(3):
                    frame = (index * 7 + snr) * 3 + seed
                    for position, name in enumerate(evaluation.measurement_order(names, frame)):
                        counts[name, position, snr] += 1
        self.assertEqual(set(counts.values()), {33, 34})

    def test_identical_reference_pixels_are_not_copied_into_new_archive(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / 'old.npz'
            frozen = torch.full((3, 256, 256), .2)
            np.savez_compressed(old, images=frozen[None].numpy())
            old_hash = sha256(old)
            previous = {'image_archive': str(old), 'image_ref': 0, 'image_sha256': digest(frozen)}
            output = root / 'new'
            output.mkdir()
            images = SourceImages(output)
            fresh = images.add(torch.full_like(frozen, .4))
            reused = images.add(frozen, previous)
            self.assertEqual(reused['image_store'], 'verified_existing_reference')
            self.assertEqual(reused['image_archive'], str(old))
            self.assertEqual(fresh['image_ref'], 0)
            images.save()
            with np.load(output / 'reconstructions.npz', allow_pickle=False) as archive:
                self.assertEqual(len(archive['images']), 1)
                self.assertTrue(np.all(archive['images'][0] == np.float32(.4)))
            self.assertEqual(sha256(old), old_hash)


if __name__ == '__main__':
    unittest.main()
