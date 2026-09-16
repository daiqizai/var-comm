import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]
from analyze_milestone import statistics
from train_milestone import choose
from wetok_comm.model import NativeWeTokJSCC
from wetok_comm.training import paired_batches
from wetok_comm.evaluation import raw_noise


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())

    def test_paired_sampler_resumes_across_epochs(self):
        batches = list(paired_batches(self.config, 16, 0, 12))
        resumed = list(paired_batches(self.config, 16, 5, 12))
        self.assertEqual([row['fingerprint'] for row in batches[5:]], [row['fingerprint'] for row in resumed])
        self.assertEqual([row['step'] for row in batches], list(range(12)))

    def test_development_noise_matches_frozen_reference_exactly(self):
        sys.path.insert(0, str(EXPERIMENT.parents[2] / 'var-next-scale-comm/src'))
        from var_comm.study import seeded_noise
        for identifier in ('n01440764/ILSVRC2012_val_00003014_n01440764', 'test_source'):
            for seed in (2001, 2002, 2003):
                np.testing.assert_array_equal(raw_noise(identifier, seed), seeded_noise(identifier, seed, (3060, 2)))

    def test_monitor_cannot_choose_checkpoint(self):
        candidate = {'scope': 'monitor', 'source_images': 100, 'lpips': .1, 'step': 1000}
        with self.assertRaises(ValueError):
            choose(None, candidate)
        full = {**candidate, 'scope': 'full', 'source_images': 1000, 'step': 2000}
        self.assertEqual(choose(None, full), full)
        self.assertEqual(choose(full, {**full, 'step': 5000}), full)

    def test_history_condition_is_not_a_name_only(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        received, snrs = torch.randn(1, 3060, 2), torch.tensor([7.])
        for variant in ('multiscale_no_history', 'multiscale_conditioned'):
            torch.manual_seed(94)
            model = NativeWeTokJSCC(variant, self.config['model']).eval()
            with torch.no_grad():
                model.receiver.output.weight.normal_(std=.05)
                original = model.receive(received, snrs)['logits']
                original_read = model.receiver.read
                def perturbed_read(guessed, memory, conditions, size, previous=None):
                    result = original_read(guessed, memory, conditions, size, previous)
                    return result + 2 if size < 16 else result
                model.receiver.read = perturbed_read
                perturbed = model.receive(received, snrs)['logits']
            if variant == 'multiscale_no_history':
                self.assertTrue(torch.equal(original, perturbed))
            else:
                self.assertGreater(float((original - perturbed).abs().max()), 1e-5)

    def test_full_wireless_grid_and_source_pairing(self):
        config = copy.deepcopy(self.config)
        config['evaluation']['bootstrap_resamples'] = 100
        arms = config['arms'] + ['wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc']
        rows = []
        for index in range(100):
            for snr in config['evaluation']['snrs_db']:
                for seed in config['evaluation']['noise_seeds']:
                    for ordinal, arm in enumerate(arms):
                        rows.append({'image_index': index, 'image_id': f'synthetic_{index}', 'snr_db': snr, 'seed': seed, 'arm': arm,
                                     'noise_sha256': f'{index}_{seed}', 'total_complex_uses': 3060, 'total_energy': 6120.,
                                     'psnr_db': 20. + ordinal + index * .01, 'ssim': .5 + ordinal * .01,
                                     'lpips': .2 + ordinal * .01, 'dino': .8 + ordinal * .01,
                                     'LPIPS_excess_from_native': .1 + ordinal * .01, 'severe_distortion': float(index % 2),
                                     'online_TX_seconds': '', 'receiver_seconds': ''})
        summaries, pairs = statistics(rows, config)
        self.assertEqual(len(rows), 14700)
        self.assertEqual(len(summaries), 56)
        selected = [row for row in pairs if row['method'] == 'multiscale_conditioned' and row['control'] == 'multiscale_no_history' and row['metric'] == 'lpips']
        self.assertTrue(all(abs(row['delta'] - .01) < 1e-12 for row in selected))
        with self.assertRaises(RuntimeError):
            statistics(rows[:-1], config)


if __name__ == '__main__':
    unittest.main()
