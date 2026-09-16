import copy
from pathlib import Path
import sys
import unittest

import torch
import numpy as np
from torch import nn
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]
from wetok_comm.bit_support import arm_definitions, clone_optimizer, optimizer_digest
from analyze_bit_support import statistics


class BitSupportTests(unittest.TestCase):
    def test_only_one_weight_changes_and_six_arms(self):
        repair = yaml.safe_load((EXPERIMENT / 'configs/bit_support.yaml').read_text())
        self.assertEqual(len(arm_definitions(repair)), 6)
        first, second = repair['recipes']['joint_original'], repair['recipes']['bit_support']
        self.assertEqual([key for key in first if first[key] != second[key]], ['bits'])
        self.assertEqual(second['bits'], 1.)

    def test_model_and_adam_state_are_equal_but_not_shared(self):
        torch.manual_seed(76)
        parent = nn.Linear(3, 2)
        original = torch.optim.AdamW(parent.parameters(), lr=.0001, betas=(.9, .99), weight_decay=.0001)
        parent(torch.randn(4, 3)).square().mean().backward()
        original.step()
        state = original.state_dict()
        first, second = copy.deepcopy(parent), copy.deepcopy(parent)
        first_optimizer = clone_optimizer(first, state, .0001)
        second_optimizer = clone_optimizer(second, state, .0001)
        self.assertEqual(optimizer_digest(first_optimizer.state_dict()), optimizer_digest(second_optimizer.state_dict()))
        before = optimizer_digest(second_optimizer.state_dict())
        first_optimizer.zero_grad(set_to_none=True)
        first(torch.randn(4, 3)).sum().backward()
        first_optimizer.step()
        self.assertEqual(optimizer_digest(second_optimizer.state_dict()), before)
        self.assertTrue(all(int(value['step']) == 1 for value in second_optimizer.state.values()))
        self.assertTrue(all(int(value['step']) == 2 for value in first_optimizer.state.values()))

    def test_recipe_statistics_include_prior_best_and_matched_controls(self):
        repair = yaml.safe_load((EXPERIMENT / 'configs/bit_support.yaml').read_text())
        base = yaml.safe_load((EXPERIMENT / repair['base_config']).read_text())
        base['evaluation']['bootstrap_resamples'] = 100
        names = list(arm_definitions(repair)) + ['wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc']
        names += ['old_selected__' + variant for variant in repair['variants']]
        rows = []
        for index in range(100):
            for snr in base['evaluation']['snrs_db']:
                for seed in base['evaluation']['noise_seeds']:
                    for ordinal, name in enumerate(names):
                        header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                        rows.append({'image_index': index, 'image_id': f'synthetic_{index}', 'snr_db': snr, 'seed': seed,
                            'arm': name, 'noise_sha256': f'{index}_{seed}', 'total_complex_uses': 3060, 'total_energy': 6120,
                            'header_uses': header, 'data_uses': 3060 - header, 'psnr_db': 20 + ordinal + index * .01,
                            'ssim': .5 + ordinal * .01, 'lpips': .2 + ordinal * .01, 'dino': .6 + ordinal * .01,
                            'LPIPS_excess_from_native': .1 + ordinal * .01, 'severe_distortion': float(index % 2),
                            'receiver_seconds': '', 'online_TX_seconds': ''})
        summary, paired = statistics(rows, repair, base)
        self.assertEqual(len(rows), 27300)
        self.assertEqual(len(summary), 104)
        self.assertEqual(len(paired), 1104)
        selected = [row for row in paired if row['method'] == 'bit_support__single_pass' and row['control'] == 'joint_original__single_pass' and row['metric'] == 'lpips']
        self.assertTrue(all(abs(row['delta'] - .01) < 1e-12 for row in selected))
        self.assertTrue(any(row['control'] == 'old_selected__single_pass' for row in paired))


if __name__ == '__main__':
    unittest.main()
