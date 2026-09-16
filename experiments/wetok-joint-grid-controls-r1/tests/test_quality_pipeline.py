"""The real quality driver resumes committed sources and keeps timings out of quality records."""

from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from grid_controls.evaluation import all_names, learned_names, new_names, validate_diagnostics, validate_rows
from grid_controls.references import project_reference, read_rows
from joint_sender.evaluation_io import digest
from wetok_comm.common import sha256
from wetok_comm.evaluation import raw_noise
from wetok_comm.native import hard_native_st
from wetok_comm.training import module_sha256
from test_quality_evaluation import fixture

SPEC = importlib.util.spec_from_file_location('grid_quality_driver_fixture', EXPERIMENT / 'scripts/evaluate_quality.py')
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class CPUTorch:
    @staticmethod
    def device(unused):
        return torch.device('cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


class NativeFixture:
    def __init__(self, fail_on_second=False):
        self.codec = nn.Identity()
        self.calls = 0
        self.fail_on_second = fail_on_second

    def encode(self, source):
        self.calls += 1
        if self.calls == 2 and self.fail_on_second:
            raise RuntimeError('synthetic interruption')
        return torch.full((len(source), 16, 16, 4), 255, dtype=torch.uint8)

    def decode(self, features):
        return functional.interpolate((features[:, :3] + 1) / 4 + .2, (256, 256))


class ModelFixture(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.weight = nn.Parameter(torch.tensor(.5))

    def transmit(self, source, snrs):
        return torch.ones(len(source), 3060, 2)

    def receive(self, received, snrs):
        logits = (received[:, :1, :1] * self.weight).reshape(-1, 1, 1, 1).expand(-1, 32, 16, 16)
        mean = torch.tanh(logits / 2)
        return {'logits': logits, 'native_fq': hard_native_st(logits), 'receiver_features': mean,
            'states': [functional.adaptive_avg_pool2d(mean, size) for size in (4, 8)],
            'feature_gates': [], 'residual_noise_ratios': []}


def fixture_metrics(source, reconstructions, unused_perceptual=None, unused_dino=None):
    rows = []
    for image in reconstructions:
        mse = float((image[None] - source).square().mean())
        rows.append({'psnr_db': float(-10 * np.log10(max(mse, 1e-12))), 'ssim': .8, 'lpips': mse, 'dino': .7})
    return rows


class QualityPipelineTests(unittest.TestCase):
    def test_real_driver_interruption_resume_keeps_reference_rows_and_no_current_times(self):
        config, original, reference, base, unused = fixture()
        base['evaluation']['snrs_db'], base['evaluation']['primary_snrs_db'], base['evaluation']['noise_seeds'] = [1., 5., 6., 19.], [1.], [2001]
        evaluation = {'source_images': 2, 'noiseless_rows': 22, 'support_rows': 4,
                     'native_metric_tolerances': {'psnr_db': 1e-4, 'ssim': 1e-5, 'lpips': 1e-5, 'dino': 1e-5}}
        source = torch.stack([torch.full((3, 256, 256), value) for value in (.1, .2)])
        codes = np.full((2, 1, 16, 16, 4), 255, dtype=np.uint8)
        identifiers = ['synthetic_source_0', 'synthetic_source_1']
        old_names = all_names(config, original, reference)[2:]
        clean, old_rows, noiseless, support = {}, [], [], []
        for index, identifier in enumerate(identifiers):
            image = torch.full((3, 256, 256), .7)
            score = fixture_metrics(source[index:index+1], [image])[0]
            clean[index] = {'image_index': index, 'image_id': identifier, 'raw_bits': 8192, **score}
            for snr in base['evaluation']['snrs_db']:
                for name in old_names:
                    old_rows.append({'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': 2001, 'arm': name,
                        'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': digest(raw_noise(identifier, 2001)),
                        **score, 'LPIPS_excess_from_native': 0., 'severe_distortion': 0, 'online_TX_seconds': '.01',
                        'receiver_seconds': '.03', 'receiver_order_index': '0', 'latency_source': 'old_verified_time',
                        'image_archive': '/unchanged/reference.npz', 'image_ref': '0', 'image_sha256': digest(image)})
            for name in learned_names(config, original, reference)[2:]:
                noiseless.append({'image_index': index, 'image_id': identifier, 'arm': name,
                    'channel': 'noiseless_nominal19_not_wireless_ranking', **score, 'LPIPS_excess_from_native': 0., 'severe_distortion': 0})
            for snr in (5., 6.):
                support.append({'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': 2001,
                    'arm': 'perceptual_deepjscc_fixed_support', **score, 'LPIPS_excess_from_native': 0., 'severe_distortion': 0})

        class ReferencesFixture:
            receipt_sha = 'old_quality_receipt'
            source_table_sha = 'frozen_source_tensors'
            frozen_models = {key: module_sha256(nn.Identity()) for key in ('native', 'lpips', 'dino')}

            def __init__(self, *args):
                self.clean, self.noiseless, self.support = clean, noiseless, support

            def source_rows(self, index):
                return [project_reference(row, self.receipt_sha) for row in old_rows if row['image_index'] == index]

            def validate_source(self, index, identifier, image):
                if identifier != identifiers[index] or not torch.equal(image, source[index:index+1]):
                    raise RuntimeError('fixture source changed')

            def validate_reuse(self, rows):
                lookup = {(row['image_index'], row['snr_db'], row['seed'], row['arm']): row for row in old_rows}
                for row in rows:
                    if not row['arm'].startswith('grid__'):
                        expected = project_reference(lookup[int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']], self.receipt_sha)
                        if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
                            raise RuntimeError('fixture reference changed')

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / 'milestone.json', root / 'review.json']
            for path in paths:
                path.write_text('{}')
            milestone = {'selected': {name: {'step': 5000, 'checkpoint_sha256': 'checkpoint'} for name in config['variants']}}

            def registry(*args):
                models = {'grid__' + name: ModelFixture(name) for name in config['variants']}
                choices = {name: {'step': 5000, 'checkpoint_sha256': 'checkpoint', 'encoder_sha256': 'encoder',
                    'selected_encoder_differs_from_parent': True, 'communication_parameters': 1, 'optimized_parameters': 1} for name in models}
                return nn.Identity(), models, choices

            def validate(values, cfg, orig, ref, current_base):
                return validate_rows(values, cfg, orig, ref, current_base, source_count=2)

            def diagnostics(values, anchors, no_noise, supplement, cfg, orig, ref, current_base):
                return validate_diagnostics(values, anchors, no_noise, supplement, cfg, orig, ref, current_base, source_count=2)

            with ExitStack() as stack:
                patches = {'torch': CPUTorch(), 'configure_torch': lambda: None,
                    'load_evaluation': lambda: (evaluation, config, original, reference, base, {}),
                    'audited_grid': lambda *args: (milestone, paths[0], paths[1], {'hashes': {'joint_analysis': 'old_analysis'}}),
                    'References': ReferencesFixture, 'output_path': lambda *args: root / 'evaluation',
                    'admit_quality': lambda cfg, shared: {'shared': shared}, 'quality_telemetry': lambda cfg: {'local_time': 'synthetic', 'compute_pids': [123, 456]},
                    'model_registry': registry, 'metric_models': lambda device: (nn.Identity(), nn.Identity()),
                    'read_population': lambda *args: (source, codes, identifiers), 'validate_population': lambda ids: None,
                    'metrics': fixture_metrics, 'validate_rows': validate, 'validate_diagnostics': diagnostics}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(EVALUATOR, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(EVALUATOR, 'FrozenWeTok', return_value=NativeFixture(True)), \
                        mock.patch.object(sys, 'argv', ['evaluate_quality.py', '--shared-gpu', '--execute']):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        EVALUATOR.main()
                output = root / 'evaluation/quality_0005000'
                first_hash = sha256(output / 'images/000/receipt.json')
                with mock.patch.object(EVALUATOR, 'FrozenWeTok', return_value=NativeFixture()), \
                        mock.patch.object(sys, 'argv', ['evaluate_quality.py', '--shared-gpu', '--resume', '--execute']):
                    EVALUATOR.main()
            self.assertEqual(sha256(output / 'images/000/receipt.json'), first_hash)
            completion = json.loads((output / 'completion.json').read_text())
            self.assertEqual((completion['rows'], completion['new_rows'], completion['noiseless_rows']), (144, 16, 22))
            self.assertFalse(completion['time_lower_bound_due_to_unfinished_sessions'])
            rows = read_rows(output / 'per_frame.csv')
            self.assertTrue(all(row['receiver_seconds'] == '' and row['online_TX_seconds'] == '' for row in rows))
            self.assertTrue(all(row['archived_receiver_seconds'] == '.03' for row in rows if not row['arm'].startswith('grid__')))
            for relative, expected in completion['output_hashes'].items():
                self.assertEqual(sha256(output / relative), expected)


if __name__ == '__main__':
    unittest.main()
