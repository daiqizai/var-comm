"""Synthetic CPU-only wiring tests; these fixtures are not scientific results."""

from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import evaluate_interfaces as evaluator
from wetok_comm.common import sha256
from wetok_comm.deep_support import SUPPORT_NAME
from wetok_comm.interface_evaluation import reference_names
from wetok_comm.interface_study import interface_definitions, receiver_features
from wetok_comm.native import hard_native_st


class CPUTorchProxy:
    cuda = SimpleNamespace(synchronize=lambda: None)

    @staticmethod
    def device(unused):
        return torch.device('cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


class NativeFixture:
    def __init__(self, fail_on_third_encode=False):
        self.codec = nn.Identity()
        self.calls = 0
        self.fail_on_third_encode = fail_on_third_encode

    def encode(self, images):
        self.calls += 1
        if self.fail_on_third_encode and self.calls == 3:
            raise RuntimeError('synthetic interruption after first committed source')
        return torch.zeros(len(images), 16, 16, 4, dtype=torch.uint8)

    def decode(self, features):
        return functional.interpolate((features[:, :3] + 1) / 2, (256, 256), mode='nearest')


class CommunicationFixture(nn.Module):
    def __init__(self, interface):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))
        self.interface = interface
        self.transmit_calls = 0

    def transmit(self, source, snrs):
        self.transmit_calls += 1
        return torch.ones(len(source), 3060, 2)

    def receive(self, received, snrs):
        logits = self.weight.expand(len(received), 32, 16, 16)
        return {'logits': logits, 'native_fq': hard_native_st(logits),
                'receiver_features': receiver_features(logits, self.interface)}


class SupportModelFixture:
    def __init__(self, unused, device):
        self.model = nn.Identity()

    def transmit(self, images, actual_snr):
        return torch.ones(len(images), 3060, 2)

    def receive(self, received, actual_snr):
        return torch.full((len(received), 3, 256, 256), .3)


def fixture_metrics(source, images, unused_perceptual, unused_dino):
    rows = []
    for image in images:
        mse = (image - source[0]).square().mean()
        rows.append({'psnr_db': float(-10 * mse.clamp_min(1e-12).log10()), 'ssim': 1 - float(mse),
                     'lpips': float(mse), 'dino': 1 - float(mse) / 2})
    return rows


class InterfacePipelineTests(unittest.TestCase):
    def test_gpu_ownership_guard_refuses_foreign_or_unknown_processes(self):
        with mock.patch.object(evaluator.subprocess, 'check_output', return_value=f'{os.getpid()}\n'):
            evaluator.require_uncontended_gpu()
        with mock.patch.object(evaluator.subprocess, 'check_output', return_value=f'{os.getpid() + 1000000}\n'):
            with self.assertRaisesRegex(RuntimeError, 'not stopping them'):
                evaluator.require_uncontended_gpu()
        with mock.patch.object(evaluator.subprocess, 'check_output', return_value='[N/A]\n'):
            with self.assertRaisesRegex(RuntimeError, 'cannot be verified'):
                evaluator.require_uncontended_gpu()

    def test_output_interfaces_reference_ownership_and_source_atomic_resume(self):
        torch.set_num_threads(2)
        study = yaml.safe_load((EXPERIMENT / 'configs/interface_study.yaml').read_text())
        base = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        evaluation = yaml.safe_load((EXPERIMENT / 'configs/interface_evaluation.yaml').read_text())
        deep = yaml.safe_load((EXPERIMENT / 'configs/deep_support.yaml').read_text())
        deep['source_hashes'] = {}
        definitions = interface_definitions(study)
        source_images = torch.stack([torch.full((3, 256, 256), value) for value in (.25, .35)])
        source_codes = np.zeros((2, 1, 16, 16, 4), dtype=np.uint8)
        identifiers = ['synthetic_source_0', 'synthetic_source_1']
        with TemporaryDirectory() as temporary:
            project = Path(temporary)
            output_for = lambda unused, kind: project / 'outputs' / kind
            training = output_for(study, 'training')
            selected = {}
            for name, definition in definitions.items():
                path = training / name / 'selected.pt'
                path.parent.mkdir(parents=True)
                torch.save({'model': CommunicationFixture(definition['interface']).state_dict(),
                            **definition, 'additional_step': 1000}, path)
                selected[name] = {'checkpoint': str(path.relative_to(training)), 'checkpoint_sha256': sha256(path), 'step': 1000}
            milestone = training / 'milestones/additional_0001000.json'
            milestone.parent.mkdir()
            milestone.write_text(json.dumps({'status': 'INTERFACE_MILESTONE_COMPLETE', 'additional_updates_per_arm': 1000,
                                             'source_hashes': {}, 'selected': selected}))
            review = output_for(study, 'analysis') / 'calibration_0001000/completion.json'
            review.parent.mkdir(parents=True)
            review.write_text(json.dumps({'status': 'INTERFACE_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE',
                                          'source_hashes': {}, 'paired_data_Adam_energy_and_selection_audit': 'PASS',
                                          'milestone_sha256': sha256(milestone)}))
            reference_file = project / 'frozen_reference.npz'
            reference_image = torch.full((3, 256, 256), .2)
            np.savez_compressed(reference_file, images=reference_image[None].numpy())
            reference_hash = sha256(reference_file)

            class ReferenceFixture:
                def __init__(self, unused_config, unused_study):
                    pass

                def image(self, index, identifier, snr, seed, name, noise_hash):
                    if identifier != identifiers[index] or name not in reference_names(study):
                        raise ValueError('fixture reference key changed')
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    scores = fixture_metrics(source_images[index:index + 1], [reference_image], None, None)[0]
                    return reference_image.clone(), {**scores, 'header_uses': header, 'data_uses': 3060 - header,
                        'raw_bits': '', 'coded_bits': '', 'bit_error_rate': '', 'crc_accepted': '',
                        'image_ref': 0, 'online_TX_seconds': '', 'receiver_seconds': '', 'selected_step': ''}, str(reference_file)

            class SupportReferenceFixture:
                def __init__(self, unused_config, unused_base):
                    pass

                def image(self, index, identifier, snr, seed):
                    image = torch.full((3, 256, 256), .3)
                    scores = fixture_metrics(source_images[index:index + 1], [image], None, None)[0]
                    return image, {'psnr_db': scores['psnr_db'], 'lpips_alex': scores['lpips'], 'dino_cosine': scores['dino']}

            instances = {}
            def network_factory(unused_base, definition, state, device):
                network = CommunicationFixture(definition['interface'])
                network.load_state_dict(state)
                instances[definition['interface'] + '__' + definition['variant']] = network
                return network

            def validate_fixture_grid(rows, unused_study, unused_base):
                expected = {(index, snr, seed, name) for index in range(2) for snr in base['evaluation']['snrs_db']
                            for seed in base['evaluation']['noise_seeds'] for name in list(definitions) + reference_names(study)}
                actual = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']) for row in rows}
                self.assertEqual(actual, expected)
                self.assertEqual(len(rows), 672)

            def validate_fixture_support(main, support, unused_study, unused_base):
                self.assertEqual(len(support), 12)
                self.assertTrue(all(row['arm'] == SUPPORT_NAME for row in support))
                return [], []

            with ExitStack() as stack:
                patches = {'PROJECT': project, 'torch': CPUTorchProxy(), 'configure_torch': lambda: None,
                    'load_evaluation_config': lambda: (evaluation, study, base, training), 'load_deep_support': lambda: deep,
                    'interface_output': output_for, 'metric_models': lambda unused: (nn.Identity(), nn.Identity()),
                    'read_population': lambda unused, population: (source_images, source_codes, identifiers),
                    'FrozenImageReferences': ReferenceFixture, 'DeepSupportReferences': SupportReferenceFixture,
                    'FrozenDeepSupport': SupportModelFixture, 'make_interface_network': network_factory,
                    'metrics': fixture_metrics, 'validate_grid': validate_fixture_grid, 'supplement_statistics': validate_fixture_support}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(evaluator, name, value))
                stack.enter_context(mock.patch.object(evaluator.subprocess, 'check_output', return_value=''))
                stack.enter_context(mock.patch('builtins.print'))
                native = NativeFixture(fail_on_third_encode=True)
                factory = stack.enter_context(mock.patch.object(evaluator, 'FrozenWeTok', return_value=native))
                argv = ['evaluate_interfaces.py', '--milestone', '1000', '--execute']
                with mock.patch.object(sys, 'argv', argv), self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    evaluator.main()
                output = output_for(study, 'evaluation') / 'additional_0001000'
                committed = output / 'images/000/receipt.json'
                committed_hash = sha256(committed)
                self.assertFalse((output / 'completion.json').exists())
                factory.return_value = NativeFixture()
                with mock.patch.object(sys, 'argv', argv + ['--resume']):
                    evaluator.main()
                self.assertEqual(sha256(committed), committed_hash)
                self.assertEqual(sha256(reference_file), reference_hash)
                completion = json.loads((output / 'completion.json').read_text())
                self.assertEqual(completion['rows'], 672)
                self.assertEqual(completion['separate_fixed_support_rows'], 12)
                self.assertEqual(completion['noiseless_rows'], 18)
                self.assertEqual(completion['reference_images_copied'], 0)
                for relative, expected_hash in completion['output_hashes'].items():
                    self.assertEqual(sha256(output / relative), expected_hash)
                self.assertEqual(completion['evaluation_sessions'], 2)
                self.assertFalse(completion['evaluation_total_time_is_lower_bound'])
                self.assertGreaterEqual(completion['evaluation_observed_wall_hours_all_sessions'], completion['evaluation_wall_hours_this_session'] * .9)
                self.assertEqual(factory.return_value.calls, 2)
                self.assertTrue(all(network.transmit_calls == 24 for network in instances.values()))
                with (output / 'per_frame.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                row = next(row for row in rows if row['arm'] == 'continuous_mean__single_pass' and int(row['image_index']) == 0)
                expected_value = float((torch.tanh(torch.tensor(.1)) + 1) / 2)
                with np.load(row['image_archive'], allow_pickle=False) as images:
                    actual_image = images['images'][int(row['image_ref'])]
                    self.assertTrue(np.all(actual_image == np.float32(expected_value)))
                    self.assertFalse(any(np.all(image == .2) for image in images['images']))
                self.assertAlmostEqual(float(row['lpips']), (expected_value - .25) ** 2, places=6)


if __name__ == '__main__':
    unittest.main()
