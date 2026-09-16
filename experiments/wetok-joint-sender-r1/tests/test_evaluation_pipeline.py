"""Synthetic CPU execution of the actual nine-model evaluator, including interruption/resume."""

from contextlib import ExitStack
import csv
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
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'tests')]

from joint_sender.evaluation import fresh_names, all_names
from joint_sender.evaluation_io import digest
from wetok_comm.common import sha256
from wetok_comm.evaluation import raw_noise
from test_interface_pipeline import CPUTorchProxy, NativeFixture, CommunicationFixture, SupportModelFixture, fixture_metrics

spec = importlib.util.spec_from_file_location('joint_image_evaluator_fixture', EXPERIMENT / 'scripts/evaluate.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class ReceiverFixture(CommunicationFixture):
    def __init__(self, variant, joint=False):
        super().__init__('continuous_mean')
        self.variant, self.joint = variant, joint
        self.observations = []

    def transmit(self, source, snrs):
        self.transmit_calls += 1
        signal = torch.ones(len(source), 3060, 2)
        return -signal if self.joint else signal

    def receive(self, received, snrs):
        self.observations.append(received.clone())
        return {**super().receive(received, snrs), 'feature_gates': [], 'residual_noise_ratios': []}


class JointImagePipelineTests(unittest.TestCase):
    def test_real_evaluator_keeps_all_controls_distinguishes_y_and_resumes(self):
        torch.set_num_threads(2)
        config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        reference = yaml.safe_load((INNOVATION / 'configs/study.yaml').read_text())
        base = yaml.safe_load((BASE / 'configs/study.yaml').read_text())
        evaluation = yaml.safe_load((EXPERIMENT / 'configs/evaluation.yaml').read_text())
        deep_config = yaml.safe_load((BASE / 'configs/deep_support.yaml').read_text())
        deep_config['source_hashes'] = {}
        source = torch.full((2, 3, 256, 256), .25)
        codes = np.zeros((2, 1, 16, 16, 4), dtype=np.uint8)
        identifiers = ['synthetic_0', 'synthetic_1']
        fresh = fresh_names(config, reference)
        with TemporaryDirectory() as temporary:
            project = Path(temporary)
            output_for = lambda unused, kind: project / 'outputs' / kind
            paths = [project / name for name in ('joint.json', 'review.json', 'control.json', 'control_review.json', 'qualification.json')]
            for path in paths:
                path.write_text('{}')
            milestone = {'selected': {name: {'step': 1000, 'checkpoint_sha256': 'f' * 64} for name in config['variants']}}
            control = {'selected': {name: {'step': 1000, 'checkpoint_sha256': 'f' * 64} for name in reference['variants']},
                       'frozen': {'encoder': 'b' * 64}}
            output_value = float((torch.tanh(torch.tensor(.1)) + 1) / 2)
            frozen_image = torch.full((3, 256, 256), output_value)
            reference_image = torch.full_like(frozen_image, .2)
            support_image = torch.full_like(frozen_image, .3)
            old_archive = project / 'old.npz'
            np.savez_compressed(old_archive, images=torch.stack([frozen_image, reference_image, support_image]).numpy())
            old_hash = sha256(old_archive)

            def previous(image, image_ref, name='', index=0):
                header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                return {**fixture_metrics(source[index:index + 1], [image], None, None)[0],
                    'header_uses': header, 'data_uses': 3060 - header, 'raw_bits': '', 'coded_bits': '', 'bit_error_rate': '',
                    'online_TX_seconds': '', 'receiver_seconds': '', 'image_ref': image_ref, 'image_archive': str(old_archive),
                    'image_sha256': digest(image)}

            class References:
                def __init__(self, unused):
                    self.qualification_path = paths[4]
                    self.clean_rows = {index: fixture_metrics(source[index:index + 1], [torch.zeros_like(frozen_image)], None, None)[0] for index in range(2)}

                def signal(self, index, identifier, snr):
                    return torch.ones(1, 3060, 2)

                def observed(self, index, identifier, snr, seed):
                    noise = torch.tensor(raw_noise(identifier, seed)[None], dtype=torch.float32)
                    return self.signal(index, identifier, snr) + noise * torch.pow(10., torch.tensor([snr]) / 10).rsqrt()[:, None, None]

                def image(self, index, identifier, snr, seed, name, noise_hash):
                    image, image_ref = (frozen_image, 0) if name.startswith('frozen__') else (reference_image, 1)
                    return image.clone(), previous(image, image_ref, name, index)

                def noiseless(self, index, identifier, variant):
                    return frozen_image.clone(), previous(frozen_image, 0, index=index)

                def support(self, index, identifier, snr, seed, noise_hash):
                    return support_image.clone(), previous(support_image, 2, index=index)

            registries = []

            def registry(unused_config, unused_reference, unused_base, unused_milestone, unused_control, device):
                parent = CommunicationFixture('continuous_mean').eval().requires_grad_(False)
                networks = {name: ReceiverFixture(name.split('__', 1)[1], name.startswith('joint__')).eval().requires_grad_(False) for name in fresh}
                choices = {name: {'step': 1000, 'checkpoint_sha256': 'f' * 64, 'encoder_sha256': ('a' if name.startswith('joint__') else 'b') * 64,
                    'training_policy': 'joint_E_R' if name.startswith('joint__') else 'receiver_only', 'selected_encoder_differs_from_parent': name.startswith('joint__'),
                    'optimized_parameters': 1} for name in fresh}
                registries.append((parent, networks))
                return parent, networks, choices

            def validate(rows, unused_config, unused_reference, unused_base):
                lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
                expected = {(index, snr, seed, name) for index in range(2) for snr in base['evaluation']['snrs_db']
                            for seed in base['evaluation']['noise_seeds'] for name in all_names(config, reference)}
                self.assertEqual(set(lookup), expected)
                self.assertEqual(len(rows), 672)
                self.assertNotEqual(lookup[0, 1., 2001, 'joint__single_pass']['received_sha256'], lookup[0, 1., 2001, 'frozen__single_pass']['received_sha256'])
                return lookup, all_names(config, reference)

            def diagnostics(rows, clean, noiseless, support, unused_config, unused_reference):
                self.assertEqual((len(clean), len(noiseless), len(support)), (2, 18, 12))

            first_native, resumed_native = NativeFixture(fail_on_third_encode=True), NativeFixture()
            with ExitStack() as stack:
                patches = {'PROJECT': project, 'torch': CPUTorchProxy(), 'configure_torch': lambda: None,
                    'require_uncontended_gpu': lambda: None, 'load_evaluation': lambda: (evaluation, config, reference, base, {}),
                    'matched_milestone': lambda cfg, ref, step: (milestone, paths[0], paths[1], control, paths[2], paths[3]),
                    'References': References, 'load_deep_support': lambda: deep_config, 'output_path': output_for,
                    'model_registry': registry, 'metric_models': lambda device: (nn.Identity(), nn.Identity()),
                    'FrozenDeepSupport': SupportModelFixture, 'read_population': lambda cfg, population: (source, codes, identifiers),
                    'validate_population': lambda value: self.assertEqual(value, identifiers),
                    'gpu_telemetry': lambda: {'uuid': 'synthetic_GPU'}, 'metrics': fixture_metrics,
                    'receive_for_image': lambda model, received, snrs: model.receive(received, snrs),
                    'verify_pruned_result': lambda model, result, received, snrs: 0., 'validate_rows': validate,
                    'validate_diagnostics': diagnostics, 'support_statistics': lambda *args: ([], [])}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(evaluator, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(evaluator, 'FrozenWeTok', return_value=first_native), mock.patch.object(sys, 'argv', ['evaluate.py', '--execute']):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        evaluator.main()
                output = output_for(config, 'evaluation') / 'step_0005000'
                first_receipt_hash = sha256(output / 'images/000/receipt.json')
                self.assertFalse((output / 'images/001/receipt.json').exists())
                with mock.patch.object(evaluator, 'FrozenWeTok', return_value=resumed_native), mock.patch.object(sys, 'argv', ['evaluate.py', '--resume', '--execute']):
                    evaluator.main()
            self.assertEqual(sha256(output / 'images/000/receipt.json'), first_receipt_hash)
            self.assertEqual(sha256(old_archive), old_hash)
            completion = json.loads((output / 'completion.json').read_text())
            self.assertEqual((completion['rows'], completion['noiseless_rows'], completion['support_rows']), (672, 18, 12))
            self.assertFalse(completion['time_lower_bound_due_to_unfinished_sessions'])
            for relative, expected_hash in completion['output_hashes'].items():
                self.assertEqual(sha256(output / relative), expected_hash)
            for parent, networks in registries:
                self.assertEqual(parent.transmit_calls, 8)
                self.assertTrue(all(len(model.observations) == 24 for model in networks.values()))
                self.assertTrue(all(model.transmit_calls == (8 if name.startswith('joint__') else 0) for name, model in networks.items()))
            with (output / 'per_frame.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row['image_archive'] == str(old_archive) for row in rows if not row['arm'].startswith('joint__')))
            with np.load(output / 'images/000/reconstructions.npz', allow_pickle=False) as archive:
                self.assertEqual(len(archive['images']), 1)
                self.assertTrue(np.all(archive['images'][0] == np.float32(output_value)))


if __name__ == '__main__':
    unittest.main()
