"""Synthetic CPU execution/resume tests; none of these outputs are scientific results."""

from contextlib import ExitStack
import csv
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
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'scripts'), str(EXPERIMENT / 'src'), str(REFERENCE / 'src'), str(REFERENCE / 'tests')]

import evaluate as evaluator
from innovation_comm.evaluation import receiver_name, reference_names
from wetok_comm.common import sha256
from test_interface_pipeline import CPUTorchProxy, CommunicationFixture, NativeFixture, SupportModelFixture, fixture_metrics


class ReceiverFixture(CommunicationFixture):
    def __init__(self, variant):
        super().__init__('continuous_mean')
        self.variant = variant
        self.observations = []

    def receive(self, received, snrs):
        self.observations.append(received.clone())
        return {**super().receive(received, snrs), 'feature_gates': [], 'residual_noise_ratios': []}


class ReceiverPipelineTests(unittest.TestCase):
    def test_same_observation_references_failure_and_resume_in_actual_engine(self):
        torch.set_num_threads(2)
        config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        evaluation = yaml.safe_load((EXPERIMENT / 'configs/evaluation.yaml').read_text())
        base = yaml.safe_load((REFERENCE / 'configs/study.yaml').read_text())
        deep = yaml.safe_load((REFERENCE / 'configs/deep_support.yaml').read_text())
        deep['source_hashes'] = {}
        source = torch.full((2, 3, 256, 256), .25)
        codes = np.zeros((2, 1, 16, 16, 4), dtype=np.uint8)
        identifiers = ['synthetic_0', 'synthetic_1']
        methods = [receiver_name(variant) for variant in config['variants']]
        with TemporaryDirectory() as temporary:
            project = Path(temporary)
            output_for = lambda unused, kind: project / 'outputs' / kind
            milestone_path, review_path = project / 'milestone.json', project / 'review.json'
            milestone = {'receiver_updates_per_arm': 1000, 'frozen': {'encoder': 'e' * 64},
                         'selected': {variant: {'step': 1000, 'checkpoint_sha256': 'f' * 64} for variant in config['variants']}}
            milestone_path.write_text(json.dumps(milestone))
            review_path.write_text('{}')
            old_file = project / 'old_reference.npz'
            old_image = torch.full((3, 256, 256), .2)
            np.savez_compressed(old_file, images=old_image[None].numpy())
            old_hash = sha256(old_file)

            class References:
                def __init__(self, unused):
                    pass

                def parent_signal(self, index, snr):
                    return torch.ones(1, 3060, 2)

                def image(self, index, identifier, snr, seed, name, noise_hash):
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    scores = fixture_metrics(source[index:index + 1], [old_image], None, None)[0]
                    return old_image.clone(), {**scores, 'header_uses': header, 'data_uses': 3060 - header,
                        'raw_bits': '', 'coded_bits': '', 'bit_error_rate': '', 'crc_accepted': '', 'image_ref': 0,
                        'online_TX_seconds': '', 'receiver_seconds': '', 'selected_step': ''}, str(old_file)

            class SupportReferences:
                def __init__(self, unused_config, unused_base):
                    pass

                def image(self, index, identifier, snr, seed):
                    image = torch.full((3, 256, 256), .3)
                    scores = fixture_metrics(source[index:index + 1], [image], None, None)[0]
                    return image, {'psnr_db': scores['psnr_db'], 'lpips_alex': scores['lpips'], 'dino_cosine': scores['dino']}

            registries = []

            def registry(unused_config, unused_base, unused_milestone, device):
                parent = CommunicationFixture('continuous_mean').eval().requires_grad_(False)
                networks = {receiver_name(variant): ReceiverFixture(variant).eval().requires_grad_(False) for variant in config['variants']}
                registries.append((parent, networks))
                return parent, networks, {name: {'step': 1000, 'checkpoint_sha256': 'f' * 64, 'receiver_trainable_parameters': 1} for name in networks}

            def check_rows(rows, unused_config, unused_base):
                expected = {(index, snr, seed, name) for index in range(2) for snr in base['evaluation']['snrs_db']
                            for seed in base['evaluation']['noise_seeds'] for name in methods + reference_names()}
                lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
                self.assertEqual(set(lookup), expected)
                for index in range(2):
                    for snr in base['evaluation']['snrs_db']:
                        for seed in base['evaluation']['noise_seeds']:
                            paired = [lookup[index, snr, seed, name] for name in methods]
                            self.assertEqual(len({row['shared_received_sha256'] for row in paired}), 1)
                            self.assertEqual(len({row['shared_transmitted_sha256'] for row in paired}), 1)
                return lookup, methods + reference_names()

            def check_native(rows, clean, noiseless, unused_config, unused_base):
                self.assertEqual(len(clean), 2)
                self.assertEqual(len(noiseless), 12)

            def check_support(rows, support, unused_config, unused_base):
                self.assertEqual(len(support), 12)
                return [], []

            first_native = NativeFixture(fail_on_third_encode=True)
            second_native = NativeFixture()
            with ExitStack() as stack:
                patches = {'PROJECT': project, 'torch': CPUTorchProxy(), 'configure_torch': lambda: None,
                    'require_uncontended_gpu': lambda: None, 'load_evaluation': lambda: (evaluation, config, base, {}),
                    'gpu_telemetry': lambda: {'local_time': 'synthetic', 'uuid': 'fixture_GPU'},
                    'receive_for_image': lambda model, received, snrs: model.receive(received, snrs),
                    'verify_pruned_result': lambda model, result, received, snrs: 0.,
                    'output_path': output_for, 'load_deep_support': lambda: deep,
                    'audited_milestone': lambda unused, step: (milestone, milestone_path, review_path),
                    'metric_models': lambda unused: (nn.Identity(), nn.Identity()), 'FrozenDeepSupport': SupportModelFixture,
                    'References': References, 'DeepSupportReferences': SupportReferences, 'model_registry': registry,
                    'read_population': lambda unused, population: (source, codes, identifiers),
                    'validate_population': lambda actual: self.assertEqual(actual, identifiers),
                    'metrics': fixture_metrics, 'validate_rows': check_rows, 'validate_native_reference': check_native,
                    'supplement_statistics': check_support}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(evaluator, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(evaluator, 'FrozenWeTok', return_value=first_native), mock.patch.object(sys, 'argv', ['evaluate.py', '--step', '1000', '--execute']):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        evaluator.main()
                output = output_for(config, 'evaluation') / 'step_0001000'
                self.assertTrue((output / 'images/000/receipt.json').exists())
                self.assertFalse((output / 'images/001/receipt.json').exists())
                first_receipt_hash = sha256(output / 'images/000/receipt.json')
                with mock.patch.object(evaluator, 'FrozenWeTok', return_value=second_native), mock.patch.object(sys, 'argv', ['evaluate.py', '--step', '1000', '--resume', '--execute']):
                    evaluator.main()
            self.assertEqual(sha256(output / 'images/000/receipt.json'), first_receipt_hash)
            self.assertEqual(sha256(old_file), old_hash)
            completion = json.loads((output / 'completion.json').read_text())
            self.assertEqual(completion['rows'], 546)
            self.assertEqual(completion['noiseless_rows'], 12)
            self.assertEqual(completion['evaluation_sessions'], 2)
            self.assertFalse(completion['evaluation_total_time_is_lower_bound'])
            self.assertEqual(completion['maximum_parent_signal_error'], 0)
            with (output / 'hardware_telemetry.csv').open() as handle:
                hardware = list(csv.DictReader(handle))
            self.assertEqual(len(hardware), 4)
            for relative, expected in completion['output_hashes'].items():
                self.assertEqual(sha256(output / relative), expected)
            for parent, networks in registries:
                self.assertEqual(parent.transmit_calls, 14)
                self.assertTrue(all(network.transmit_calls == 0 for network in networks.values()))
                histories = [network.observations for network in networks.values()]
                self.assertTrue(all(len(history) == 24 for history in histories))
                self.assertTrue(all(torch.equal(values, other) for history in histories[1:] for values, other in zip(histories[0], history)))
            self.assertEqual(second_native.calls, 2)
            with (output / 'per_frame.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            for index in range(2):
                with np.load(output / f'images/{index:03d}/waveforms.npz', allow_pickle=False) as signals:
                    for row in rows:
                        if int(row['image_index']) != index or row['arm'] not in methods:
                            continue
                        key = f'received__snr{row["snr_db"]}__seed{row["seed"]}'
                        self.assertEqual(evaluator.array_sha256(signals[key]), row['shared_received_sha256'])
            self.assertTrue(all(row['receiver_seconds'] == '' for row in rows if row['arm'] in reference_names()))


if __name__ == '__main__':
    unittest.main()
