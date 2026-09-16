"""Synthetic CPU wiring test, not a geometry quality experiment."""

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
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import evaluate_geometry as evaluator
from wetok_comm.common import sha256
from wetok_comm.deep_support import SUPPORT_NAME
from wetok_comm.geometry_evaluation import method_definitions
from wetok_comm.interface_evaluation import SYSTEM_REFERENCES
from test_interface_pipeline import CPUTorchProxy, CommunicationFixture, NativeFixture, SupportModelFixture, fixture_metrics


class GeometryPipelineTests(unittest.TestCase):
    def test_real_engine_routes_models_controls_and_supplement_separately(self):
        torch.set_num_threads(2)
        config = yaml.safe_load((EXPERIMENT / 'configs/geometry_study.yaml').read_text())
        config['parent_step'] = 2000
        evaluation = yaml.safe_load((EXPERIMENT / 'configs/geometry_evaluation.yaml').read_text())
        base = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        deep = yaml.safe_load((EXPERIMENT / 'configs/deep_support.yaml').read_text())
        deep['source_hashes'] = {}
        definitions = method_definitions(config)
        source = torch.full((1, 3, 256, 256), .25)
        codes = np.zeros((1, 1, 16, 16, 4), dtype=np.uint8)
        image_value = float((torch.tanh(torch.tensor(.1)) + 1) / 2)
        with TemporaryDirectory() as temporary:
            project = Path(temporary)
            output_for = lambda unused, kind: project / 'outputs' / kind
            training = output_for(config, 'training')
            milestone = training / 'milestones/total_0007000.json'
            milestone.parent.mkdir(parents=True)
            milestone.write_text(json.dumps({'status': 'GEOMETRY_MILESTONE_COMPLETE', 'total_updates_per_new_arm': 7000,
                                             'image_updates_per_new_arm': 5000, 'source_hashes': {}}))
            review = output_for(config, 'analysis') / 'calibration_total_0007000/completion.json'
            review.parent.mkdir(parents=True)
            review.write_text(json.dumps({'source_hashes': {}, 'data_noise_phase_Adam_power_selection_audit': 'PASS',
                                          'milestone_sha256': sha256(milestone)}))
            old_file = project / 'old_reference.npz'
            old_image = torch.full((3, 256, 256), .2)
            np.savez_compressed(old_file, images=old_image[None].numpy())
            old_hash = sha256(old_file)

            class References:
                def __init__(self, unused):
                    pass

                def control_image(self, index, identifier, snr, seed, variant, noise_hash):
                    image = torch.full((3, 256, 256), image_value)
                    return image, fixture_metrics(source, [image], None, None)[0]

                def image(self, index, identifier, snr, seed, name, noise_hash):
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    scores = fixture_metrics(source, [old_image], None, None)[0]
                    return old_image.clone(), {**scores, 'header_uses': header, 'data_uses': 3060 - header,
                        'raw_bits': '', 'coded_bits': '', 'bit_error_rate': '', 'crc_accepted': '', 'image_ref': 0,
                        'online_TX_seconds': '', 'receiver_seconds': '', 'selected_step': ''}, str(old_file)

            class SupportReferences:
                def __init__(self, unused_config, unused_base):
                    pass

                def image(self, index, identifier, snr, seed):
                    image = torch.full((3, 256, 256), .3)
                    scores = fixture_metrics(source, [image], None, None)[0]
                    return image, {'psnr_db': scores['psnr_db'], 'lpips_alex': scores['lpips'], 'dino_cosine': scores['dino']}

            def registry(unused_config, unused_base, unused_qualification, unused_milestone, unused_training, device):
                networks = {name: CommunicationFixture('continuous_mean').eval().requires_grad_(False) for name in definitions}
                return networks, {name: {'step': 5000} for name in definitions}

            def check_rows(rows, unused_config, unused_base):
                self.assertEqual(len(rows), 210)
                expected = {(0, snr, seed, name) for snr in base['evaluation']['snrs_db'] for seed in base['evaluation']['noise_seeds']
                            for name in list(definitions) + list(SYSTEM_REFERENCES)}
                self.assertEqual({(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']) for row in rows}, expected)
                return {}, []

            def check_support(rows, support, unused_config, unused_base):
                self.assertEqual(len(support), 6)
                self.assertTrue(all(row['arm'] == SUPPORT_NAME for row in support))
                return [], []

            with ExitStack() as stack:
                patches = {'PROJECT': project, 'torch': CPUTorchProxy(), 'configure_torch': lambda: None,
                    'require_uncontended_gpu': lambda: None, 'load_geometry_evaluation': lambda: (evaluation, config, base, {}),
                    'geometry_output': output_for, 'load_deep_support': lambda: deep,
                    'FrozenWeTok': lambda unused_device, unused_part: NativeFixture(),
                    'metric_models': lambda unused: (nn.Identity(), nn.Identity()), 'FrozenDeepSupport': SupportModelFixture,
                    'GeometryReferences': References, 'DeepSupportReferences': SupportReferences, 'model_registry': registry,
                    'read_population': lambda unused, population: (source, codes, ['synthetic_geometry_source']),
                    'metrics': fixture_metrics, 'validate_geometry_rows': check_rows, 'geometry_supplement_statistics': check_support}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(evaluator, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(sys, 'argv', ['evaluate_geometry.py', '--total', '7000', '--execute']):
                    evaluator.main()
            output = output_for(config, 'evaluation') / 'total_0007000'
            completion = json.loads((output / 'completion.json').read_text())
            self.assertEqual(completion['rows'], 210)
            self.assertEqual(completion['noiseless_rows'], 6)
            self.assertEqual(completion['separate_fixed_support_rows'], 6)
            self.assertEqual(completion['maximum_control_pixel_error'], 0)
            for relative, expected_hash in completion['output_hashes'].items():
                self.assertEqual(sha256(output / relative), expected_hash)
            self.assertEqual(sha256(old_file), old_hash)
            with (output / 'per_frame.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            models = [row for row in rows if row['arm'] in definitions]
            self.assertEqual(len(models), 126)
            self.assertTrue(all(row['selected_image_step'] == '5000' and row['available_image_updates'] == '5000' for row in models))
            self.assertTrue(all(float(row['control_replay_pixel_error']) == 0 for row in models if row['geometry'] == '153x40'))
            with np.load(models[0]['image_archive'], allow_pickle=False) as data:
                self.assertTrue(np.all(data['images'][int(models[0]['image_ref'])] == np.float32(image_value)))
                self.assertFalse(any(np.all(image == .2) for image in data['images']))


if __name__ == '__main__':
    unittest.main()
