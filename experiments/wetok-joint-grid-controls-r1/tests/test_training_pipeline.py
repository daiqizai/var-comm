"""CPU fixtures test the actual grid training driver and its immutable paired resume."""

from contextlib import ExitStack
import csv
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from grid_controls.common import initial_system
from joint_sender.runtime import tensor_sha256
from wetok_comm.common import sha256
from wetok_comm.geometry_candidate import GeometryJSCC
from wetok_comm.native import indices_to_features
from wetok_comm.training import module_sha256
from test_grid_controls import ToyDecoder, ToyPerceptual

SPEC = importlib.util.spec_from_file_location('grid_training_fixture', EXPERIMENT / 'scripts/train.py')
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class CPUTorch:
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0)

    @staticmethod
    def device(unused):
        return torch.device('cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


class TrainingPipelineTests(unittest.TestCase):
    def test_grid_driver_resume_matches_uninterrupted_and_rejects_changed_noise(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        original = yaml.safe_load((JOINT / 'configs/study.yaml').read_text())
        reference = yaml.safe_load((INNOVATION / 'configs/study.yaml').read_text())
        base = yaml.safe_load((BASE / 'configs/study.yaml').read_text())
        config['full_calibration_steps'], config['monitor_every'], config['planned_updates'] = [0, 1, 2], 1, 2
        base['channel']['snrs_db'] = [1., 7.]

        def parent_factory(unused_reference, unused_base, device):
            torch.manual_seed(931)
            return GeometryJSCC('single_pass', dict(width=24, heads=2, encoder_layers=1, receiver_layers=1,
                channel_positions=204, channel_features=30)).eval().requires_grad_(False)

        def visual_factory(device, part):
            torch.manual_seed(934)
            decoder = ToyDecoder()
            decoder.codec = decoder.output
            return decoder

        torch.manual_seed(936)
        images = torch.rand(4, 3, 16, 16)
        source = indices_to_features(torch.randint(256, (4, 16, 16, 4), dtype=torch.uint8))

        def batches(unused_base, count, start, until):
            for step in range(start, until):
                generator = torch.Generator().manual_seed(937 + step)
                yield {'step': step, 'fingerprint': f'synthetic_batch_{step}',
                       'noise': torch.randn(4, 3060, 2, generator=generator)}

        def inputs(unused_population, batch, device):
            return {'source_fq': source, 'images': images, 'snrs': torch.tensor([1., 7., 1., 7.]), 'noise': batch['noise']}

        def calibrated(system, population, decoder, perceptual, ref, base, device, positions=None):
            magnitude = float(next(system.encoder.parameters()).detach().square().mean())
            return [{'image_id': 'synthetic_calibration', 'snr_db': snr, 'psnr_db': 20 + magnitude,
                     'lpips': .2 + magnitude, 'ssim': .8, 'bit_error_rate': .1, 'feature_mse': .2} for snr in base['channel']['snrs_db']]

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path = root / 'outputs' / original['outputs']['training'] / 'single_pass/training.csv'
            trace_path.parent.mkdir(parents=True)
            trace = [{'batch_sha256': batch['fingerprint'], 'global_data_step': batch['step'] + 1,
                      'paired_standard_noise_sha256': tensor_sha256(batch['noise'])} for batch in batches(base, 20000, 7000, 7002)]

            def write_trace():
                with trace_path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
                    writer.writeheader()
                    writer.writerows(trace)

            write_trace()
            parent = parent_factory(reference, base, 'cpu')
            decoder = visual_factory('cpu', 'decoder')
            perceptual = ToyPerceptual()
            reference_hashes = {'completed_original_trial': 'synthetic_qualified_reference'}
            profile = {'status': 'GRID_IMAGE_GRADIENT_PROFILE_PASS', 'optimizer_updates': 0, 'source_hashes': {},
                'reference_hashes': reference_hashes,
                'frozen': {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)},
                'model_hashes_before': {name: module_sha256(initial_system(parent, name, reference, 'cpu')) for name in config['variants']}}
            profile_root = root / 'profile'
            profile_root.mkdir()
            (profile_root / 'profile.json').write_text(json.dumps(profile))
            paths = {'profile': profile_root, 'training': root / 'resumed'}
            with ExitStack() as stack:
                patches = {'PROJECT': root, 'torch': CPUTorch(), 'configure_torch': lambda: None,
                    'require_uncontended_gpu': lambda: None, 'settings': lambda: (config, original, reference, base, {}),
                    'completed_joint_reference': lambda *args: {'hashes': reference_hashes}, 'load_parent': parent_factory,
                    'FrozenWeTok': visual_factory, 'load_lpips': lambda device: ToyPerceptual(),
                    'output_path': lambda cfg, kind: paths[kind], 'read_population': lambda cfg, population: population,
                    'monitoring_indices': lambda population, cfg: [], 'paired_batches': batches, 'batch_inputs': inputs, 'calibration': calibrated}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(TRAINER, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '1', '--execute']):
                    TRAINER.main()
                first_checkpoint = paths['training'] / 'milestones/step_0000001_optimizer.pt'
                first_hash = sha256(first_checkpoint)
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '2', '--resume', '--execute']):
                    TRAINER.main()
                resumed = torch.load(paths['training'] / 'resume.pt', map_location='cpu', weights_only=True)
                self.assertEqual(sha256(first_checkpoint), first_hash)
                paths['training'] = root / 'uninterrupted'
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '2', '--execute']):
                    TRAINER.main()
                uninterrupted = torch.load(paths['training'] / 'resume.pt', map_location='cpu', weights_only=True)
                paths['training'] = root / 'invalid_noise'
                trace[0]['paired_standard_noise_sha256'] = 'not_the_original_noise'
                write_trace()
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '1', '--execute']):
                    with self.assertRaisesRegex(RuntimeError, 'standard noise differs'):
                        TRAINER.main()
            self.assertEqual(resumed['completed_grid_updates'], 2)
            self.assertEqual(resumed['calibrated'], [0, 1, 2])
            for name in config['variants']:
                self.assertEqual(len(resumed['histories'][name]), 2)
                self.assertEqual([row['global_data_step'] for row in resumed['histories'][name]], [7001, 7002])
                self.assertTrue(all(int(state['step']) == 2 for state in resumed['optimizers'][name]['state'].values()))
                for key, value in resumed['models'][name].items():
                    torch.testing.assert_close(value, uninterrupted['models'][name][key], rtol=0, atol=0)
                for row, expected in zip(resumed['histories'][name], uninterrupted['histories'][name]):
                    self.assertEqual(row['received_sha256'], expected['received_sha256'])
                    self.assertEqual(row['loss'], expected['loss'])


if __name__ == '__main__':
    unittest.main()
