"""Exercise the actual R3 driver through a real Adam resume, not a synthetic checkpoint counter."""

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from test_model import EXPERIMENT, ToyDecoder, ToyPerceptual, parent_factory

import torch

from joint_sender.runtime import tensor_sha256
from vector_control.common import initial_system, tree_digest
from wetok_comm.common import sha256
from wetok_comm.native import indices_to_features
from wetok_comm.training import module_sha256

SPEC = importlib.util.spec_from_file_location('r3_actual_training_fixture', EXPERIMENT / 'scripts/train.py')
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class CPUTorch:
    class cuda:
        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 0

    @staticmethod
    def device(unused):
        return torch.device('cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


class ActualTrainingTests(unittest.TestCase):
    def test_resumed_model_and_adam_equal_uninterrupted_and_initial_calibration_not_repeated(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        reference = {'training': {'new_parameter_seed': 81}, 'fusion': {'ratio_maximum': 8., 'gate_hidden': 16, 'gate_initial_bias': -2.}}
        base = {'training': {'betas': [.9, .99], 'weight_decay': 1e-4}}
        config = {'variant': 'full_grid_prediction_features', 'full_calibration_steps': [0, 1, 2], 'global_data_offset': 7000,
            'monitor_every': 1, 'learning_rate': 1e-4, 'gradient_clip': 1., 'unchanged_loss': {'mse': 1., 'lpips': .01, 'bits': .01, 'state': .01}}

        def batches(unused_base, population, start, stop):
            for step in range(start, stop):
                generator = torch.Generator().manual_seed(step)
                yield {'step': step, 'fingerprint': f'paired_batch_{step}', 'noise': torch.randn((4, 3060, 2), generator=generator)}

        def inputs(unused_population, batch, device):
            generator = torch.Generator().manual_seed(batch['step'] + 10)
            return {'source_fq': indices_to_features(torch.randint(256, (4, 16, 16, 4), generator=generator, dtype=torch.uint8)),
                'images': torch.rand((4, 3, 16, 16), generator=generator), 'snrs': torch.tensor([1., 4., 13., 19.]), 'noise': batch['noise']}

        def visual(*args):
            torch.manual_seed(509)
            decoder = ToyDecoder()
            decoder.codec = decoder.output
            return decoder

        def parent(*args):
            return parent_factory()

        calibration_calls = []

        def calibrated(*args):
            point = len(calibration_calls) % 2 + 1
            calibration_calls.append(point)
            return [{'image_id': 'fixture_calibration', 'snr_db': 1., 'psnr_db': 20., 'lpips': .2 if point == 1 else .3}]

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {'preparation': root / 'preparation', 'profile': root / 'profile', 'training': root / 'resumed'}
            paths['preparation'].mkdir()
            paths['profile'].mkdir()
            qualified = {'bindings': {'closed_reference': 'synthetic_qualified_reference'},
                'zero_rows': [{'image_id': 'fixture_calibration', 'snr_db': 1., 'psnr_db': 19., 'lpips': .4}],
                'zero_calibration': {'path': str(root / 'old_zero.csv'), 'sha256': 'old_zero_unchanged'},
                'trace': [{'step': index + 1, 'global_data_step': batch['step'] + 1, 'batch_sha256': batch['fingerprint'],
                    'paired_standard_noise_sha256': tensor_sha256(batch['noise'])} for index, batch in enumerate(batches(base, 20000, 7000, 7002))],
                'frozen': {'parent': module_sha256(parent()), 'codec': module_sha256(visual().codec), 'lpips': module_sha256(ToyPerceptual())}}
            qualification_path = paths['preparation'] / 'initialization.json'
            qualification_path.write_text(json.dumps({'status': 'R3_ORIGINAL_INITIALIZATION_AND_COMPLETE_HISTORY_PASS',
                'reference_bindings': qualified['bindings'], 'source_hashes': {}}))
            profile = {'status': 'R3_REAL_INITIAL_FUNCTION_AND_IMAGE_GRADIENT_PROFILE_PASS', 'optimizer_updates': 0,
                'qualification_sha256': sha256(qualification_path), 'reference_bindings': qualified['bindings'], 'source_hashes': {},
                'initial_model_sha256': module_sha256(initial_system(parent(), reference, 'cpu'))}
            (paths['profile'] / 'profile.json').write_text(json.dumps(profile))
            with ExitStack() as stack:
                patches = {'torch': CPUTorch(), 'configure_torch': lambda: None, 'require_uncontended_gpu': lambda: None,
                    'settings': lambda: (config, {}, {}, {}, reference, base, {}), 'qualified_reference': lambda *args: qualified,
                    'output_path': lambda cfg, key: paths[key], 'load_parent': parent, 'FrozenWeTok': visual,
                    'load_lpips': lambda device: ToyPerceptual(), 'read_population': lambda *args: (None, None, []),
                    'monitoring_indices': lambda *args: [], 'paired_batches': batches, 'batch_inputs': inputs, 'calibration': calibrated}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(TRAINER, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '1', '--execute']):
                    TRAINER.main()
                checkpoint = paths['training'] / 'milestones/step_0000001_optimizer.pt'
                first_sha = sha256(checkpoint)
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '2', '--resume', '--execute']):
                    TRAINER.main()
                resumed = torch.load(paths['training'] / 'resume.pt', map_location='cpu', weights_only=True)
                self.assertEqual(sha256(checkpoint), first_sha)
                paths['training'] = root / 'continuous'
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '2', '--execute']):
                    TRAINER.main()
                uninterrupted = torch.load(paths['training'] / 'resume.pt', map_location='cpu', weights_only=True)
            self.assertEqual(tree_digest(resumed['model']), tree_digest(uninterrupted['model']))
            self.assertEqual(tree_digest(resumed['optimizer']), tree_digest(uninterrupted['optimizer']))
            self.assertEqual({int(state['step']) for state in resumed['optimizer']['state'].values()}, {2})
            self.assertEqual(resumed['global_data_step'], 7002)
            self.assertEqual(resumed['selected']['step'], 1)
            self.assertEqual(resumed['calibration_files']['0'], qualified['zero_calibration'])
            self.assertEqual(calibration_calls, [1, 2, 1, 2])


if __name__ == '__main__':
    unittest.main()
