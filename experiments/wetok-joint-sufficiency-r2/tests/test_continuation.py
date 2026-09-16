"""CPU fixtures exercise nonzero Adam restoration and the real four-arm continuation driver."""

from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from sufficiency.common import ORIGINS, VARIANTS, restore_systems, tree_digest, validate_optimizer
from grid_controls.common import initial_system as grid_system
from joint_sender.common import initial_system as basic_system
from joint_sender.runtime import backward_paired_batch, tensor_sha256
from wetok_comm.common import sha256
from wetok_comm.geometry_candidate import GeometryJSCC
from wetok_comm.native import indices_to_features
from wetok_comm.training import module_sha256

SPEC = importlib.util.spec_from_file_location('sufficiency_train_fixture', EXPERIMENT / 'scripts/train.py')
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(32, 3, 1)
        self.codec = self.output
        self.requires_grad_(False)

    def decode(self, features):
        return self.output(features).sigmoid()


class Perceptual(nn.Module):
    def forward(self, image, target):
        return (image - target).square().mean((1, 2, 3), keepdim=True)


class CPUTorch:
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0)

    @staticmethod
    def device(unused):
        return torch.device('cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        self.config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
        self.reference = yaml.safe_load((INNOVATION / 'configs/study.yaml').read_text())
        self.base = yaml.safe_load((BASE / 'configs/study.yaml').read_text())
        self.config.update(initial_total_updates=2, planned_total_updates=4, new_full_steps=[3, 4],
                           initial_global_data_step=7002, inherited_full_steps=[0, 2], monitor_every=1)
        self.base['channel']['snrs_db'] = [1., 7.]
        torch.manual_seed(731)
        self.images = torch.rand(4, 3, 16, 16)
        self.truth = indices_to_features(torch.randint(256, (4, 16, 16, 4), dtype=torch.uint8))

    def parent(self, *args):
        torch.manual_seed(732)
        return GeometryJSCC('single_pass', dict(width=24, heads=2, encoder_layers=1, receiver_layers=1,
            channel_positions=204, channel_features=30)).eval().requires_grad_(False)

    def decoder(self, *args):
        torch.manual_seed(733)
        return Decoder()

    def batches(self, unused_base, count, start, until):
        for step in range(start, until):
            generator = torch.Generator().manual_seed(734 + step)
            yield {'step': step, 'fingerprint': f'synthetic_{step}', 'noise': torch.randn(4, 3060, 2, generator=generator)}

    def inputs(self, unused_population, batch, device):
        return {'source_fq': self.truth, 'images': self.images, 'snrs': torch.tensor([1., 7., 1., 7.]), 'noise': batch['noise']}

    def source_fixture(self, root):
        parent, decoder, perceptual = self.parent(), self.decoder(), Perceptual()
        states = {origin: {'models': {}, 'optimizers': {}} for origin in ('joint', 'grid')}
        histories, selected, calibration_files = {}, {}, {}
        for name in VARIANTS:
            factory = basic_system if ORIGINS[name] == 'joint' else grid_system
            model = factory(parent, name, self.reference, 'cpu')
            optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=1e-4, betas=(.9, .99), weight_decay=1e-4)
            histories[name] = []
            for index, batch in enumerate(self.batches(self.base, 20000, 7000, 7002)):
                report = backward_paired_batch(model, self.inputs(None, batch, 'cpu'), decoder, perceptual, self.config['unchanged_loss'])
                norm = float(torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1., error_if_nonfinite=True))
                optimizer.step()
                histories[name].append({'step': index+1, 'global_data_step': batch['step']+1,
                    'batch_sha256': batch['fingerprint'], 'gradient_norm': norm, 'step_seconds': 0.,
                    'peak_GPU_allocated_bytes': 0, **report})
            origin = states[ORIGINS[name]]
            origin['models'][name] = copy.deepcopy(model.state_dict())
            origin['optimizers'][name] = copy.deepcopy(optimizer.state_dict())
            checkpoint = root / f'old_{name}.pt'
            torch.save({'model': model.state_dict(), 'variant': name, 'step': 2}, checkpoint)
            selected[name] = {'step': 2, 'scope': 'full', 'source_images': 1000, 'lpips': .01,
                'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'checkpoint_origin': ORIGINS[name]}
            calibration_files[name] = {}
        return {'states': states, 'hashes': {'joint': 'sealed_joint', 'grid': 'sealed_grid'},
            'frozen': {'parent': module_sha256(parent), 'codec': module_sha256(decoder.codec), 'lpips': module_sha256(perceptual)},
            'histories': histories, 'selected': selected, 'summaries': [], 'calibrated': [0, 2],
            'calibration_files': calibration_files, 'prior_timings': {name: {'training': 0., 'calibration': 0.} for name in VARIANTS}}

    def test_optimizer_loader_rejects_reset_moments_or_different_parameter_count(self):
        with TemporaryDirectory() as temporary:
            sources = self.source_fixture(Path(temporary))
            models, optimizers, audit = restore_systems(self.config, self.reference, self.base, copy.deepcopy(sources), self.parent(), 'cpu')
            for name in VARIANTS:
                original = sources['states'][ORIGINS[name]]['optimizers'][name]
                self.assertEqual(tree_digest(optimizers[name].state_dict()), tree_digest(original))
                self.assertEqual(audit[name]['optimizer_steps'], 2)
                self.assertTrue(any(bool(state['exp_avg'].abs().sum() > 0) for state in original['state'].values()))
            name = VARIANTS[0]
            changed = copy.deepcopy(sources['states'][ORIGINS[name]]['optimizers'][name])
            next(iter(changed['state'].values()))['step'].zero_()
            with self.assertRaisesRegex(RuntimeError, 'reset'):
                validate_optimizer(changed, list(models[name].parameters()), self.config, self.base, 2)
            changed = copy.deepcopy(sources['states'][ORIGINS[name]]['optimizers'][name])
            changed['param_groups'][0]['params'].pop()
            with self.assertRaisesRegex(RuntimeError, 'coverage'):
                validate_optimizer(changed, list(models[name].parameters()), self.config, self.base, 2)

    def test_actual_driver_continues_adam_and_data_without_repeating_initial_calibration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = self.source_fixture(root)
            old_hashes = {name: sha256(choice['checkpoint']) for name, choice in sources['selected'].items()}
            unused_models, unused_optimizers, restored = restore_systems(self.config, self.reference, self.base,
                copy.deepcopy(sources), self.parent(), 'cpu')
            first_batch = next(self.batches(self.base, 20000, 7002, 7003))
            preparation = root / 'preparation'
            preparation.mkdir()
            (preparation / 'resume_initialization.json').write_text(json.dumps({
                'status': 'FOUR_MODEL_AND_ADAM_RESTORATION_PASS_NO_NEW_UPDATES', 'optimizer_updates': 0,
                'source_endpoint_hashes': sources['hashes'], 'source_hashes': {}, 'restored': restored,
                'first_new_batch_sha256': first_batch['fingerprint'], 'first_new_noise_sha256': tensor_sha256(first_batch['noise'])}))
            paths = {'preparation': preparation, 'training': root / 'resumed'}
            calls = []

            def calibrate(model, population, decoder, perceptual, ref, base, device, positions=None):
                calls.append(model.variant)
                magnitude = float(next(model.encoder.parameters()).detach().square().mean())
                return [{'image_id': 'synthetic_calibration', 'snr_db': snr, 'psnr_db': 20 + magnitude,
                    'lpips': .2 + magnitude, 'ssim': .8, 'bit_error_rate': .1} for snr in base['channel']['snrs_db']]

            with ExitStack() as stack:
                replacements = {'torch': CPUTorch(), 'configure_torch': lambda: None, 'require_uncontended_gpu': lambda: None,
                    'settings': lambda: (self.config, {}, {}, self.reference, self.base, {}),
                    'load_sources': lambda cfg: copy.deepcopy(sources), 'load_parent': self.parent,
                    'output_path': lambda cfg, kind: paths[kind], 'FrozenWeTok': self.decoder,
                    'load_lpips': lambda device: Perceptual(), 'read_population': lambda cfg, name: name,
                    'monitoring_indices': lambda *args: [], 'paired_batches': self.batches, 'batch_inputs': self.inputs,
                    'calibration': calibrate}
                for name, value in replacements.items():
                    stack.enter_context(mock.patch.object(TRAINER, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '3', '--execute']):
                    TRAINER.main()
                self.assertEqual(len(calls), 4)
                first_checkpoint = paths['training'] / 'milestones/step_0000003_optimizer.pt'
                first_hash = sha256(first_checkpoint)
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '4', '--resume', '--execute']):
                    TRAINER.main()
                resumed = torch.load(paths['training'] / 'resume.pt', map_location='cpu', weights_only=True)
                self.assertEqual(sha256(first_checkpoint), first_hash)
                paths['training'] = root / 'uninterrupted'
                with mock.patch.object(sys, 'argv', ['train.py', '--until-update', '4', '--execute']):
                    TRAINER.main()
                uninterrupted = torch.load(paths['training'] / 'resume.pt', map_location='cpu', weights_only=True)
            self.assertEqual((resumed['completed_total_updates'], resumed['completed_new_updates'], resumed['global_data_step']), (4, 2, 7004))
            self.assertEqual(resumed['calibrated'], [0, 2, 3, 4])
            self.assertEqual(tree_digest(resumed['models']), tree_digest(uninterrupted['models']))
            self.assertEqual(tree_digest(resumed['optimizers']), tree_digest(uninterrupted['optimizers']))
            for name in VARIANTS:
                self.assertEqual(sha256(sources['selected'][name]['checkpoint']), old_hashes[name])
                self.assertEqual(resumed['selected'][name]['step'], 2)
                self.assertEqual(resumed['selected'][name]['checkpoint'], sources['selected'][name]['checkpoint'])
                self.assertEqual([row['global_data_step'] for row in resumed['histories'][name]], [7001, 7002, 7003, 7004])
                self.assertTrue(all(int(state['step']) == 4 for state in resumed['optimizers'][name]['state'].values()))
                self.assertEqual(tree_digest(resumed['histories'][name][:2]), tree_digest(sources['histories'][name]))
                self.assertEqual(resumed['histories'][name][-1]['received_sha256'], uninterrupted['histories'][name][-1]['received_sha256'])


if __name__ == '__main__':
    unittest.main()
