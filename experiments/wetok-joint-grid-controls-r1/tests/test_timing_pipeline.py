"""Exercise the actual timing driver on frozen synthetic quality artifacts and a resumed source boundary."""

from contextlib import ExitStack
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

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from grid_controls.evaluation import learned_names
from joint_sender.evaluation_io import digest, write_rows
from wetok_comm.common import sha256
from wetok_comm.evaluation import raw_noise
from wetok_comm.native import indices_to_features
from wetok_comm.training import module_sha256
from test_quality_evaluation import fixture
from test_quality_pipeline import ModelFixture, NativeFixture

SPEC = importlib.util.spec_from_file_location('grid_timing_driver_fixture', EXPERIMENT / 'scripts/evaluate_timing.py')
TIMER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIMER)


class CPUTorch:
    class cuda:
        @staticmethod
        def synchronize():
            return None

    @staticmethod
    def device(unused):
        return torch.device('cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


class TimingModel(ModelFixture):
    def __init__(self, variant):
        super().__init__(variant)
        self.encoder = nn.Identity()


class TimingPipelineTests(unittest.TestCase):
    def test_real_timing_replays_quality_and_resumes_without_rerendering_committed_sources(self):
        torch.set_num_threads(2)
        config, original, reference, base, unused = fixture()
        names = learned_names(config, original, reference)
        evaluation = {'timing_source_indices': [0, 1], 'timing_snrs_db': [1.], 'timing_noise_seed': 2001}
        images = torch.stack([torch.full((3, 256, 256), value) for value in (.1, .2)])
        codes = np.full((2, 1, 16, 16, 4), 255, dtype=np.uint8)
        identifiers = ['synthetic_source_0', 'synthetic_source_1']
        native = NativeFixture()

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            quality = root / 'evaluation/quality_0005000'
            old = root / 'old'
            rows = []
            for index, identifier in enumerate(identifiers):
                new_waves, old_waves, generated = {}, {}, []
                directory = quality / f'images/{index:03d}'
                directory.mkdir(parents=True)
                old_directory = old / f'images/{index:03d}'
                old_directory.mkdir(parents=True)
                for name in names:
                    signal = np.ones((1, 3060, 2), dtype=np.float32)
                    received = signal + raw_noise(identifier, 2001).astype(np.float32)[None] * np.float32(10**(-1/20))
                    model = TimingModel(name.split('__', 1)[1]).eval()
                    image = native.decode(model.receive(torch.tensor(received), torch.tensor([1.]))['receiver_features'])[0].detach().numpy()
                    waves = new_waves if name.startswith('grid__') else old_waves
                    waves[TIMER.key(name, 1.)] = signal
                    waves[TIMER.key(name, 1., 2001)] = received
                    rows.append({'image_index': index, 'image_id': identifier, 'snr_db': 1., 'seed': 2001, 'arm': name,
                        'checkpoint_sha256': 'frozen_checkpoint', 'encoder_sha256': module_sha256(model.encoder),
                        'transmitted_sha256': digest(signal), 'received_sha256': digest(received),
                        'noise_sha256': digest(raw_noise(identifier, 2001)), 'image_archive': str(directory/'reconstructions.npz'),
                        'image_ref': len(generated), 'image_sha256': digest(image)})
                    generated.append(image)
                np.savez(directory/'waveforms.npz', **new_waves)
                np.savez(old_directory/'waveforms.npz', **old_waves)
                np.savez(directory/'reconstructions.npz', images=np.stack(generated))
            write_rows(quality/'per_frame.csv', rows)
            milestone_path, review_path = root/'milestone.json', root/'review.json'
            milestone_path.write_text('{}'); review_path.write_text('{}')
            (quality/'completion.json').write_text(json.dumps({'status': 'GRID_QUALITY_COMPLETE',
                'grid_milestone_sha256': sha256(milestone_path), 'source_hashes': {}, 'output_hashes': {}}))
            quality_hash = sha256(quality/'per_frame.csv')

            class References:
                root = old
                receipt_sha = 'old_receipt'
                frozen_models = {'native': module_sha256(nn.Identity())}

                def __init__(self, *args):
                    pass

                def validate_source(self, index, identifier, source):
                    if identifier != identifiers[index] or not torch.equal(source, images[index:index+1]):
                        raise RuntimeError('timing source changed')

            def registry(selected_names):
                models = {name: TimingModel(name.split('__', 1)[1]).eval() for name in selected_names}
                return nn.Identity(), models, {name: {'checkpoint_sha256': 'frozen_checkpoint'} for name in selected_names}

            call_count = 0

            def admit():
                nonlocal call_count
                call_count += 1
                if call_count == 4:
                    raise RuntimeError('other GPU compute workloads are present: [999]; not stopping them')

            with ExitStack() as stack:
                patches = {'torch': CPUTorch(), 'configure_torch': lambda: None,
                    'load_evaluation': lambda: (evaluation, config, original, reference, base, {}),
                    'validate_config': lambda cfg: None, 'require_uncontended_gpu': admit,
                    'audited_grid': lambda *args: ({}, milestone_path, review_path, {'joint': {}, 'frozen': {}}),
                    'References': References, 'output_path': lambda *args: root/'evaluation',
                    'model_registry': lambda *args: registry(names[:2]), 'old_model_registry': lambda *args: registry(names[2:]),
                    'FrozenWeTok': lambda *args: NativeFixture(), 'read_population': lambda *args: (images, codes, identifiers),
                    'gpu_telemetry': lambda: {'local_time': 'synthetic'},
                    'receive_for_image': lambda model, received, snrs: model.receive(received, snrs),
                    'verify_pruned_result': lambda *args: 0.,
                    'summarize': lambda values, *args: ([{'rows': len(values)}], [{'scope': 'synthetic_fixture'}])}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(TIMER, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(sys, 'argv', ['evaluate_timing.py', '--execute']):
                    with self.assertRaisesRegex(RuntimeError, 'other GPU compute'):
                        TIMER.main()
                output = root/'evaluation/timing_0005000'
                first_hash = sha256(output/'images/000/receipt.json')
                with mock.patch.object(TIMER, 'require_uncontended_gpu', lambda: None), \
                        mock.patch.object(sys, 'argv', ['evaluate_timing.py', '--resume', '--execute']):
                    TIMER.main()
            self.assertEqual(sha256(output/'images/000/receipt.json'), first_hash)
            self.assertEqual(sha256(quality/'per_frame.csv'), quality_hash)
            completion = json.loads((output/'completion.json').read_text())
            self.assertEqual(completion['rows'], 22)
            self.assertFalse(completion['quality_selection_changed'])
            for relative, expected in completion['output_hashes'].items():
                self.assertEqual(sha256(output/relative), expected)


if __name__ == '__main__':
    unittest.main()
