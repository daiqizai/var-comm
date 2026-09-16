"""Exercise the real R2 timing driver, including all three waveform origins and source-boundary resume."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from test_quality_contracts import fixture
from pipeline_fixtures import CPUTorch, ModelFixture, NativeFixture, driver
from joint_sender.evaluation_io import digest, write_rows
from sufficiency.evaluation import learned_names
from sufficiency import timing
from wetok_comm.common import sha256
from wetok_comm.evaluation import raw_noise
from wetok_comm.training import module_sha256

TIMER = driver('evaluate_timing')


class TimingPipelineTests(unittest.TestCase):
    def test_saved_quality_is_replayed_not_reselected_and_committed_sources_are_preserved(self):
        torch.set_num_threads(2)
        config, original, grid, reference, base, unused = fixture()
        names = learned_names(config, original, grid, reference)
        evaluation = {'required_total_updates': 10000, 'timing_source_indices': [0, 1], 'timing_snrs_db': [1.], 'timing_noise_seed': 2001}
        base['evaluation']['primary_snrs_db'] = [1.]
        images = torch.stack([torch.full((3, 256, 256), value) for value in (.1, .2)])
        codes = np.full((2, 1, 16, 16, 4), 255, dtype=np.uint8)
        identifiers = ['synthetic_source_0', 'synthetic_source_1']
        native = NativeFixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            quality, output = root / 'quality', root / 'timing'
            roots = {'r2': quality, 'grid': root / 'old_grid', 'joint': root / 'old_joint'}
            rows = []
            for index, identifier in enumerate(identifiers):
                waves = {owner: {} for owner in roots}
                generated = []
                directory = quality / f'images/{index:03d}'
                for archive_root in roots.values():
                    (archive_root / f'images/{index:03d}').mkdir(parents=True)
                for name in names:
                    signal = np.ones((1, 3060, 2), dtype=np.float32)
                    received = signal + raw_noise(identifier, 2001).astype(np.float32)[None] * np.float32(10 ** (-1 / 20))
                    model = ModelFixture(name.split('__', 1)[1]).eval()
                    image = native.decode(model.receive(torch.tensor(received), torch.tensor([1.]))['receiver_features'])[0].detach().numpy()
                    archive = waves[timing.archive_origin(name)]
                    archive[timing.waveform_key(name, 1.)] = signal
                    archive[timing.waveform_key(name, 1., 2001)] = received
                    rows.append({'image_index': index, 'image_id': identifier, 'snr_db': 1., 'seed': 2001, 'arm': name,
                        'checkpoint_sha256': 'selected_checkpoint', 'encoder_sha256': module_sha256(model.encoder),
                        'transmitted_sha256': digest(signal), 'received_sha256': digest(received),
                        'noise_sha256': digest(raw_noise(identifier, 2001)), 'image_archive': str(directory / 'reconstructions.npz'),
                        'image_ref': len(generated), 'image_sha256': digest(image)})
                    generated.append(image)
                for owner, values in waves.items():
                    np.savez(roots[owner] / f'images/{index:03d}/waveforms.npz', **values)
                np.savez(directory / 'reconstructions.npz', images=np.stack(generated))
            write_rows(quality / 'per_frame.csv', rows)
            milestone_path, review_path = root / 'milestone.json', root / 'review.json'
            milestone_path.write_text('{}')
            review_path.write_text('{}')
            (quality / 'completion.json').write_text(json.dumps({'status': 'R2_QUALITY_COMPLETE',
                'r2_milestone_sha256': sha256(milestone_path), 'r2_review_sha256': sha256(review_path),
                'reference_quality_sha256': 'sealed_grid_receipt', 'source_hashes': {},
                'output_hashes': {'per_frame.csv': sha256(quality / 'per_frame.csv')}}))
            quality_hash = sha256(quality / 'per_frame.csv')

            class ReferencesFixture:
                receipt_sha = 'sealed_grid_receipt'
                frozen_models = {'native': module_sha256(nn.Identity())}

                def __init__(self, *args):
                    self.root = roots['grid']
                    self.previous = SimpleNamespace(root=roots['joint'])

                def validate_source(self, index, identifier, source):
                    if identifier != identifiers[index] or not torch.equal(source, images[index:index + 1]):
                        raise RuntimeError('timing source changed')

            def registry(*args):
                models = {name: ModelFixture(name.split('__', 1)[1]).eval() for name in names}
                return nn.Identity(), models, {name: {'checkpoint_sha256': 'selected_checkpoint'} for name in names}

            def admit():
                if (output / 'images/000/receipt.json').exists():
                    raise RuntimeError('other GPU compute workloads are present: [999]; not stopping them')

            with ExitStack() as stack:
                patches = {'torch': CPUTorch(), 'configure_torch': lambda: None,
                    'load_evaluation': lambda: (evaluation, config, original, grid, reference, base, {}),
                    'validate_config': lambda cfg: None, 'require_uncontended_gpu': admit,
                    'audited_endpoint': lambda *args: ({}, milestone_path, review_path, {}),
                    'References': ReferencesFixture, 'evaluation_output': lambda cfg, kind: quality if kind == 'quality' else output,
                    'all_models': registry, 'FrozenWeTok': lambda *args: NativeFixture(),
                    'read_population': lambda *args: (images, codes, identifiers), 'gpu_telemetry': lambda: {'local_time': 'synthetic'},
                    'receive_for_image': lambda model, received, snrs: model.receive(received, snrs), 'verify_pruned_result': lambda *args: 0.}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(TIMER, name, value))
                stack.enter_context(mock.patch.object(timing, 'validate_config', lambda cfg: None))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(sys, 'argv', ['evaluate_timing.py', '--execute']):
                    with self.assertRaisesRegex(RuntimeError, 'other GPU compute'):
                        TIMER.main()
                first_hash = sha256(output / 'images/000/receipt.json')
                with mock.patch.object(TIMER, 'require_uncontended_gpu', lambda: None), \
                        mock.patch.object(sys, 'argv', ['evaluate_timing.py', '--resume', '--execute']):
                    TIMER.main()
            self.assertEqual(sha256(output / 'images/000/receipt.json'), first_hash)
            self.assertEqual(sha256(quality / 'per_frame.csv'), quality_hash)
            completion = json.loads((output / 'completion.json').read_text())
            self.assertEqual(completion['rows'], 30)
            self.assertFalse(completion['quality_selection_changed'])
            self.assertFalse(completion['time_lower_bound_due_to_unfinished_sessions'])
            for relative, expected in completion['output_hashes'].items():
                self.assertEqual(sha256(output / relative), expected)


if __name__ == '__main__':
    unittest.main()
