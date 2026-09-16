"""Run the new production quality driver through interruption, resume and CPU archive verification."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from test_quality_contracts import fixture
from pipeline_fixtures import CPUTorch, ModelFixture, NativeFixture, driver, fixture_metrics
from joint_sender.evaluation_io import digest
from sufficiency import archive_audit
from sufficiency.evaluation import all_names, learned_names, new_names, validate_diagnostics, validate_rows
from sufficiency.references import project_reference, read_rows
from wetok_comm.common import sha256
from wetok_comm.evaluation import raw_noise
from wetok_comm.training import module_sha256

EVALUATOR = driver('evaluate_quality')


class QualityPipelineTests(unittest.TestCase):
    def test_driver_resume_preserves_all18_references_and_actual_CPU_archive_audit(self):
        torch.set_num_threads(2)
        config, original, grid, reference, base, unused = fixture()
        base['evaluation'].update(snrs_db=[1., 5., 6., 19.], primary_snrs_db=[1.], noise_seeds=[2001])
        evaluation = {'required_total_updates': 10000, 'source_images': 2, 'noiseless_rows': 30, 'support_rows': 4,
            'native_metric_tolerances': {'psnr_db': 1e-4, 'ssim': 1e-5, 'lpips': 1e-5, 'dino': 1e-5}}
        images = torch.stack([torch.full((3, 256, 256), value) for value in (.1, .2)])
        codes = np.full((2, 1, 16, 16, 4), 255, dtype=np.uint8)
        identifiers = ['synthetic_source_0', 'synthetic_source_1']
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_names = all_names(config, original, grid, reference)[4:]
            clean, old_rows, noiseless, support = {}, [], [], []
            for index, identifier in enumerate(identifiers):
                image = torch.full((3, 256, 256), .7)
                score = fixture_metrics(images[index:index + 1], [image])[0]
                archive = root / f'old_image_{index}.npz'
                np.savez(archive, images=image[None].numpy())
                locator = {'image_archive': str(archive), 'image_ref': 0, 'image_sha256': digest(image)}
                common = {'image_index': index, 'image_id': identifier, **score, **locator, 'online_TX_seconds': '', 'receiver_seconds': '',
                    'LPIPS_excess_from_native': 0., 'severe_distortion': 0, 'quality_origin': 'sealed_grid_reference'}
                clean[index] = {**common, 'raw_bits': 8192}
                for snr in base['evaluation']['snrs_db']:
                    for name in old_names:
                        old_rows.append({**common, 'snr_db': snr, 'seed': 2001, 'arm': name,
                            'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': digest(raw_noise(identifier, 2001)),
                            'archived_receiver_seconds': '.03', 'archived_online_TX_seconds': '.01'})
                for name in learned_names(config, original, grid, reference)[4:]:
                    noiseless.append({**common, 'arm': name, 'channel': 'noiseless_nominal19_not_wireless_ranking'})
                for snr in (5., 6.):
                    support.append({**common, 'snr_db': snr, 'seed': 2001, 'arm': 'perceptual_deepjscc_fixed_support'})

            class ReferencesFixture:
                receipt_sha = 'sealed_grid_receipt'
                source_table_sha = 'frozen_source_tensors'
                frozen_models = {key: module_sha256(nn.Identity()) for key in ('native', 'lpips', 'dino')}

                def __init__(self, *args):
                    self.clean, self.noiseless, self.support = clean, noiseless, support

                def source_rows(self, index):
                    return [project_reference(row, self.receipt_sha) for row in old_rows if row['image_index'] == index]

                def validate_source(self, index, identifier, source):
                    if identifier != identifiers[index] or not torch.equal(source, images[index:index + 1]):
                        raise RuntimeError('fixture source changed')

                def validate_reuse(self, rows):
                    lookup = {(row['image_index'], row['snr_db'], row['seed'], row['arm']): row for row in old_rows}
                    for row in rows:
                        if not row['arm'].startswith('r2__'):
                            expected = project_reference(lookup[int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']], self.receipt_sha)
                            if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
                                raise RuntimeError('a sealed reference changed in the actual quality driver')

            milestone_path, review_path = root / 'milestone.json', root / 'review.json'
            milestone_path.write_text('{}')
            review_path.write_text('{}')
            milestone = {'selected': {name: {'step': 10000, 'checkpoint_sha256': 'checkpoint'} for name in config['variants']}}

            def registry(*args):
                models = {name: ModelFixture(name.split('__', 1)[1]).eval() for name in new_names(config)}
                choices = {name: {'step': 10000, 'checkpoint_sha256': 'checkpoint', 'encoder_sha256': module_sha256(model.encoder),
                    'selected_encoder_differs_from_parent': True, 'communication_parameters': 1, 'optimized_parameters': 1} for name, model in models.items()}
                return nn.Identity(), models, choices

            with ExitStack() as stack:
                patches = {'torch': CPUTorch(), 'configure_torch': lambda: None,
                    'load_evaluation': lambda: (evaluation, config, original, grid, reference, base, {}),
                    'audited_endpoint': lambda *args: (milestone, milestone_path, review_path, {}),
                    'References': ReferencesFixture, 'evaluation_output': lambda *args: root / 'quality',
                    'admit_quality': lambda cfg, shared: {'shared': shared}, 'quality_telemetry': lambda cfg: {'compute_pids': [123, 456]},
                    'model_registry': registry, 'metric_models': lambda device: (nn.Identity(), nn.Identity()),
                    'read_population': lambda *args: (images, codes, identifiers), 'validate_population': lambda ids: None,
                    'metrics': fixture_metrics, 'validate_rows': lambda *args: validate_rows(*args, source_count=2),
                    'validate_diagnostics': lambda *args: validate_diagnostics(*args, source_count=2)}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(EVALUATOR, name, value))
                stack.enter_context(mock.patch('builtins.print'))
                with mock.patch.object(EVALUATOR, 'FrozenWeTok', return_value=NativeFixture(True)), \
                        mock.patch.object(sys, 'argv', ['evaluate_quality.py', '--shared-gpu', '--execute']):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        EVALUATOR.main()
                output = root / 'quality'
                first_hash = sha256(output / 'images/000/receipt.json')
                with mock.patch.object(EVALUATOR, 'FrozenWeTok', return_value=NativeFixture()), \
                        mock.patch.object(sys, 'argv', ['evaluate_quality.py', '--shared-gpu', '--resume', '--execute']):
                    EVALUATOR.main()
                rows, anchors, no_noise, supplement = [read_rows(output / filename) for filename in
                    ('per_frame.csv', 'native_reference.csv', 'noiseless_mapping.csv', 'deep_support_supplement.csv')]
                with mock.patch.object(archive_audit, 'read_population', return_value=(images, codes, identifiers)), \
                        mock.patch.object(archive_audit, 'validate_population', lambda ids: None):
                    audit = archive_audit.audit_saved(output, rows, anchors, no_noise, supplement, config, base, ReferencesFixture())
            self.assertEqual(sha256(output / 'images/000/receipt.json'), first_hash)
            completion = json.loads((output / 'completion.json').read_text())
            self.assertEqual((completion['rows'], completion['new_rows'], completion['noiseless_rows']), (176, 32, 30))
            self.assertFalse(completion['time_lower_bound_due_to_unfinished_sessions'])
            self.assertEqual((audit['checked_image_rows'], audit['checked_new_feature_rows']), (212, 40))
            self.assertTrue(all(row['receiver_seconds'] == '' and row['online_TX_seconds'] == '' for row in rows))
            self.assertTrue(all(row['archived_receiver_seconds'] == '.03' for row in rows if not row['arm'].startswith('r2__')))
            for relative, expected in completion['output_hashes'].items():
                self.assertEqual(sha256(output / relative), expected)


if __name__ == '__main__':
    unittest.main()
