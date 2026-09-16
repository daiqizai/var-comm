import copy
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from grid_controls.archive_audit import check_channel, waveform_key
from grid_controls.evaluation import all_names, comparison_pairs, statistics, validate_rows, validate_selection
from grid_controls.hardware import admit_quality
from grid_controls.references import ImageCache, project_reference, source_tensor_digest
from joint_sender.evaluation_io import digest
from wetok_comm.evaluation import raw_noise


def fixture():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    original = yaml.safe_load((JOINT / 'configs/study.yaml').read_text())
    reference = yaml.safe_load((INNOVATION / 'configs/study.yaml').read_text())
    base = yaml.safe_load((BASE / 'configs/study.yaml').read_text())
    base['evaluation']['bootstrap_resamples'] = 20
    rows = []
    for index in range(2):
        identifier = f'synthetic_source_{index}'
        for snr in base['evaluation']['snrs_db']:
            for seed in base['evaluation']['noise_seeds']:
                for name in all_names(config, original, reference):
                    new = name.startswith('grid__')
                    value = .2 + .01 * index + (.005 if new else 0)
                    rows.append({'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed, 'arm': name,
                        'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': digest(raw_noise(identifier, seed)),
                        'psnr_db': 20., 'ssim': .8, 'lpips': value, 'dino': .7, 'LPIPS_excess_from_native': value-.1,
                        'severe_distortion': 0, 'online_TX_seconds': '', 'receiver_seconds': '',
                        'quality_origin': 'new_grid_measurement' if new else 'sealed_joint5000_reference',
                        'available_updates': 5000, 'parent_updates': 7000, 'header_uses': 0, 'data_uses': 3060,
                        'raw_bits': 8192, 'decoder_interface': 'continuous_mean', 'bit_error_rate': .1,
                        'feature_mse': .2, 'feature_abs_mean': .8, 'feature_saturation_fraction': .3,
                        'transmitted_sha256': f'{name}_{index}_{snr}', 'selected_step': 5000,
                        'selected_global_data_step': 12000, 'checkpoint_sha256': 'sealed_checkpoint'})
    return config, original, reference, base, rows


class QualityEvaluationTests(unittest.TestCase):
    def test_complete_strong_matrix_and_image_level_statistics(self):
        config, original, reference, base, rows = fixture()
        lookup, names = validate_rows(rows, config, original, reference, base, 2)
        self.assertEqual(len(names), 18)
        self.assertEqual(len(rows), 756)
        self.assertIn('frozen__full_grid_innovation', names)
        self.assertIn('digital_adaptive', names)
        self.assertEqual(len(comparison_pairs(config, original, reference)), 33)
        summaries, paired = statistics(rows, config, original, reference, base, 2)
        self.assertEqual(len(paired), 1584)
        contrast = next(row for row in paired if row['method']=='joint__multiscale_state_history' and row['control']=='grid__full_grid_state_history')
        self.assertAlmostEqual(next(row for row in paired if row['method']==contrast['method'] and row['control']==contrast['control'] and row['metric']=='lpips')['delta'], -.005)

    def test_missing_rows_resource_changes_and_latency_are_rejected(self):
        config, original, reference, base, rows = fixture()
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            validate_rows(rows[:-1], config, original, reference, base, 2)
        for key, value in [('total_complex_uses', 3061), ('total_energy', 6121), ('receiver_seconds', .03), ('noise_sha256', 'wrong')]:
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.assertRaises(RuntimeError):
                validate_rows(changed, config, original, reference, base, 2)

    def test_transmitter_noise_dependence_and_wrong_selection_are_rejected(self):
        config, original, reference, base, rows = fixture()
        selected = {'selected': {name: {'step': 5000, 'checkpoint_sha256': 'sealed_checkpoint'} for name in config['variants']}}
        validate_selection(rows, selected)
        new_row = next(row for row in rows if row['arm'].startswith('grid__'))
        new_row['selected_step'] = 2500
        with self.assertRaisesRegex(RuntimeError, 'selection'):
            validate_selection(rows, selected)
        new_row['selected_step'] = 5000
        new_row['transmitted_sha256'] = 'noise-dependent'
        with self.assertRaisesRegex(RuntimeError, 'noise seed'):
            validate_rows(rows, config, original, reference, base, 2)

    def test_reference_projection_does_not_mutate_quality_or_preserve_current_latency(self):
        original = {'arm': 'joint__single_pass', 'lpips': '.2', 'checkpoint_sha256': 'fixed', 'image_archive': '/original.npz',
            'image_ref': '2', 'image_sha256': 'image', 'online_TX_seconds': '.02', 'receiver_seconds': '.04', 'receiver_order_index': '1', 'latency_source': 'old_measurement'}
        saved = copy.deepcopy(original)
        result = project_reference(original, 'original_receipt')
        self.assertEqual(original, saved)
        self.assertEqual(result['lpips'], '.2')
        self.assertEqual(result['image_archive'], '/original.npz')
        self.assertEqual(result['receiver_seconds'], '')
        self.assertEqual(result['archived_receiver_seconds'], '.04')

    def test_saved_channel_and_image_identity(self):
        config, original, reference, base, rows = fixture()
        name, identifier = 'grid__full_grid_state_history', 'synthetic_source_0'
        signal = np.ones((1, 3060, 2), dtype=np.float32)
        waveforms, channel_rows = {}, []
        for snr in base['evaluation']['snrs_db']:
            waveforms[waveform_key(name, snr)] = signal
            for seed in base['evaluation']['noise_seeds']:
                noise = raw_noise(identifier, seed)
                received = signal + noise.astype(np.float32)[None] * np.float32(10**(-snr/20))
                waveforms[waveform_key(name, snr, seed)] = received
                channel_rows.append({'arm': name, 'image_id': identifier, 'snr_db': snr, 'seed': seed,
                    'noise_sha256': digest(noise), 'transmitted_sha256': digest(signal), 'received_sha256': digest(received)})
        self.assertEqual(check_channel(waveforms, channel_rows, identifier, [name], base), (0., 0.))
        waveforms[waveform_key(name, 1., 2001)] = np.zeros_like(signal)
        with self.assertRaises(RuntimeError):
            check_channel(waveforms, channel_rows, identifier, [name], base)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'images.npz'
            image = np.zeros((3, 256, 256), dtype=np.float32)
            np.savez(path, images=image[None])
            cache = ImageCache()
            np.testing.assert_array_equal(cache.image({'image_archive': str(path), 'image_ref': 0, 'image_sha256': digest(image)}), image)
            with self.assertRaises(RuntimeError):
                cache.image({'image_archive': str(path), 'image_ref': 0, 'image_sha256': 'different'})
        tensor = torch.zeros(1, 3, 256, 256)
        self.assertNotEqual(source_tensor_digest(tensor), source_tensor_digest(tensor[0]))

    def test_shared_quality_has_explicit_admission_and_allocator_limit_not_driver_changes(self):
        settings = {'shared_start_free_GiB': 8., 'pytorch_allocator_limit_GiB': 8.}
        with mock.patch('grid_controls.hardware.process_ids', return_value=[99999999]), \
                mock.patch('grid_controls.hardware.gpu_memory', return_value=(16*1024**3, 24*1024**3)), \
                mock.patch('grid_controls.hardware.torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=24*1024**3)), \
                mock.patch('grid_controls.hardware.torch.cuda.memory.set_per_process_memory_fraction') as limiter:
            with self.assertRaisesRegex(RuntimeError, 'explicit'):
                admit_quality(settings, False)
            result = admit_quality(settings, True)
            self.assertTrue(result['not_a_whole_process_or_compute_isolation_guarantee'])
            limiter.assert_called_once_with(1/3, device=0)
        with mock.patch('grid_controls.hardware.process_ids', return_value=[]), \
                mock.patch('grid_controls.hardware.gpu_memory', return_value=(1024**3, 24*1024**3)):
            with self.assertRaisesRegex(RuntimeError, 'insufficient free GPU memory'):
                admit_quality(settings, True)


if __name__ == '__main__':
    unittest.main()
