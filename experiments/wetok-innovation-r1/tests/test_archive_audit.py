from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src')]

from innovation_comm.archive_audit import check_channel, check_features, check_image, digest
from wetok_comm.evaluation import raw_noise


class ArchiveAuditTests(unittest.TestCase):
    def test_real_saved_awgn_not_just_matching_row_hashes(self):
        base = {'evaluation': {'snrs_db': [1., 7.], 'noise_seeds': [1, 2]}}
        source_id, signals, rows = 'synthetic_archive_source', {}, []
        for snr in base['evaluation']['snrs_db']:
            signal = np.ones((1, 3060, 2), dtype=np.float32)
            signals[f'shared__snr{snr}'] = signal
            for seed in base['evaluation']['noise_seeds']:
                noise = raw_noise(source_id, seed)
                received = signal + noise.astype(np.float32)[None] * np.float32(10 ** (-snr / 20))
                signals[f'received__snr{snr}__seed{seed}'] = received
                for index in range(6):
                    rows.append({'arm': f'receiver__{index}', 'image_id': source_id, 'snr_db': snr, 'seed': seed,
                        'noise_sha256': digest(noise), 'shared_transmitted_sha256': digest(signal), 'shared_received_sha256': digest(received)})
        self.assertEqual(check_channel(signals, rows, source_id, base), (0., 0.))
        signals['received__snr1.0__seed1'] = signals['received__snr1.0__seed1'] + np.float32(.01)
        changed = [{**row, 'shared_received_sha256': digest(signals['received__snr1.0__seed1'])}
                   if row['snr_db'] == 1. and row['seed'] == 1 else row for row in rows]
        with self.assertRaisesRegex(RuntimeError, 'AWGN'):
            check_channel(signals, changed, source_id, base)

    def test_image_pointer_and_feature_diagnostics_recomputed(self):
        source = np.zeros((3, 256, 256), dtype=np.float32)
        image = np.full_like(source, .5)
        row = {'image_sha256': digest(image), 'psnr_db': float(-10 * np.log10(.25))}
        self.assertEqual(check_image(image, source, row), 0.)
        with self.assertRaisesRegex(RuntimeError, 'PSNR'):
            check_image(image, source, {**row, 'psnr_db': 8.})
        with self.assertRaisesRegex(RuntimeError, 'reconstruction'):
            check_image(image, source, {**row, 'image_sha256': '0' * 64})
        features = np.full((32, 16, 16), .5, dtype=np.float32)
        truth = np.ones_like(features)
        check_features(features, truth, {'feature_mse': .25, 'bit_error_rate': 0.})
        with self.assertRaisesRegex(RuntimeError, 'diagnostic'):
            check_features(features, truth, {'feature_mse': .25, 'bit_error_rate': .1})


if __name__ == '__main__':
    unittest.main()
