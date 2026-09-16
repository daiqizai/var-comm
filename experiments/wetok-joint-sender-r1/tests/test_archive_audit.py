from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

from joint_sender.archive_audit import check_channel, waveform_key
from joint_sender.evaluation import fresh_names
from joint_sender.evaluation_io import digest
from wetok_comm.evaluation import raw_noise


class JointArchiveTests(unittest.TestCase):
    def test_actual_joint_signal_and_noise_are_checked_not_just_hash_agreement(self):
        config = {'variants': ['single_pass', 'multiscale_no_history', 'multiscale_state_history']}
        reference = {'variants': config['variants'] + ['multiscale_prediction_features', 'multiscale_innovation', 'full_grid_innovation']}
        base = {'evaluation': {'snrs_db': [1., 7.], 'noise_seeds': [2001, 2002]}}
        identifier, waveforms, rows = 'synthetic_joint_source', {}, []
        for snr in base['evaluation']['snrs_db']:
            for seed in base['evaluation']['noise_seeds']:
                noise = raw_noise(identifier, seed)
                for name in fresh_names(config, reference):
                    sign = -1. if name.startswith('joint__') else 1.
                    signal = np.full((1, 3060, 2), sign, dtype=np.float32)
                    observed = signal + noise.astype(np.float32)[None] * np.float32(10 ** (-snr / 20))
                    waveforms[waveform_key(name, snr)] = signal
                    waveforms[waveform_key(name, snr, seed)] = observed
                    rows.append({'arm': name, 'snr_db': snr, 'seed': seed, 'image_id': identifier,
                        'noise_sha256': digest(noise), 'transmitted_sha256': digest(signal), 'received_sha256': digest(observed)})
        self.assertEqual(check_channel(waveforms, rows, identifier, config, reference, base), (0., 0.))
        changed_name = 'joint__single_pass'
        key = waveform_key(changed_name, 1., 2001)
        waveforms[key] = waveforms[key] + np.float32(.01)
        changed = [{**row, 'received_sha256': digest(waveforms[key])} if row['arm'] == changed_name and
                   row['snr_db'] == 1. and row['seed'] == 2001 else row for row in rows]
        with self.assertRaisesRegex(RuntimeError, 'AWGN'):
            check_channel(waveforms, changed, identifier, config, reference, base)

    def test_only_frozen_receivers_share_an_archive_waveform_key(self):
        self.assertEqual(waveform_key('frozen__single_pass', 7., 2001), waveform_key('frozen__full_grid_innovation', 7., 2001))
        self.assertNotEqual(waveform_key('joint__single_pass', 7., 2001), waveform_key('joint__multiscale_state_history', 7., 2001))


if __name__ == '__main__':
    unittest.main()
