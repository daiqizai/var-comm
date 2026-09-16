import copy
import unittest

import yaml

from test_quality_contracts import EXPERIMENT, fixture
from pipeline_fixtures import driver
from sufficiency.evaluation import learned_names
from sufficiency.timing import archive_origin, measurement_order, summarize, validate_config, validate_rows, waveform_key

FINISHER = driver('finish')


class TimingContractTests(unittest.TestCase):
    def test_all_fifteen_models_rotation_seventy_intervals_and_training_scope(self):
        config, original, grid, reference, base, unused = fixture()
        evaluation = yaml.safe_load((EXPERIMENT / 'configs/evaluation.yaml').read_text())
        names = learned_names(config, original, grid, reference)
        rows = []
        for position, index in enumerate(evaluation['timing_source_indices']):
            for snr_position, snr in enumerate(evaluation['timing_snrs_db']):
                order = measurement_order(names, position * 5 + snr_position)
                for name in names:
                    rows.append({'image_index': index, 'image_id': f'source_{index}', 'snr_db': snr, 'seed': 2001, 'arm': name,
                        'receiver_order_index': order.index(name), 'total_complex_uses': 3060, 'total_energy': 6120.,
                        'online_TX_seconds': .02, 'receiver_seconds': .05 + index * .00001 - .001 * name.startswith('r2__'),
                        'timing_scope': 'separate_matched_uncontended_RX_including_visual_Decoder'})
        summary, paired = summarize(rows, evaluation, names, base)
        self.assertEqual((len(rows), len(summary), len(paired)), (2400, 105, 70))
        self.assertEqual(sum(row['comparison_scope'] == 'matched10000_structure_comparison' for row in paired), 42)
        for name in names:
            self.assertEqual({measurement_order(names, frame).index(name) for frame in range(15)}, set(range(15)))
        with self.assertRaises(RuntimeError):
            validate_rows(rows[:-1], evaluation, names)
        for key, value in (('total_energy', 7000), ('receiver_order_index', -1), ('receiver_seconds', float('nan')),
            ('timing_scope', 'shared_quality_latency'), ('image_id', 'substituted_source')):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.assertRaises(RuntimeError):
                validate_rows(changed, evaluation, names)
        changed = copy.deepcopy(evaluation)
        changed['timing_source_indices'][0] = 1
        with self.assertRaises(RuntimeError):
            validate_config(changed)

    def test_waveforms_are_routed_to_their_actual_source_trial(self):
        self.assertEqual(archive_origin('r2__single_pass'), 'r2')
        self.assertEqual(archive_origin('grid__full_grid_state_history'), 'grid')
        self.assertEqual(archive_origin('joint__single_pass'), 'joint')
        self.assertEqual(archive_origin('frozen__full_grid_innovation'), 'joint')
        self.assertEqual(waveform_key('frozen__single_pass', 7., 2001), 'frozen_shared__snr7.0__seed2001')
        with self.assertRaises(ValueError):
            archive_origin('digital_m8')
        with self.assertRaises(ValueError):
            measurement_order(['duplicate'] * 15, 0)

    def test_only_exact_owned_resource_guard_failures_are_retryable(self):
        cases = [('quality', 'insufficient free GPU memory for the registered quality admission'),
            ('quality', 'insufficient free GPU memory between quality source images'),
            ('timing', 'other GPU compute workloads are present: [123, 456]; not stopping them')]
        for stage, message in cases:
            record = {'pid': 789, 'status': 'R2_' + stage.upper() + '_FAILED_OR_INTERRUPTED'}
            text = 'Traceback\nRuntimeError: ' + message + '\n'
            self.assertTrue(FINISHER.retryable_guard(stage, 1, text, record, 789))
            self.assertFalse(FINISHER.retryable_guard(stage, 1, text, record, 999))
            self.assertFalse(FINISHER.retryable_guard(stage, -9, text, record, 789))
            self.assertFalse(FINISHER.retryable_guard(stage, 1, text + 'RuntimeError: changed checkpoint\n', record, 789))
            self.assertFalse(FINISHER.retryable_guard('analysis', 1, text, record, 789))
        self.assertFalse(FINISHER.retryable_guard('timing', 1, 'CUDA out of memory', {'pid': 789}, 789))


if __name__ == '__main__':
    unittest.main()
