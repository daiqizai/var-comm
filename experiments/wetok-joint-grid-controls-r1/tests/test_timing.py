import copy
from pathlib import Path
import sys
import unittest

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT.parent / 'wetok-joint-sender-r1/src'),
               str(EXPERIMENT.parent / 'wetok-innovation-r1/src'), str(EXPERIMENT.parent / 'wetok-comm-v2-20260912/src')]

from grid_controls.evaluation import learned_names
from grid_controls.timing import measurement_order, summarize, validate_config, validate_rows
from test_quality_evaluation import fixture


class TimingTests(unittest.TestCase):
    def test_fixed_source_subset_and_balanced_eleven_model_order(self):
        config, original, reference, base, unused = fixture()
        evaluation = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/evaluation.yaml').read_text())
        validate_config(evaluation)
        names = learned_names(config, original, reference)
        counts = {name: [0] * 11 for name in names}
        for frame in range(160):
            order = measurement_order(names, frame)
            self.assertEqual(set(order), set(names))
            for position, name in enumerate(order):
                counts[name][position] += 1
        self.assertTrue(all(max(values)-min(values)<=1 for values in counts.values()))
        changed = copy.deepcopy(evaluation)
        changed['timing_source_indices'][1] = 2
        with self.assertRaises(RuntimeError):
            validate_config(changed)
        with self.assertRaises(ValueError):
            measurement_order(names[:-1], 0)

    def test_matched_timing_statistics_reject_missing_or_invalid_rows(self):
        config, original, reference, base, unused = fixture()
        evaluation = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/evaluation.yaml').read_text())
        names = learned_names(config, original, reference)
        rows = []
        for position, index in enumerate(evaluation['timing_source_indices']):
            for snr_position, snr in enumerate(evaluation['timing_snrs_db']):
                order = measurement_order(names, position*5 + snr_position)
                for name in names:
                    rows.append({'image_index': index, 'snr_db': snr, 'seed': 2001, 'arm': name,
                        'total_complex_uses': 3060, 'total_energy': 6120., 'receiver_order_index': order.index(name),
                        'receiver_seconds': .03 + .001 * names.index(name) + .00001 * index, 'online_TX_seconds': .01,
                        'timing_scope': 'separate_matched_uncontended_RX_including_visual_Decoder'})
        self.assertEqual(len(rows), 1760)
        validate_rows(rows, evaluation, names)
        summary, paired = summarize(rows, evaluation, names, base)
        self.assertEqual((len(summary), len(paired)), (77, 21))
        first = paired[0]
        self.assertEqual(first['source_images'], 32)
        self.assertAlmostEqual(first['delta'], .001 * (names.index(first['method'])-names.index(first['control'])))
        with self.assertRaises(RuntimeError):
            validate_rows(rows[:-1], evaluation, names)
        rows[0]['receiver_seconds'] = -1
        with self.assertRaises(RuntimeError):
            validate_rows(rows, evaluation, names)


if __name__ == '__main__':
    unittest.main()
