"""CPU tests for registered budget boundaries and calibration-only selection."""
import json
import math
import unittest
from pathlib import Path
from training_rules import extension_decision, select_checkpoint, N, SEED, TRAIN_SNRS, MAX_UPDATES


def history(step, values):
    return [dict(step=step-5000+i*2500, utility=value) for i, value in enumerate(values)]


class TrainingRulesTest(unittest.TestCase):
    def test_first_boundary_extends_only_to_30000(self):
        decision = extension_decision(history(20000, [1., .99, .98]), 20000)
        self.assertTrue(decision['extend'])
        self.assertEqual(decision['next_limit'], 30000)

    def test_second_boundary_extends_only_to_40000(self):
        decision = extension_decision(history(30000, [1., .99, .98]), 30000)
        self.assertTrue(decision['extend'])
        self.assertEqual(decision['next_limit'], 40000)

    def test_both_intervals_required(self):
        for values in ([1., .99, .991], [1., .999, .99]):
            with self.subTest(values=values):
                decision = extension_decision(history(20000, values), 20000)
                self.assertFalse(decision['extend'])
                self.assertEqual(decision['next_limit'], 20000)
                self.assertEqual(decision['reason'], 'CALIBRATION_GATE_STOP')

    def test_threshold_is_registered_relative_change(self):
        values = [1000., 998., 996.004]
        decision = extension_decision(history(20000, values), 20000)
        self.assertEqual(decision['extend'], all(gain >= .002 for gain in decision['relative_improvements']))
        for gain in decision['relative_improvements']:
            self.assertTrue(math.isclose(gain, .002, abs_tol=1e-14))
        # An absolute .0005 reduction exceeds the relative .002 gate at this scale.
        self.assertTrue(extension_decision(history(20000, [.1, .0995, .099]), 20000)['extend'])
        self.assertFalse(extension_decision(history(20000, [1000., 998.0001, 990.]), 20000)['extend'])

    def test_uses_adjacent_actual_utilities_not_running_best(self):
        candidate_history = [dict(step=0, utility=.5), *history(20000, [1., .99, .98])]
        # An earlier best does not erase two real recent improvements.
        self.assertTrue(extension_decision(candidate_history, 20000)['extend'])

    def test_40000_always_stops_and_distinguishes_truncation(self):
        improving = extension_decision(history(40000, [1., .99, .98]), 40000)
        flat = extension_decision(history(40000, [1., .999, .998]), 40000)
        self.assertFalse(improving['extend'])
        self.assertFalse(flat['extend'])
        self.assertEqual(improving['next_limit'], 40000)
        self.assertTrue(improving['budget_truncated'])
        self.assertFalse(flat['budget_truncated'])
        self.assertFalse(improving['convergence_claimed'])
        self.assertFalse(improving['development_used'])

    def test_no_decision_at_10000_or_other_steps(self):
        for step in (10000, 22500, 50000):
            with self.subTest(step=step), self.assertRaises(ValueError):
                extension_decision(history(step, [1., .99, .98]), step)

    def test_calibration_selection_earliest_exact_tie_and_step0(self):
        self.assertEqual(select_checkpoint([dict(step=2500, utility=.8), dict(step=0, utility=.8)])['step'], 0)
        self.assertEqual(select_checkpoint([dict(step=0, utility=.7), dict(step=2500, utility=.8)])['step'], 0)
        self.assertEqual(select_checkpoint([dict(step=2500, utility=.7), dict(step=0, utility=.8)])['step'], 2500)

    def test_rejects_invalid_selection(self):
        for records in ([], [dict(step=0, utility=float('nan'))], [dict(step=0, utility=float('inf'))],
                        [dict(step=0, utility=1), dict(step=0, utility=2)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                select_checkpoint(records)

    def test_rejects_duplicate_missing_and_invalid_boundary_evidence(self):
        for records in ([*history(20000, [1., .99, .98]), dict(step=0, utility=1), dict(step=0, utility=2)],
                        history(20000, [1., .99, .98])[:2], history(20000, [1., .99, float('nan')]),
                        history(20000, [0., .99, .98])):
            with self.subTest(records=records), self.assertRaises((ValueError, KeyError)):
                extension_decision(records, 20000)

    def test_protocol_constants_agree(self):
        record = json.loads((Path(__file__).parent/'training_protocol.json').read_text())
        self.assertEqual(record['N'], N)
        self.assertEqual(record['training_seed'], SEED)
        self.assertEqual(record['maximum_updates'], MAX_UPDATES)
        self.assertEqual(record['calibration_snrs_db'], TRAIN_SNRS)
        self.assertEqual(record['rows_per_full_calibration'], 1000*len(TRAIN_SNRS)*3)


if __name__ == '__main__':
    unittest.main()
