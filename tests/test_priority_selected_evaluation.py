import unittest
from tools.priority_selected_evaluation import inventory, require_final, same_process, verify_completion

class PriorityEvaluationTests(unittest.TestCase):
    def test_inventory_excludes_unfinished_third_seed_controls(self):
        entries = inventory()
        self.assertEqual(sum(len(e[1]) for e in entries), 14)
        third = [e for e in entries if e[3] == 2026092504]
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0][1], ['H6-V', 'H6-P'])
        self.assertTrue(any('scoped_N4084' in str(e[0]) and e[3] == 2026092404 for e in entries))

    def test_refuse_recycled_or_changed_process(self):
        expected = dict(pid=12, start_ticks='40', cmdline='python main ', state='S')
        self.assertTrue(same_process(expected, expected))
        for change in [dict(start_ticks='41'), dict(cmdline='other'), dict(state='Z'), dict(pid=13)]:
            self.assertFalse(same_process({**expected, **change}, expected))
        self.assertFalse(same_process(None, expected))

    def test_only_final_calibration_boundaries(self):
        done = {'state': {'step': 30000, 'updates': {'H6-V': 30000, 'H6-P': 30000}}}
        decision = dict(step=30000, extend=False, development_used=False)
        require_final(done, decision, ['H6-V', 'H6-P'])
        for change in [dict(extend=True), dict(extend=0), dict(step=20000), dict(development_used=True)]:
            with self.assertRaises(RuntimeError):
                require_final(done, {**decision, **change}, ['H6-V', 'H6-P'])
        with self.assertRaises(RuntimeError):
            require_final({'state': {'step': 30000, 'updates': {'H6-V': 29999}}}, decision, ['H6-V'])

    def test_no_resume_after_partial_or_synthetic_grid(self):
        reg = dict(expected_frame_rows=21000, expected_timed_calls=1400)
        done = dict(status='REAL_C_SELECTED_GRID_COMPLETE', frame_rows=21000,
                    timed_calls=1400, synthetic=False, new_holdout=False)
        verify_completion(done, reg)
        for change in [dict(frame_rows=20999), dict(timed_calls=1399), dict(synthetic=True),
                       dict(new_holdout=True), dict(status='RUNNING')]:
            with self.assertRaises(RuntimeError):
                verify_completion({**done, **change}, reg)
