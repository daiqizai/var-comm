import unittest
from tools.restore_priority_reference_waiters import eligible, WORKERS

class PriorityWaiterRecoveryTests(unittest.TestCase):
    def test_only_two_original_waiters(self):
        self.assertEqual([x[0] for x in WORKERS],
                         ['author_native_timing_v1', 'n3060_native_timing_v1'])

    def test_reject_unrelated_or_older_failure(self):
        for _, _, _, state, error, _ in WORKERS:
            current = dict(status=state, error=error, time=101)
            self.assertTrue(eligible(current, state, error, 100))
            for change in [dict(time=99), dict(error='CUDA error'), dict(status='RUNNING')]:
                self.assertFalse(eligible({**current, **change}, state, error, 100))
