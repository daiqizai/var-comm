"""Boundary checks for calibration-only extension and checkpoint selection."""
import unittest
from protocol import extension_decision, select_checkpoint


class ProtocolTests(unittest.TestCase):
    def history(self, values, step=10000):
        return [dict(step=step-5000+i*2500, utility=v) for i,v in enumerate(values)]

    def test_both_intervals_required(self):
        self.assertTrue(extension_decision(self.history([1,.997,.994]),10000)['extend'])
        self.assertFalse(extension_decision(self.history([1,.997,.997]),10000)['extend'])
        self.assertFalse(extension_decision(self.history([1,1.001,.990]),10000)['extend'])

    def test_cap_even_if_improving(self):
        decision=extension_decision(self.history([1,.9,.8],30000),30000)
        self.assertFalse(decision['extend'])
        self.assertEqual(decision['next_limit'],30000)

    def test_current_not_best_utility(self):
        rows=[dict(step=0,utility=.1),*self.history([1,.997,.994])]
        self.assertTrue(extension_decision(rows,10000)['extend'])
        self.assertEqual(select_checkpoint(rows)['step'],0)

    def test_selection_ties_choose_earlier(self):
        self.assertEqual(select_checkpoint([dict(step=2500,utility=.1),dict(step=0,utility=.1)])['step'],0)

    def test_incomplete_or_invalid_evidence_fails(self):
        with self.assertRaises(KeyError):extension_decision(self.history([1,.9]),10000)
        with self.assertRaises(ValueError):extension_decision(self.history([1,.9,0]),10000)
        with self.assertRaises(ValueError):extension_decision(self.history([1,.9,.8]),12500)
        with self.assertRaises(ValueError):extension_decision(self.history([1,.9,.8])+[dict(step=10000,utility=.8)],10000)


if __name__=='__main__':unittest.main()
