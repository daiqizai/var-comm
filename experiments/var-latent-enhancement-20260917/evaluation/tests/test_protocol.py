import unittest

import numpy as np

from latent_enhancement_eval.runner import (
    arithmetic_transmit_budget, method_specs, raw_receive_budget, raw_transmit_budget,
)


class EvaluationProtocolTests(unittest.TestCase):
    def source(self):
        return [np.arange(size * size, dtype=np.int64) % 4096 for size in (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)]

    def test_raw_budgets_are_exact(self):
        source = self.source()
        for budget in (3060, 3572, 4084):
            signal, ledger = raw_transmit_budget(source, 17, 8, budget)
            self.assertEqual(signal.shape, (budget, 2))
            self.assertEqual(int(np.square(signal).sum()), 2 * budget)
            self.assertEqual(ledger["data_uses"], budget - 68)

    def test_arithmetic_budget_is_exact(self):
        payload = {"payload": np.array([0, 1, 1, 0], dtype=np.uint8), "length_field": 4,
                   "raw_payload_bits": 100, "attempted_arithmetic_bits": 4, "raw_fallback": False}
        for budget in (3060, 3572, 4084):
            signal, ledger = arithmetic_transmit_budget(payload, 17, 8, budget)
            self.assertEqual(signal.shape, (budget, 2))
            self.assertEqual(int(np.square(signal).sum()), 2 * budget)
            self.assertEqual(ledger["data_uses"], budget - 94)

    def test_method_count_and_no_synthetic_selection(self):
        config = {"methods": {"latent": [{"name": "one"}], "digital": {"families": ["raw", "arithmetic"],
            "budgets": [3060, 3572, 4084], "modes": [7, 8, 9], "renderers": ["D0", "Dc"]}}}
        methods = method_specs(config)
        self.assertEqual(len(methods), 37)
        self.assertEqual(len({method["name"] for method in methods}), len(methods))


if __name__ == "__main__":
    unittest.main()
