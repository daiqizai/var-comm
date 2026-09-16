import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/position_completed_results.py"
SPEC = importlib.util.spec_from_file_location("completed_positioning", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PositioningTests(unittest.TestCase):
    def test_no_self_dominance(self):
        point = {"complex_uses": 3060, "lpips": .2, "psnr_db": 24., "RX_mean_ms": 3.}
        self.assertFalse(MODULE.dominate(point, point))

    def test_more_resource_does_not_dominate_lower_resource(self):
        lower = {"complex_uses": 3060, "lpips": .2, "psnr_db": 24., "RX_mean_ms": 3.}
        higher = {"complex_uses": 4498, "lpips": .1, "psnr_db": 30., "RX_mean_ms": 2.}
        self.assertFalse(MODULE.dominate(higher, lower))
        self.assertFalse(MODULE.dominate(lower, higher))

    def test_same_resource_quality_and_cost_improvement(self):
        control = {"complex_uses": 3060, "lpips": .2, "psnr_db": 20., "RX_mean_ms": 60., "dino": 1.}
        candidate = {"complex_uses": 3060, "lpips": .1, "psnr_db": 24., "RX_mean_ms": 39., "dino": 0.}
        self.assertTrue(MODULE.dominate(candidate, control))
        self.assertFalse(MODULE.dominate(control, candidate))


if __name__ == "__main__":
    unittest.main()
