import ast
import importlib.util
from pathlib import Path
import sys
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))
from analyze_hifi_stage20 import quality_tables, source_matrix
from evaluate_hifi_stage20 import OneRegisteredNoise


class Stage20Tests(unittest.TestCase):
    def setUp(self):
        self.indices = list(range(20))
        self.sources = {index: {"image_id": str(index), "source_pixels_sha256": str(index)} for index in self.indices}
        self.rows = [{"image_index": index, "image_id": str(index), "source_pixels_sha256": str(index),
                      "snr_db": snr, "seed": 2001, "method": "example", "protocol": "common_paid_information",
                      "complex_uses": 4204, "total_energy": 8408, "psnr_db": 20., "ssim": .5, "lpips": .2, "dino": .8}
                     for index in self.indices for snr in (1., 4., 7., 13., 19.)]

    def test_exact20_by5_by1_scope_and_source_bootstrap(self):
        groups, summary, sources, draws = quality_tables(self.rows, self.indices, self.sources)
        self.assertEqual(len(groups), 1)
        self.assertEqual(draws.shape, (10000, 20))
        self.assertEqual(len(summary), 7)
        self.assertTrue(all(row["sources"] == 20 for row in summary))
        self.assertEqual(len(sources), 140)
        self.assertEqual(source_matrix(self.rows, (1., 4., 7.), self.indices).shape, (20, 4))

    def test_failure_or_source_cannot_be_dropped(self):
        with self.assertRaises(RuntimeError):
            quality_tables(self.rows[:-1], self.indices, self.sources)

    def test_second_noise_cannot_silently_enter(self):
        self.rows[0]["seed"] = 2002
        with self.assertRaises(RuntimeError):
            quality_tables(self.rows, self.indices, self.sources)

    def test_scope_adapter_changes_only_registered_noise_assignment(self):
        module = ast.parse((EXPERIMENT / "scripts/evaluate_authors.py").read_text())
        before = [ast.dump(node) for node in ast.walk(module) if isinstance(node, ast.Call)]
        transformer = OneRegisteredNoise()
        after = transformer.visit(module)
        self.assertEqual(transformer.replacements, 1)
        self.assertEqual(before, [ast.dump(node) for node in ast.walk(after) if isinstance(node, ast.Call)])
        assignment = next(node for node in ast.walk(after) if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "seeds")
        self.assertEqual(ast.literal_eval(assignment.value), [2001])


if __name__ == "__main__":
    unittest.main()
