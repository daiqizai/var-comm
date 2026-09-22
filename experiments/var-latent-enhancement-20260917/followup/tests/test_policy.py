import unittest
import importlib.util
from pathlib import Path

from var_comm.mode_policies import choose_modes


class PolicyTests(unittest.TestCase):
    def candidates(self):
        return [{"mode": mode, "source_packet_BLER": .05 if mode == 7 else (.08 if mode == 8 else .3),
                 "accepted_correct_probability": .95 if mode == 7 else (.92 if mode == 8 else .7),
                 "source_index_goodput": 1000 * mode, "psnr_db": 20 + mode * .1, "lpips": .2 - mode * .01}
                for mode in (7, 8, 9)]

    def test_registered_rules_are_deterministic(self):
        selected = choose_modes(self.candidates(), .10, .25, 1e-12)
        self.assertEqual(set(selected), {"reliability", "quality", "goodput"})
        self.assertEqual(selected["reliability"], 8)
        self.assertEqual(selected["quality"], 9)
        self.assertEqual(selected["goodput"], 9)

    def test_source_snr_rejects_duplicate_complete_pairing_key(self):
        source = Path(__file__).resolve().parents[1] / "src" / "latent_followup" / "policy_development.py"
        spec = importlib.util.spec_from_file_location("policy_development_under_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = {"source_index": "0", "snr_db": "1", "seed": "2001", "preprocessing_id": "p",
               "psnr_db": "20", "lpips_alex": ".2", "dino_cosine": ".8"}
        with self.assertRaisesRegex(RuntimeError, "duplicate complete pairing key"):
            module._aggregate_source_snr([row, dict(row)])


if __name__ == "__main__":
    unittest.main()
