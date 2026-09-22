import unittest

from latent_enhancement_eval.result_validator import (
    aggregate_source, aggregate_source_snr, paired_source_differences, validate_rows,
)


def row(method, source=0, image="img0", snr=1, seed=2001, value=1.0, decoder="dec-a"):
    return {
        "method": method, "source_index": source, "image_id": image,
        "snr_db": snr, "noise_seed": seed,
        "population_role": "target_development", "family": "raw", "N": 3060,
        "renderer": "D0", "model_sha256": "model-a", "decoder_sha256": decoder,
        "protocol_id": "raw:N3060:m8",
        "noise_model": "seeded_noise_v1:variance=1/SNR", "budget_scope": "N3060:E6120",
        "psnr_db": value, "lpips_alex": value / 10, "dino_cosine": value / 2,
    }


class ResultValidatorTests(unittest.TestCase):
    def test_exact_grid_and_two_stage_pairing(self):
        rows = [row(method, seed=seed, value=(1 if method == "ref" else 2) + seed / 10000)
                for method in ("ref", "arm") for seed in (2001, 2002)]
        checked = validate_rows(rows, expected_methods=("ref", "arm"), expected_sources=(0,),
                                expected_snrs=(1,), expected_seeds=(2001, 2002))
        source_snr = aggregate_source_snr(checked)
        self.assertEqual(len(source_snr), 2)
        source = aggregate_source(source_snr)
        self.assertEqual(len(source), 2)
        delta = paired_source_differences(checked, "ref")
        self.assertAlmostEqual(float(delta["arm"][0, 0]), 1.0, places=6)

    def test_fractional_snr_is_rejected(self):
        bad = row("ref", snr=1.9)
        with self.assertRaises(ValueError):
            validate_rows([bad], expected_snrs=(1,))

    def test_duplicate_and_scope_mismatch_are_rejected(self):
        valid = row("ref")
        with self.assertRaises(ValueError):
            validate_rows([valid, dict(valid)], expected_methods=("ref",), expected_sources=(0,),
                          expected_snrs=(1,), expected_seeds=(2001,))
        changed = dict(valid, protocol_id="raw:N4084:m8")
        with self.assertRaises(ValueError):
            validate_rows([valid, changed])
        changed_key = dict(valid, noise_seed=2002, protocol_id="raw:N4084:m8")
        with self.assertRaises(ValueError):
            validate_rows([valid, changed_key])


if __name__ == "__main__":
    unittest.main()
