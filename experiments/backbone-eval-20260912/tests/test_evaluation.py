"""CPU-only protocol/accounting tests. No pretrained model construction."""
import importlib.util
import pathlib
import sys
import unittest
import tempfile
import json
import hashlib
import subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_backbones as evaluate
import analyze_backbones as analyze


class EvaluationProtocolTests(unittest.TestCase):
    def test_natural_bits_and_branch_count(self):
        schedule = [1, 1, 2, 3, 3, 4, 5, 6, 8, 11]
        for prefix, bits in ((8, 2424), (9, 3960), (10, 6864)):
            self.assertEqual(sum(p*p for p in schedule[:prefix]) * 2 * 12, bits)
            self.assertEqual(evaluate.expected_raw_bits('xq', prefix), bits)
        self.assertEqual(evaluate.expected_raw_bits('old-official', 8), 3060)
        self.assertEqual(evaluate.expected_raw_bits('old-official', 9), 5088)
        self.assertEqual(evaluate.expected_raw_bits('old-fidelity', 10), 8160)
        self.assertEqual(evaluate.expected_raw_bits('wetok', None), 8192)
        with self.assertRaises(RuntimeError):
            evaluate.check_token_metadata({'raw_bits': 3432, 'index_count': 286, 'shapes': [[1, 286]]}, 'xq', 10)

    def test_receiver_has_no_suffix(self):
        original = [list(range(10)), list(range(10, 20))]
        receiver = evaluate.truncate_tokens(original, 8, 'xq')
        self.assertEqual(receiver, [list(range(8)), list(range(10, 18))])
        original[0][9] = 100
        self.assertEqual(receiver[0], list(range(8)))
        self.assertEqual(evaluate.truncate_tokens(list(range(10)), 9, 'old-official'), list(range(9)))
        with self.assertRaises(ValueError):
            evaluate.truncate_tokens(original, 8, 'wetok')

    def test_direct_decoder_receives_only_prefix(self):
        class Capture:
            def decode(self, tokens, prefix_scales):
                self.received = tokens
                return 'image'
        adapter = Capture()
        case = {'kind': 'direct', 'prefix_scales': 8}
        evaluate.execute_case(adapter, list(range(10)), None, case, 'old-official')
        self.assertEqual(adapter.received, list(range(8)))
        evaluate.execute_case(adapter, [list(range(10)), list(range(10))], None, case, 'xq')
        self.assertEqual([len(branch) for branch in adapter.received], [8, 8])

    def test_old_builder_global_reset_patch_is_scoped(self):
        import torch.nn as nn
        original = nn.Linear.reset_parameters
        with self.assertRaises(RuntimeError):
            with evaluate.preserve_nn_reset_parameters():
                nn.Linear.reset_parameters = lambda self: None
                raise RuntimeError('simulated strict-checkpoint failure')
        self.assertIs(nn.Linear.reset_parameters, original)

    def test_preregistered_conditions(self):
        config = {'xq_sampling': {'enabled': True, 'seeds': [0, 1, 2], 'cfg': 3.25}}
        cases = evaluate.conditions('xq', config)
        self.assertEqual(len(cases), 11)
        self.assertEqual(len([case for case in cases if case['primary']]), 5)
        for case in cases:
            if case['mode'] == 'argmax':
                self.assertEqual(case['cfg'], 1.0)
            if case['mode'] == 'sample':
                self.assertEqual(case['cfg'], 3.25)
                self.assertFalse(case['primary'])
        self.assertEqual(len(evaluate.conditions('old-official', config)), 5)
        self.assertEqual(len(evaluate.conditions('old-fidelity', config)), 1)
        self.assertEqual(len(evaluate.conditions('wetok', config)), 1)

    def test_bootstrap_is_image_paired_and_deterministic(self):
        first = analyze.bootstrap([1, 2, 3, 4], resamples=1000)
        second = analyze.bootstrap([1, 2, 3, 4], resamples=1000)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 2.5)
        self.assertEqual(analyze.bootstrap([1, 1, 1], resamples=100)[1:], (1.0, 1.0))
        with self.assertRaises(ValueError):
            analyze.bootstrap([np.nan])

    def test_seeds_are_averaged_within_image(self):
        groups = {}
        for seed in (0, 1, 2):
            groups[('xq', f'p8_sample_s{seed}')] = [
                {'image_id': str(i), 'seed': seed, 'name': f'p8_sample_s{seed}',
                 'inference_measure': {'seconds': float(seed+1)},
                 **{metric: float(i + seed) for metric in analyze.METRICS}}
                for i in range(2)]
        result = analyze.image_averaged_sampling(groups)[('xq', 'p8_sample_mean')]
        self.assertEqual(len(result), 2)  # not six independent image samples
        self.assertEqual([row['psnr_db'] for row in result], [1.0, 2.0])
        self.assertEqual(result[0]['sampling_seeds'], [0, 1, 2])
        self.assertEqual(result[0]['inference_measure']['seconds'], 2.0)

    def test_analysis_end_to_end_synthetic(self):
        # Temporary synthetic smoke artifacts; never scientific output.
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='synthetic-analysis-test-', dir=ROOT / 'tests') as temp:
            temporary = pathlib.Path(temp)
            runs = temporary / 'smoke'
            config = {'xq_sampling': {'enabled': True, 'seeds': [0, 1, 2], 'cfg': 3.25}}
            for model_number, model in enumerate(analyze.MODELS):
                destination = runs / model
                destination.mkdir(parents=True)
                cases = evaluate.conditions(model, config)
                rows = []
                for image_index in range(2):
                    images = destination / 'images' / f'{image_index:03d}'
                    images.mkdir(parents=True)
                    picture = Image.new('RGB', (256, 256), (image_index * 60, 40, 70))
                    picture.save(images / 'source.png')
                    for case in cases:
                        picture.save(images / (case['name'] + '.png'))
                        row = {**case, 'model': model, 'image_id': str(image_index), 'image_index': image_index,
                            'source_file_sha256': f'source-{image_index}', 'source_tensor_sha256': f'tensor-{image_index}',
                            'class_index': image_index, 'raw_bits': evaluate.expected_raw_bits(model, case['prefix_scales']),
                            'class_side_bits_if_sent': 10 if case['kind'] == 'completion' else 0,
                            'png_relative_path': f"images/{image_index:03d}/{case['name']}.png",
                            'encode_measure': {'seconds': .001, 'peak_allocated_bytes': 200, 'peak_reserved_bytes': 400},
                            'inference_measure': {'seconds': .002, 'peak_allocated_bytes': 300, 'peak_reserved_bytes': 400},
                            'prefix_audit': {'prefix_preserved': True, 'true_suffix_passed': False},
                            **{metric: float(model_number + image_index + 1) / 10 for metric in analyze.METRICS}}
                        rows.append(row)
                content = ''.join(json.dumps(row) + '\n' for row in rows)
                (destination / 'per_image.jsonl').write_text(content)
                (destination / 'metadata.json').write_text(json.dumps({'conditions': cases,
                    'adapter_metadata': {'paired_tokenizer_matches_standalone': False}}))
                (destination / 'completion.json').write_text(json.dumps({'status': 'COMPLETE', 'rows': len(rows),
                    'images': 2, 'smoke': True, 'manifest_sha256': 'synthetic-not-data',
                    'per_image_jsonl_sha256': hashlib.sha256(content.encode()).hexdigest()}))
            output = temporary / 'analysis-smoke'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/analyze_backbones.py'),
                '--results-root', str(runs), '--output-dir', str(output), '--smoke', '--bootstrap-resamples', '100'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            completion = json.loads((output / 'completion.json').read_text())
            self.assertEqual(completion['images'], 2)
            self.assertEqual(completion['bootstrap_unit'], 'image')
            self.assertTrue(completion['complete_four_model_protocol'])
            self.assertIn('CHECKPOINT IDENTITY WARNING', (output / 'report.md').read_text())
            self.assertTrue((output / 'rate_quality_discrete.png').exists())
            self.assertTrue((output / 'xq_same_prefix_difficult_cases.png').exists())

    def test_comparisons_keep_actual_rate_mismatch_explicit(self):
        groups = {('xq', 'p8_argmax'): [], ('old-official', 'p8_argmax'): []}
        pairs = analyze.pair_comparisons(groups)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][2], 'natural_prefix_unequal_raw_bits')


if __name__ == '__main__':
    unittest.main()
