"""Regression against the frozen parity code and actual historical CSV schemas."""
import ast
import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import replay_compat as compat


def locate(repository, snapshot):
    for root in (Path.cwd(), *Path(__file__).resolve().parents):
        for rel in (repository, snapshot):
            candidate = root / rel
            if candidate.is_file(): return candidate
    raise FileNotFoundError('Required frozen regression input: ' + repository)


REPLAY = locate('experiments/unified-metrics-20261002/replay.py', '.research/metrics_20261002/source/replay.py')
spec = importlib.util.spec_from_file_location('_alias_regression_original_replay', REPLAY)
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)


def function(path, name, class_name=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = tree.body if class_name is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def native_latent_fields():
    path = locate('experiments/extreme-bandwidth-20260930/digital.py', '.research/extreme_bw/digital.py')
    node = function(path, 'latent_fields')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['latent_fields']


class TensorFixture:
    def __init__(self, x): self.x = np.asarray(x)
    def double(self): return TensorFixture(self.x.astype(np.float64))
    def __getitem__(self, key): return TensorFixture(self.x[key])
    def __sub__(self, other): return TensorFixture(self.x - other.x)
    def square(self): return TensorFixture(np.square(self.x))
    def sum(self): return self.x.sum()


class AliasRegression(unittest.TestCase):
    def measurements(self, latent=True):
        clean = np.asarray([[1.25, -2.5], [3.75, .5]], dtype=np.float32)
        received = np.asarray([[[.5, -2.], [3., -.5]]], dtype=np.float32)
        fields = native_latent_fields()(TensorFixture(clean), TensorFixture(received) if latent else None)
        fields.update(psnr_db=20., lpips_alex=.2, dino_cosine=.7)
        error = float(np.square(clean.astype(np.float64) - received[0].astype(np.float64)).sum()) if latent else ''
        saved = dict(fields, latent_sq_error=error)
        return saved, fields, error

    def test_actual_native_latent_computation_fails_before_alias_and_passes_both_after(self):
        saved, computed, expected = self.measurements()
        with self.assertRaisesRegex(original.ReplayMismatch, 'latent_sq_error'):
            original.parity_check(saved, computed)
        result = compat.add_computed_alias(computed)
        self.assertEqual(result['latent_sq_error'], expected)
        self.assertNotIn('latent_sq_error', computed)
        checked = original.parity_check(saved, result)
        self.assertIn('latent_sq_error', checked['checked_fields'])
        self.assertIn('latent_sq_err_final', checked['checked_fields'])
        self.assertEqual(checked['metric_deltas']['latent_sq_error'], 0.)
        for field in ('latent_sq_error', 'latent_sq_err_final'):
            with self.assertRaisesRegex(original.ReplayMismatch, field):
                original.parity_check(dict(saved, **{field: expected+1}), result)

    def test_erasure_remains_blank_and_unrelated_missing_fields_still_fail(self):
        saved, computed, _ = self.measurements(False)
        result = compat.add_computed_alias(computed)
        self.assertEqual(result['latent_sq_error'], '')
        self.assertFalse(result['latent_valid'])
        self.assertGreater(result['zero_erasure_proxy_sq_error'], 0)
        original.parity_check(saved, result)
        with self.assertRaisesRegex(original.ReplayMismatch, 'waveform_sha256'):
            original.parity_check(dict(saved, waveform_sha256='a'*64), result)
        with self.assertRaisesRegex(RuntimeError, 'blank'):
            compat.add_computed_alias(dict(computed, latent_sq_err_final=0.))

    def test_adapter_preserves_row_and_rgb_and_rejects_conflicting_alias(self):
        saved, computed, _ = self.measurements()
        pixels = np.zeros((3, 256, 256), np.float32)
        def native(self, index, rows):
            yield saved, pixels, computed
        wrapped = compat.adapt_digital_source(native)
        row, rgb, got = next(wrapped(SimpleNamespace(N=512), 0, [saved]))
        self.assertIs(row, saved); self.assertIs(rgb, pixels)
        original.parity_check(row, got)
        with self.assertRaisesRegex(RuntimeError, 'disagrees'):
            compat.add_computed_alias(dict(computed, latent_sq_error=12345.))
        with self.assertRaisesRegex(RuntimeError, 'N512/N1024'):
            next(wrapped(SimpleNamespace(N=2048), 0, [saved]))

    def test_all_48000_historical_digital_rows_have_only_this_missing_native_alias(self):
        parity_fields = set(original.PARITY_TOLERANCES) | set(original.HASH_FIELDS) | set(original.EXACT_FIELDS)
        adapter = function(REPLAY, '_digital_source', '_OldAdapter')
        total = 0
        for N, folder, snapshot in ((512, 'extreme-bandwidth-20260930', 'extreme_bw'),
                                    (1024, 'extreme-bandwidth-20261001-N1024', 'extreme_bw_n1024')):
            digital = locate(f'experiments/{folder}/digital.py', f'.research/{snapshot}/digital.py')
            phy = locate(f'experiments/{folder}/digital_protocol.py', f'.research/{snapshot}/digital_protocol.py')
            analysis = locate(f'experiments/{folder}/analysis.py', f'.research/{snapshot}/analysis.py')
            self.assertTrue(compat.has_historical_alias(analysis))
            nodes = [adapter, function(digital, 'latent_fields'), function(digital, 'quality_outputs'),
                     function(phy, 'action_record'), function(phy, 'transmit')]
            provided = set()
            for node in nodes:
                for item in ast.walk(node):
                    if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == 'dict':
                        provided.update(k.arg for k in item.keywords if k.arg is not None)
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                                provided.add(target.slice.value)
            event = function(digital, 'event_fields')
            provided.update(ast.literal_eval(next(n.value for n in event.body if isinstance(n, ast.Assign))))
            date, suffix = ('20260930', '') if N == 512 else ('20261001', '_N1024')
            table = locate(f'results/extreme_bandwidth_{date}_R1{suffix}/per_frame.csv',
                           f'.research/{snapshot}_results/per_frame.csv')
            required, count = set(), 0
            with table.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    if not row['method'].startswith('D_'): continue
                    count += 1
                    self.assertEqual(row['latent_sq_error'], row['latent_sq_err_final'])
                    required.update(k for k in parity_fields if row.get(k) not in ('', None))
            self.assertEqual(count, 24000)
            self.assertEqual(required - provided, {'latent_sq_error'})
            self.assertFalse(required - (provided | {'latent_sq_error'}))
            total += count
        self.assertEqual(total, 48000)


if __name__ == '__main__': unittest.main()
