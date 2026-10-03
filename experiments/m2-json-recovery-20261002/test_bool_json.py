"""CPU-only scope and exact Boolean serialization compatibility tests."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('m2_bool_json_tested', HERE / 'bool_json.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.original = json.JSONEncoder.default; self.addCleanup(setattr, json.JSONEncoder, 'default', self.original)
        self.records = []; self.counts, old = b.install_default(np, lambda s, n: self.records.append((s, n)))
        self.assertIs(old, self.original)

    def test_only_exact_numpy_boolean_values_are_converted_and_counted(self):
        obj = dict(selected=dict(selection_feasible=np.bool_(True), dino_feasible=np.bool_(False)),
                   candidates=[np.bool_(True), np.bool_(False)], native=True, score=1.2345)
        self.assertEqual(json.loads(json.dumps(obj)), dict(selected=dict(selection_feasible=True, dino_feasible=False),
            candidates=[True, False], native=True, score=1.2345))
        self.assertEqual(self.counts, dict(numpy_bool_conversions=4, converted_true=2, converted_false=2))
        self.assertEqual(len(self.records), 1)
        self.assertEqual(self.records[0][1]['numpy_bool_conversions'], 1)
        self.assertIsInstance(obj['selected']['selection_feasible'], np.bool_)  # No in-memory rewrite.

    def test_native_bool_numbers_and_none_retain_exact_json_bytes(self):
        obj = dict(value=[True, False, 1, .25, None, 'bool'])
        actual = json.dumps(obj, indent=2)
        json.JSONEncoder.default = self.original
        self.assertEqual(actual, json.dumps(obj, indent=2)); self.assertEqual(self.counts['numpy_bool_conversions'], 0)

    def test_unsupported_numpy_types_and_arbitrary_objects_remain_errors(self):
        for value in (np.int64(1), np.float32(.5), np.array([True]), object()):
            with self.subTest(type=type(value)), self.assertRaises(TypeError): json.dumps(value)
        self.assertEqual(self.counts['numpy_bool_conversions'], 0)

    def test_supported_numpy_float64_behavior_is_not_changed(self):
        value = np.float64(.123456789)
        actual = json.dumps(value); json.JSONEncoder.default = self.original
        self.assertEqual(actual, json.dumps(value))

    def test_double_install_refused(self):
        with self.assertRaises(RuntimeError): b.install_default(np)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.out = self.root / 'outputs/M2-JSON-RECOVERY-20261002'
        self.here = self.out / 'runtime'; self.here.mkdir(parents=True)
        self.entry = self.root / 'outputs/METRIC-CONCURRENT-R4-20261002/runtime/scheduled_process.py'
        self.entry.parent.mkdir(parents=True); self.entry.write_text('# frozen R4 scheduled fixture\n')
        (self.entry.parent / 'm2_controller.py').write_text('# frozen R4 controller fixture\n')
        self.admission = self.entry.parent.parent / 'admission.json'
        self.admission.write_text('{}')
        for name in ('bool_json.py', 'sitecustomize.py', 'README.md'):
            (self.here / name).write_text('# ENGINEERING_SYNTHETIC_RUNTIME\n')
        originals = b.source_bindings(self.entry.parent)
        for rel in ('experiments/scale-causal-partial-residual-20261002/common.py',
                    'experiments/scale-causal-partial-residual-20261002/m2_runner.py', 'experiments/metric-speed-20261002/wrapper.py'):
            path = self.root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# frozen original\n')
            originals[str(path)] = b.sha(path)
        self.path = self.out / 'manifest.json'
        self.manifest = dict(status='REGISTERED_M2_BOOLEAN_JSON_RECOVERY', entrypoint=str(self.entry),
            runtime_source_bindings=b.source_bindings(self.here), original_source_bindings=originals,
            training_updates=0, policy_selection_updates=0, inference_functions_changed=False)
        self.save()
        self.argv = [str(self.entry), '--root', str(self.root), '--admission', str(self.admission), '--stage', 'calibration']

    def save(self): self.path.write_text(json.dumps(self.manifest))
    def validate(self, argv=None):
        return b.validate_manifest(self.path, self.argv if argv is None else argv, self.here,
            root=self.root, out=self.out, entry=self.entry, admission=self.admission)

    def test_only_exact_original_entry_and_four_stage_arguments_eligible(self):
        for stage in b.STAGES:
            args = list(self.argv); args[-1] = stage; self.validate(args)
        for path in (self.entry.parent / 'm2_controller.py', self.entry.with_name('concurrent_runner.py')):
            self.assertFalse(b.eligible([str(path)], self.entry))
        for change in (['extra'], ['--stage', 'all']):
            with self.assertRaises(RuntimeError): self.validate(self.argv + change)

    def test_no_torch_import_in_serializer_module(self):
        self.assertNotIn('import torch', (HERE / 'bool_json.py').read_text())

    def test_missing_original_or_r4_member_rejected(self):
        del self.manifest['original_source_bindings'][str(self.entry)]; self.save()
        with self.assertRaises(RuntimeError): self.validate()

    def test_mutated_source_and_late_runtime_source_refused(self):
        self.validate(); (self.here / 'late.py').write_text('# late source\n')
        with self.assertRaises(RuntimeError): self.validate()
        (self.here / 'late.py').unlink(); self.entry.write_text('# frozen source modified\n')
        with self.assertRaises(RuntimeError): self.validate()

    def test_manifest_scope_and_path_refused(self):
        self.manifest['inference_functions_changed'] = True; self.save()
        with self.assertRaises(RuntimeError): self.validate()
        with self.assertRaises(RuntimeError): b.validate_manifest(self.path.with_name('other.json'), self.argv, self.here,
            root=self.root, out=self.out, entry=self.entry, admission=self.admission)

    def test_per_process_execution_receipt_and_first_conversion_count(self):
        original = json.JSONEncoder.default
        try:
            with mock.patch.object(b, 'process_identity', return_value=dict(pid=1234, start_ticks='456')), \
                    mock.patch.object(b.atexit, 'register') as register:
                path = b.install(self.path, self.argv, self.here, root=self.root, out=self.out, entry=self.entry, admission=self.admission)
                self.assertEqual(b.read(path)['status'], 'REGISTERED')
                json.dumps(dict(feasible=np.bool_(False)))
                record = b.read(path); self.assertEqual(record['numpy_bool_conversions'], 1)
                self.assertEqual(record['stage'], 'calibration'); self.assertFalse(record['scientific_completion_claimed'])
                b.verify(record['inputs']); register.call_args[0][0]()
                self.assertEqual(b.read(path)['status'], 'PROCESS_EXIT_RECORDED')
        finally: json.JSONEncoder.default = original


class StartupTests(unittest.TestCase):
    def test_controller_and_unrelated_entries_are_inert_without_manifest(self):
        with mock.patch.object(sys, 'argv', ['unrelated.py']), mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(os, '_exit') as exit:
            exec(compile((HERE / 'sitecustomize.py').read_text(), str(HERE / 'sitecustomize.py'), 'exec'), {'__file__': str(HERE / 'sitecustomize.py')})
            exit.assert_not_called()

    def test_eligible_missing_manifest_fails_closed_with_exit78(self):
        with mock.patch.object(sys, 'argv', [str(b.ENTRY)]), mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(os, '_exit', side_effect=SystemExit(78)):
            with self.assertRaises(SystemExit) as caught:
                exec(compile((HERE / 'sitecustomize.py').read_text(), str(HERE / 'sitecustomize.py'), 'exec'), {'__file__': str(HERE / 'sitecustomize.py')})
        self.assertEqual(caught.exception.code, 78)


if __name__ == '__main__': unittest.main()
