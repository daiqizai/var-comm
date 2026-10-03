"""Real file provenance and exact R2 scope; no scientific image result claims."""
import ast
from copy import deepcopy
from importlib.machinery import ModuleSpec
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from history_common import python_source_in_repo
from register_selected_queue import ROSTER, selected_identity, continuation_queue


class PythonSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve(); self.root = self.parent/'repo'
        self.root.mkdir(); self.real = self.root/'real.py'; self.real.write_text('# source\n')

    def test_absolute_repository_source_is_bound_and_external_file_is_skipped(self):
        self.assertEqual(python_source_in_repo(SimpleNamespace(__file__=str(self.real)), self.root), self.real)
        external = self.parent/'outside.py'; external.write_text('# dependency\n')
        self.assertIsNone(python_source_in_repo(SimpleNamespace(__file__=str(external)), self.root))

    def test_virtual_class_file_and_dynamic_getattr_are_never_read(self):
        class Virtual(ModuleType):
            __file__ = '_ops.py'
            def __getattr__(self, name):
                raise AssertionError('Dynamic module metadata must not execute')
        self.assertIsNone(python_source_in_repo(Virtual('fixture'), self.root))

    def test_generated_relative_marker_cannot_become_a_cwd_file(self):
        for marker in ('<_remote_module_non_scriptable>.py', 'real.py', '_ops.py'):
            module = SimpleNamespace(__file__=marker)
            with patch.object(Path, 'resolve', side_effect=AssertionError('Do not resolve a relative marker')):
                self.assertIsNone(python_source_in_repo(module, self.root))

    def test_relative_file_uses_absolute_loader_origin(self):
        module = SimpleNamespace(__file__='virtual.py', __spec__=ModuleSpec('fixture', loader=None,
                                                                          origin=str(self.real)))
        self.assertEqual(python_source_in_repo(module, self.root), self.real)

    def test_missing_or_relative_loader_origin_does_not_use_cwd(self):
        for origin in (None, 'frozen', 'relative.py', '<generated>'):
            module = SimpleNamespace(__file__='marker.py', __spec__=ModuleSpec('fixture', None, origin=origin))
            self.assertIsNone(python_source_in_repo(module, self.root))

    def test_absolute_repository_missing_sources_are_not_silently_skipped(self):
        missing = self.root/'missing.py'
        for module in (SimpleNamespace(__file__=str(missing)),
                       SimpleNamespace(__file__='marker.py', __spec__=SimpleNamespace(origin=str(missing)))):
            with self.assertRaisesRegex(RuntimeError, 'source is missing'):
                python_source_in_repo(module, self.root)

    def test_non_file_and_no_instance_source_are_rejected_or_skipped(self):
        directory = self.root/'directory.py'; directory.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'not a file'):
            python_source_in_repo(SimpleNamespace(__file__=str(directory)), self.root)
        for module in (None, SimpleNamespace(__file__=None), SimpleNamespace(__file__=''),
                       SimpleNamespace(__file__=12), SimpleNamespace(__spec__=SimpleNamespace(origin=str(self.real)))):
            self.assertIsNone(python_source_in_repo(module, self.root))

    def test_non_python_sources_are_skipped(self):
        for suffix in ('.pyc', '.so', '.pyd'):
            self.assertIsNone(python_source_in_repo(SimpleNamespace(__file__=str(self.real.with_suffix(suffix))), self.root))

    def test_all_three_full_module_scans_use_shared_provenance_helper(self):
        here = Path(__file__).parent
        for name in ('historical_budget.py', 'historical_selected_budget.py', 'historical_selected_references.py'):
            tree = ast.parse((here/name).read_text(encoding='utf-8'))
            scans = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                     and ast.unparse(node.iter) == 'tuple(sys.modules.values())']
            self.assertEqual(len(scans), 1, name)
            calls = [node for node in ast.walk(scans[0]) if isinstance(node, ast.Call)]
            self.assertTrue(any(ast.unparse(node.func) == 'python_source_in_repo' for node in calls), name)
            self.assertFalse(any(ast.unparse(node.func) == 'getattr' for node in calls), name)


class R2RegistrationTests(unittest.TestCase):
    def test_legacy_cache_builder_cannot_write_original_admission(self):
        from build_selected_caches import build
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root/'outputs/HISTORICAL-METRICS-20261003/selected_cache_admission.json'
            original.parent.mkdir(parents=True); original.write_bytes(b'original immutable admission')
            with self.assertRaisesRegex(RuntimeError, 'rebuilding or overwriting it is disabled'):
                build(root)
            self.assertEqual(original.read_bytes(), b'original immutable admission')

    def audit(self):
        return dict(status='PASS', GPU=False, studies=[dict(adapter=adapter, study=study, frames=count,
            sources=100, methods=['frozen_method'], selected_rows_sha256='original-row-sha', status='PASS')
            for adapter,study,count in ROSTER])

    def test_constructor_identity_preserves_rows_methods_counts_and_order(self):
        original = self.audit(); identity = selected_identity(original)
        for key, value in [('selected_rows_sha256','changed'), ('methods',['other']), ('sources',99), ('frames',1)]:
            changed = deepcopy(original); changed['studies'][0][key] = value
            try:
                result = selected_identity(changed)
            except RuntimeError:
                continue
            self.assertNotEqual(result, identity, key)
        changed = deepcopy(original); changed['studies'].reverse()
        with self.assertRaisesRegex(RuntimeError, 'ordering'):
            selected_identity(changed)

    def test_new_queue_changes_only_runtime_and_additive_proofs(self):
        original = dict(status='REGISTERED', source_bindings={'v1.py':'old'},
            input_proof_bindings={'v1_cache.json':'cache', 'v1_audit.json':'audit'},
            shared_metric_bindings={'evaluator.py':'fixed'},
            jobs=[dict(adapter='original',study='FIRST')], contrasts=[dict(name='registered')],
            coverage_manifest=dict(path='v1_coverage.json',sha256='coverage'),
            execution_groups=dict(required=['FIRST'],analysis_supplements=[]),
            training_updates=0,policy_selection_updates=0,scope='original selection',stop_after_scope=True)
        before = deepcopy(original)
        result = continuation_queue(original, {'r2.py':'new'}, original_path='v1_queue.json',
            original_sha='queue', audit_path='r2_audit.json', audit_sha='newaudit')
        self.assertEqual(original, before)
        self.assertEqual(result['source_bindings'], {'r2.py':'new'})
        self.assertEqual(result['input_proof_bindings'], {**original['input_proof_bindings'],
            'v1_queue.json':'queue', 'r2_audit.json':'newaudit'})
        for key in original.keys()-{'source_bindings','input_proof_bindings'}:
            self.assertEqual(result[key], original[key], key)

    def test_conflicting_old_proof_is_not_rebound(self):
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            continuation_queue(dict(input_proof_bindings={'v1_queue.json':'old'}), {},
                original_path='v1_queue.json',original_sha='changed',audit_path='r2_audit.json',audit_sha='new')


if __name__ == '__main__':
    unittest.main()
