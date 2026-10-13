"""Synthetic CPU-only tests; no saved scientific ledger, image, model or channel read."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

P = Path(__file__).with_name('r0_finalize_resource_v2.py')
sys.path.insert(0, str(P.parent))
spec = importlib.util.spec_from_file_location('_finalizer_resource_adapter_test', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def registration(self):
        return dict(path=str(m.CLEAN_REGISTRATION), sha256=m.CLEAN_REGISTRATION_SHA)

    def test_original_and_new_clean_sources_have_exact_hashes(self):
        evidence = m.source_bindings()
        self.assertEqual(evidence['original_finalizer']['sha256'], m.ORIGINAL_SHA)
        self.assertEqual(evidence['clean_entry']['sha256'], m.CLEAN_SHA)

    def test_original_bler_audit_is_exact_function(self):
        self.assertIs(m.audit_bler, m.original.audit_bler)
        self.assertEqual(m.SCHEMA, m.original.SCHEMA)

    def test_clone_preserves_function_code_and_default_objects(self):
        def sample(a=1, *, b=2):
            return a + b
        copied = m.clone(sample, {})
        self.assertIs(copied.__code__, sample.__code__)
        self.assertIs(copied.__defaults__, sample.__defaults__)
        self.assertIs(copied.__kwdefaults__, sample.__kwdefaults__)
        self.assertEqual(copied(), 3)

    def test_exact_actual_v2_registration_required(self):
        with patch.object(m, 'source_bindings'), patch.object(m, 'checkpin') as verify:
            m.check_clean_registration(self.registration())
            verify.assert_called_once_with(self.registration())
            for p, h in [(m.CLEAN_REGISTRATION.with_name('other.json'), m.CLEAN_REGISTRATION_SHA),
                         (m.CLEAN_REGISTRATION, '0' * 64)]:
                with self.assertRaises(ValueError):
                    m.check_clean_registration(dict(path=str(p), sha256=h))

    def test_clean_validator_rebinding_does_not_change_original_module(self):
        captured = {}
        old_clean = m.original.clean
        def fake_clone(function, namespace):
            captured.update(function=function, namespace=namespace)
            return lambda registration: ('fake-quality', {'registration': registration})
        with patch.object(m, 'check_clean_registration'), patch.object(m, 'clone', side_effect=fake_clone):
            result = m.audit_clean(self.registration())
        self.assertIs(captured['function'], m.original.audit_clean)
        self.assertIs(captured['namespace']['clean'], m.revised_clean)
        self.assertIs(m.original.clean, old_clean)
        self.assertEqual(result[0], 'fake-quality')

    def test_finalizer_function_body_is_reused_and_provenance_injection_is_local(self):
        captured = {}
        old_save = m.original.bler.save
        def fake_clone(function, namespace):
            captured.update(function=function, namespace=namespace)
            return lambda *args: ('synthetic-result', args)
        with patch.object(m, 'check_clean_registration'), patch.object(m, 'clone', side_effect=fake_clone):
            result = m.finalize({'body': 'fixture'}, self.registration(), m.NOMINATION_OUTPUT)
        self.assertIs(captured['function'], m.original.finalize)
        self.assertIs(captured['namespace']['audit_clean'], m.audit_clean)
        self.assertIs(captured['namespace']['clean'], m.revised_clean)
        self.assertEqual(captured['namespace']['__file__'], str(P))
        self.assertIs(captured['namespace']['bler'].validate, m.original.bler.validate)
        self.assertIs(captured['namespace']['bler'].save, m.save_with_adapter_evidence)
        self.assertIs(m.original.bler.save, old_save)
        self.assertEqual(result[0], 'synthetic-result')

    def test_unique_authoritative_output_only(self):
        with patch.object(m, 'check_clean_registration'), patch.object(m, 'clone') as copied:
            with self.assertRaises(ValueError):
                m.finalize({}, self.registration(), m.NOMINATION_OUTPUT.with_name('alternate'))
            copied.assert_not_called()
        source = Path(m.original.__file__).read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'finalize')
        self.assertIn('New independent finalization directory required', ast.get_source_segment(source, f))

    def test_nominations_are_not_augmented_or_recomputed_by_save_wrapper(self):
        value = dict(status='synthetic-nominations', rows=[dict(score=0.1)])
        with patch.object(m.original.bler, 'save', return_value='saved') as save, patch.object(m, 'adapter_evidence') as evidence:
            self.assertEqual(m.save_with_adapter_evidence(Path('nominations.json'), value), 'saved')
            self.assertIs(save.call_args.args[1], value)
            evidence.assert_not_called()

    def test_closure_adds_only_adapter_metadata_without_mutating_original(self):
        value = dict(schema=m.SCHEMA, status='R0_COMPLETE_AUTHENTICATED_R1_NOT_REGISTERED',
                     no_new_science=True, clean={'sentinel': 1}, BLER={'sentinel': 2})
        with patch.object(m.original.bler, 'save', return_value='saved') as save, patch.object(m, 'adapter_evidence', return_value={'proof': 3}):
            m.save_with_adapter_evidence(Path('closure_audit.json'), value)
            actual = save.call_args.args[1]
        self.assertEqual(actual, dict(value, resource_adapter={'proof': 3}))
        self.assertNotIn('resource_adapter', value)
        self.assertIs(actual['clean'], value['clean'])
        self.assertIs(actual['BLER'], value['BLER'])

    def test_wrong_or_double_augmented_closure_rejected(self):
        good = dict(schema=m.SCHEMA, status='R0_COMPLETE_AUTHENTICATED_R1_NOT_REGISTERED', no_new_science=True)
        for key, val in [('schema', 'wrong'), ('status', 'wrong'), ('no_new_science', False), ('resource_adapter', {})]:
            with patch.object(m.original.bler, 'save') as save, self.assertRaises(ValueError):
                m.save_with_adapter_evidence(Path('closure_audit.json'), dict(good, **{key: val}))
            save.assert_not_called()

    def test_evidence_binds_both_sources_and_exact_clean_registration(self):
        expected = dict(original_finalizer={'sha256': m.ORIGINAL_SHA}, clean_entry={'sha256': m.CLEAN_SHA})
        with patch.object(m, 'source_bindings', return_value=expected), patch.object(m, 'pin', return_value=self.registration()), patch.object(m, 'check_clean_registration') as guard:
            result = m.adapter_evidence()
        guard.assert_called_once_with(self.registration())
        for key, value in expected.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['actual_clean_registration'], self.registration())
        self.assertEqual(result['resource_policy'], m.revised_clean.RESOURCE_POLICY)
        self.assertTrue(result['audit_function_bodies_unchanged'])
        self.assertEqual(result['scientific_calls'], 0)
        self.assertEqual(result['bootstrap_calls'], 0)

    def test_builder_can_verify_adapter_metadata_without_reauditing_science(self):
        finalizer = dict(path='fixture.py', sha256='a' * 64)
        evidence = dict(actual_clean_registration=self.registration(), proof='fixture')
        audit = dict(schema=m.SCHEMA, status='R0_COMPLETE_AUTHENTICATED_R1_NOT_REGISTERED',
                     finalizer=finalizer, resource_adapter=evidence, clean={'registration': self.registration()})
        with patch.object(m, 'pin', return_value=finalizer), patch.object(m, 'checkpin'), patch.object(m, 'adapter_evidence', return_value=evidence), patch.object(m, 'audit_clean') as clean_audit, patch.object(m, 'audit_bler') as body_audit:
            self.assertEqual(m.validate_adapter_evidence(audit), evidence)
            clean_audit.assert_not_called(); body_audit.assert_not_called()
            with self.assertRaises(ValueError):
                m.validate_adapter_evidence(dict(audit, resource_adapter={}))

    def test_source_hash_drift_rejected(self):
        with patch.object(m, 'sha', return_value='0' * 64), self.assertRaises(ValueError):
            m.source_bindings()


if __name__ == '__main__':
    unittest.main()
