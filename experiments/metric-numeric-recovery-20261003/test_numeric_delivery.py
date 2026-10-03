"""CPU fixtures for sequential numeric compatibility publication."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numeric_delivery as pub


class NumericDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.runtime = self.root / 'runtime'; self.runtime.mkdir()
        (self.runtime / 'adapter.py').write_text('# frozen numeric adapter\n')
        self.raw = self.root / 'raw.csv'; self.raw.write_text('snr_db\n13\n')
        self.checkpoint = self.root / 'checkpoint.json'; pub.write(self.checkpoint, {'unchanged': True})
        self.cpu = self.root / 'numeric_snr_qualification.json'
        pub.write(self.cpu, dict(status='REAL_NUMERIC_SNR_COMPATIBILITY_PASS', inputs=pub.bindings([self.raw])))
        self.real = self.root / 'real_first_source.json'
        pub.write(self.real, dict(status='REAL_FIRST_SOURCE_PARITY_PASS', parity_passed=True,
            synthetic=False, source_index=0, original_source_row_ids_preserved=True,
            checkpoints_bound=pub.bindings([self.checkpoint]), inputs=pub.bindings([self.raw, self.cpu])))
        self.manifest = self.root / 'outputs' / pub.NAME / 'numeric_publication_manifest.json'
        pub.write(self.manifest, dict(status='NUMERIC_RECOVERY_READY_FOR_PUBLICATION', scope=pub.SCOPE,
            source_bindings=pub.bindings(pub.source_files(self.runtime)),
            proof_bindings=pub.bindings([self.cpu, self.real]), artifacts=pub.bindings([self.cpu, self.real]),
            qualification_receipts=dict(cpu=str(self.cpu.resolve()), real_first_source=str(self.real.resolve())),
            training_updates=0, policy_selection_updates=0, inference_functions_changed=False,
            original_values_changed=False, row_hashes_changed=False, model_weights_changed=False))

    def update(self, path, **fields):
        value = pub.read(path); value.update(fields); pub.write(path, value)

    def rebind_proofs(self):
        self.update(self.manifest, proof_bindings=pub.bindings([self.cpu, self.real]),
                    artifacts=pub.bindings([self.cpu, self.real]))

    def delivered(self):
        canonical = self.root / 'experiments/m2-json-recovery-20261002'; canonical.mkdir(parents=True)
        helper = canonical / 'publish_when_complete.py'; helper.write_text('# frozen helper\n')
        old = dict(status='PUSHED', checks='PASS', commit='a' * 40, remote_commit='a' * 40,
                   source_bindings=pub.bindings([helper]), published_files=pub.bindings([helper]))
        pub.write(self.root / 'outputs' / pub.JSON_NAME / 'publication.json', old)
        record = dict(status='PUSHED', checks='PASS', commit='b' * 40, remote_commit='b' * 40,
                      published_files=pub.bindings([helper]), inputs=pub.bindings([self.raw, helper]))
        folder = self.root / 'outputs' / pub.PREVIOUS_NAME
        pub.write(folder / 'publication.json', record)
        pub.write(folder / 'publication_status.json', dict(status='COMPLETE', publication=record, stop=True, pid=400))

    def test_waits_for_telemetry_publication(self):
        with self.assertRaises(pub.NotReady): pub.previous_delivery_gate(self.root, {})

    def test_complete_exited_predecessor_passes(self):
        self.delivered()
        value = pub.previous_delivery_gate(self.root, {})
        self.assertEqual(value['status'], 'TELEMETRY_PUBLISHED_AND_EXITED')
        self.assertIn(str(self.root / 'outputs' / pub.PREVIOUS_NAME / 'publication.json'), value['inputs'])

    def test_live_or_reused_pid_waits(self):
        self.delivered()
        for process in (dict(state='S', start_ticks='999'), dict(unreadable=True)):
            with self.subTest(process=process), self.assertRaises(pub.NotReady):
                pub.previous_delivery_gate(self.root, {400: process})

    def test_zombie_predecessor_passes(self):
        self.delivered(); pub.previous_delivery_gate(self.root, {400: dict(state='Z')})

    def test_wrong_push_or_status_rejected(self):
        self.delivered()
        path = self.root / 'outputs' / pub.PREVIOUS_NAME / 'publication.json'
        self.update(path, remote_commit='c' * 40)
        with self.assertRaises(RuntimeError): pub.previous_delivery_gate(self.root, {})

    def test_changed_published_framework_rejected(self):
        self.delivered()
        (self.root / 'experiments/m2-json-recovery-20261002/publish_when_complete.py').write_text('# changed\n')
        with self.assertRaises(RuntimeError): pub.previous_delivery_gate(self.root, {})

    def test_valid_numeric_scope_and_receipts(self):
        value = pub.numeric_gate(self.root, self.runtime, self.manifest)
        self.assertFalse(value['scientific_result'])
        self.assertIn(str(self.checkpoint.resolve()), value['inputs'])

    def test_changed_runtime_rejected(self):
        (self.runtime / 'adapter.py').write_text('# changed\n')
        with self.assertRaises(RuntimeError): pub.numeric_gate(self.root, self.runtime, self.manifest)

    def test_changed_original_scope_rejected(self):
        for field in ('inference_functions_changed', 'original_values_changed', 'row_hashes_changed', 'model_weights_changed'):
            with self.subTest(field=field):
                self.update(self.manifest, **{field: True})
                with self.assertRaises(RuntimeError): pub.numeric_gate(self.root, self.runtime, self.manifest)
                self.update(self.manifest, **{field: False})

    def test_synthetic_or_nonfirst_source_rejected(self):
        for fields in (dict(synthetic=True), dict(source_index=1), dict(source_index=False),
                       dict(original_source_row_ids_preserved=False), dict(parity_passed=False)):
            with self.subTest(fields=fields):
                self.update(self.real, **fields); self.rebind_proofs()
                with self.assertRaises(RuntimeError): pub.numeric_gate(self.root, self.runtime, self.manifest)
                self.update(self.real, synthetic=False, source_index=0, original_source_row_ids_preserved=True,
                            parity_passed=True); self.rebind_proofs()

    def test_cpu_native_test_pass_cannot_substitute_scope_qualification(self):
        self.update(self.cpu, status='PASS'); self.rebind_proofs()
        with self.assertRaises(RuntimeError): pub.numeric_gate(self.root, self.runtime, self.manifest)

    def test_checkpoint_and_input_changes_rejected(self):
        self.checkpoint.write_text('{}\n')
        with self.assertRaises(RuntimeError): pub.numeric_gate(self.root, self.runtime, self.manifest)

    def test_receipt_whitelist_required(self):
        self.update(self.manifest, qualification_receipts=dict(cpu=str(self.cpu)))
        with self.assertRaises(RuntimeError): pub.numeric_gate(self.root, self.runtime, self.manifest)

    def test_artifact_outside_project_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            outside = Path(folder) / 'outside.json'; outside.write_text('{}')
            self.update(self.manifest, artifacts={**pub.bindings([self.cpu, self.real]), **pub.bindings([outside])})
            with self.assertRaises(RuntimeError): pub.numeric_gate(self.root, self.runtime, self.manifest)

    def test_no_git_or_helper_load_while_waiting(self):
        with patch.object(pub, 'process_snapshot', return_value={}), \
             patch.object(pub, 'load_base', side_effect=AssertionError('base loaded early')), \
             patch('subprocess.run', side_effect=AssertionError('Git run early')):
            self.assertIsNone(pub.main(self.root, runtime=self.runtime, manifest=self.manifest, once=True))

    def test_explicit_receipt_arguments_must_match_manifest(self):
        with patch.object(pub, 'process_snapshot', side_effect=AssertionError('checked too late')):
            with self.assertRaises(RuntimeError):
                pub.main(self.root, runtime=self.runtime, manifest=self.manifest,
                         cpu_qualification=self.raw, once=True)

    def test_frozen_framework_reused_with_only_declared_changes(self):
        relatives = ('.research/m2_json_recovery_20261002/publish_when_complete.py',
                     'experiments/m2-json-recovery-20261002/publish_when_complete.py',
                     'outputs/M2-JSON-RECOVERY-20261002/runtime/publish_when_complete.py')
        path = next(ancestor / relative for ancestor in Path(__file__).resolve().parents
                    for relative in relatives if (ancestor / relative).is_file())
        spec = importlib.util.spec_from_file_location('_fixture_numeric_base', path)
        base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
        adapted = pub.adapted_publisher(base, self.root)
        strings = {value for value in adapted.__code__.co_consts if isinstance(value, str)}
        self.assertIn('experiments/metric-numeric-recovery-20261003', strings)
        self.assertIn('results/metric_numeric_recovery_20261003', strings)
        self.assertNotIn('experiments/m2-json-recovery-20261002', strings)
        self.assertEqual(adapted.__globals__['NAME'], pub.NAME)
        self.assertIs(adapted.__globals__['recovery_gate'], pub.numeric_gate)
        self.assertEqual(base.NAME, pub.JSON_NAME)

    def test_no_early_waiter_deadlock_marker(self):
        self.assertEqual(Path(pub.__file__).name, 'numeric_delivery.py')
        self.assertNotEqual(Path(pub.__file__).name, 'publish_when_complete.py')


if __name__ == '__main__': unittest.main()
