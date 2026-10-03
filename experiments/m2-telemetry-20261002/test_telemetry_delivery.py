"""CPU fixtures for sequential telemetry publication; no GPU or Git mutation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import telemetry_delivery as pub


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.runtime=self.root/'runtime';self.runtime.mkdir()
        (self.runtime/'adapter.py').write_text('# frozen\n')
        self.source=pub.bindings(pub.source_files(self.runtime));self.proof=self.root/'engineering.json'
        pub.write(self.proof,dict(status='PASS'))
        self.manifest=self.root/'outputs'/pub.NAME/'telemetry_publication_manifest.json'
        pub.write(self.manifest,dict(status='TELEMETRY_READY_FOR_PUBLICATION',source_bindings=self.source,
            proof_bindings=pub.bindings([self.proof]),artifacts=pub.bindings([self.proof]),
            training_updates=0,policy_selection_updates=0,inference_functions_changed=False,
            scientific_parity_status='PASS',healthy_query_ttl_seconds=1.0))

    def update(self,path,**fields):
        value=pub.read(path);value.update(fields);pub.write(path,value)

    def delivered(self):
        self.old=self.root/'outputs'/pub.JSON_NAME
        canonical=self.root/'experiments/m2-json-recovery-20261002';canonical.mkdir(parents=True)
        helper=canonical/'publish_when_complete.py';helper.write_text('# frozen helper\n')
        record=dict(status='PUSHED',checks='PASS',commit='a'*40,remote_commit='a'*40,
            source_bindings=pub.bindings([helper]),published_files=pub.bindings([helper]),inputs=pub.bindings([self.proof]))
        pub.write(self.old/'publication.json',record)
        pub.write(self.old/'publication_status.json',dict(status='COMPLETE',publication=record,stop=True,pid=400))

    def test_waits_for_json_publication(self):
        with self.assertRaises(pub.NotReady):pub.json_delivery_gate(self.root,{})

    def test_json_complete_and_exited_passes(self):
        self.delivered();self.assertEqual(pub.json_delivery_gate(self.root,{})['status'],'JSON_RECOVERY_PUBLISHED_AND_EXITED')

    def test_json_owner_live_or_reused_pid_waits(self):
        self.delivered()
        with self.assertRaises(pub.NotReady):pub.json_delivery_gate(self.root,{400:dict(state='S',start_ticks='999')})

    def test_json_zombie_allowed(self):
        self.delivered();pub.json_delivery_gate(self.root,{400:dict(state='Z')})

    def test_json_changed_helper_rejected(self):
        self.delivered();p=self.root/'experiments/m2-json-recovery-20261002/publish_when_complete.py';p.write_text('# changed\n')
        with self.assertRaises(RuntimeError):pub.json_delivery_gate(self.root,{})

    def test_telemetry_scope(self):
        pub.telemetry_gate(self.root,self.runtime,self.manifest)
        self.update(self.manifest,healthy_query_ttl_seconds=5)
        with self.assertRaises(RuntimeError):pub.telemetry_gate(self.root,self.runtime,self.manifest)

    def test_nonfinite_ttl_rejected(self):
        self.update(self.manifest,healthy_query_ttl_seconds='NaN')
        with self.assertRaises(RuntimeError):pub.telemetry_gate(self.root,self.runtime,self.manifest)

    def test_changed_runtime_rejected(self):
        (self.runtime/'adapter.py').write_text('# changed\n')
        with self.assertRaises(RuntimeError):pub.telemetry_gate(self.root,self.runtime,self.manifest)

    def test_inference_change_rejected(self):
        self.update(self.manifest,inference_functions_changed=True)
        with self.assertRaises(RuntimeError):pub.telemetry_gate(self.root,self.runtime,self.manifest)

    def test_no_git_or_helper_load_while_waiting(self):
        with patch.object(pub,'process_snapshot',return_value={}),patch.object(pub,'load_base',side_effect=AssertionError('base loaded early')),patch.object(pub.subprocess,'run',side_effect=AssertionError('Git run early')):
            self.assertIsNone(pub.main(self.root,runtime=self.runtime,manifest=self.manifest,once=True))

    def test_frozen_framework_reused_with_declared_changes(self):
        path=Path(__file__).parent.parent/'m2_json_recovery_20261002/publish_when_complete.py'
        if not path.is_file():
            # CPU checks run before JSON publication: its frozen ignored runtime
            # is sufficient for this structural fixture. Production still waits
            # for the checked canonical publication before loading the helper.
            relatives=('experiments/m2-json-recovery-20261002/publish_when_complete.py',
                       'outputs/M2-JSON-RECOVERY-20261002/runtime/publish_when_complete.py',
                       '.research/m2_json_recovery_20261002/publish_when_complete.py')
            path=next(a/relative for a in Path(__file__).resolve().parents for relative in relatives
                      if (a/relative).is_file())
        spec=importlib.util.spec_from_file_location('_fixture_json_publisher',path)
        base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        adapted=pub.adapted_publisher(base,self.root)
        strings={x for x in adapted.__code__.co_consts if isinstance(x,str)}
        self.assertIn('experiments/m2-telemetry-20261002',strings)
        self.assertIn('results/m2_telemetry_20261002',strings)
        self.assertNotIn('experiments/m2-json-recovery-20261002',strings)
        self.assertEqual(adapted.__globals__['NAME'],pub.NAME)
        self.assertIs(adapted.__globals__['recovery_gate'],pub.telemetry_gate)
        self.assertEqual(base.NAME,pub.JSON_NAME)

    def test_no_early_waiter_deadlock_marker(self):
        self.assertEqual(Path(pub.__file__).name,'telemetry_delivery.py')
        self.assertNotEqual(Path(pub.__file__).name,'publish_when_complete.py')


if __name__=='__main__':unittest.main()
