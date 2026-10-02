"""CPU fixtures for final cache publication gates, with no scientific scoring."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import publish_extension as pub


class Fixture:
    def __init__(self, root):
        self.root = root
        self.out = root / 'outputs/METRIC-FINAL-CACHE-20261002'
        self.runtime = self.out / 'runtime'; self.runtime.mkdir(parents=True)
        (self.runtime / 'controller.py').write_text('# frozen new owner\n')
        self.bound = pub.bindings(pub.source_files(self.runtime))
        original = root / 'experiments/unified-metrics-20261002'; original.mkdir(parents=True)
        (original / 'runner.py').write_text('# original frozen\n')
        self.original = pub.bindings(pub.source_files(original))
        self.old = root / 'outputs/METRIC-CONCURRENT-R4-20261002'
        (self.old / 'runtime').mkdir(parents=True)
        (self.old / 'runtime/controller.py').write_text('# old frozen\n')
        self.oldbound = pub.bindings(pub.source_files(self.old / 'runtime'))
        self.oldowner = dict(pid=200, start_ticks='20')
        for name in ('runtime_registration.json', 'admission.json', 'concurrent_registration.json'):
            pub.write(self.old / name, {'fixture': name})
        pub.write(self.old / 'controller_launch.json', self.oldowner)
        pub.write(self.old / 'controller_status.json', dict(pid=200,status='WAITING_FOR_ORIGINAL_COMPLETION',worker_pid=None))
        pub.write(self.old / 'partial_launch.json', dict(pid=201,start_ticks='21'))
        pub.write(self.old / 'partial_completion.json', dict(sources_complete=100,outputs={str(i):'fixture' for i in range(100)}))
        hand = dict(new_source_bindings=self.bound, previous_owner=self.oldowner,
            previous_launch_sha256=pub.sha(self.old / 'controller_launch.json'),
            previous_runtime_registration_sha256=pub.sha(self.old / 'runtime_registration.json'))
        pub.write(self.out / 'handoff.json', hand)
        reg = dict(status='REGISTERED_FINAL_CACHE_GUARDIAN',runtime_source_bindings=self.bound,
            original_metric_source_bindings=self.original,training_updates=0,policy_selection_updates=0,
            previous_runtime_registration_sha256=pub.sha(self.old / 'runtime_registration.json'),
            handoff_sha256=pub.sha(self.out / 'handoff.json'))
        pub.write(self.out / 'runtime_registration.json', reg)
        history = self.out / 'launch_history/300.json'
        launch = dict(status='DETACHED_CPU_GUARDIAN',pid=300,start_ticks='30',source_bindings=self.bound,
            runtime_registration_sha256=pub.sha(self.out / 'runtime_registration.json'),history_path=str(history))
        pub.write(self.out / 'controller_launch.json',launch); pub.write(history,launch)
        names=('concurrent_registration.json','partial_completion.json','controller_launch.json',
               'controller_status.json','runtime_registration.json','admission.json','partial_launch.json')
        pub.write(self.out / 'r4_retirement.json',dict(status='R4_IDLE_METRIC_OWNER_RETIRED',
            oldowner=self.oldowner,original_scientific_owner_untouched=True,
            proof_bindings=pub.bindings([self.old/n for n in names])))
        evidence = self.root / 'scientific_evidence.json'; pub.write(evidence, {'fixture': True})
        b=pub.bindings([evidence]); self.base=types.SimpleNamespace(
            parent_gate=lambda *_:dict(inputs=b,parent_commit='a'*40),
            concurrent_gate=lambda *_:(self.oldbound,dict(inputs=b)),
            predecessor_gate=lambda *_:dict(inputs=b), inherited_cache_gate=lambda *_:dict(inputs=b))
        self.oldpub=dict(runtime_source_bindings=self.oldbound,inputs=b,commit='b'*40)
        self.oldpubpath=self.old/'extension_source_publication.json';pub.write(self.oldpubpath,self.oldpub)
        self.processes={300:dict(state='S',start_ticks='30',command=(self.runtime/'controller.py').as_posix())}

    def gate(self):
        with patch.object(pub,'load_r4_publisher',return_value=(self.base,self.oldpub,self.oldpubpath)):
            return pub.final_cache_gate(self.root,self.out,self.runtime,self.processes,300,'30')

    def change(self, path, **fields):
        v=pub.read(path);v.update(fields);pub.write(path,v)


class PublisherGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(Path(self.tmp.name))

    def test_valid_full_cache(self):
        bound, evidence=self.f.gate()
        self.assertEqual(bound,self.f.bound);self.assertEqual(evidence['cache_sources_complete'],100)
        self.assertNotIn(str(self.f.out/'controller_launch.json'),evidence['inputs'])
        self.assertIn(str(self.f.out/'launch_history/300.json'),evidence['inputs'])
        self.assertFalse(evidence['scientific_result'])

    def test_partial_cache_rejected(self):
        self.f.change(self.f.old/'partial_completion.json',sources_complete=99)
        with self.assertRaisesRegex(RuntimeError,'100'): self.f.gate()

    def test_live_retired_owner_rejected(self):
        self.f.processes[200]=dict(state='S',start_ticks='20',command='old-controller')
        with self.assertRaisesRegex(RuntimeError,'not exited'): self.f.gate()

    def test_idle_boundary_required(self):
        self.f.change(self.f.old/'controller_status.json',status='CONCURRENT_SCORING')
        receipt=self.f.out/'r4_retirement.json';v=pub.read(receipt)
        v['proof_bindings'][str(self.f.old/'controller_status.json')]=pub.sha(self.f.old/'controller_status.json');pub.write(receipt,v)
        with self.assertRaisesRegex(RuntimeError,'idle'): self.f.gate()

    def test_new_workers_rejected(self):
        self.f.processes[301]=dict(state='S',start_ticks='31',command=(self.f.runtime/'final_runner.py').as_posix())
        with self.assertRaisesRegex(RuntimeError,'workers'): self.f.gate()

    def test_changed_runtime_rejected(self):
        (self.f.runtime/'controller.py').write_text('# changed\n')
        with self.assertRaisesRegex(RuntimeError,'registration'): self.f.gate()

    def test_original_source_change_rejected(self):
        (self.f.root/'experiments/unified-metrics-20261002/runner.py').write_text('# changed original\n')
        with self.assertRaisesRegex(RuntimeError,'registration'): self.f.gate()

    def test_guardian_pid_reuse_rejected(self):
        self.f.processes[300]['start_ticks']='31'
        with self.assertRaisesRegex(RuntimeError,'identity'): self.f.gate()

    def test_immutable_launch_history_required(self):
        self.f.change(self.f.out/'launch_history/300.json',start_ticks='999')
        with self.assertRaisesRegex(RuntimeError,'history'): self.f.gate()

    def test_missing_retirement_evidence_rejected(self):
        p=self.f.out/'r4_retirement.json';v=pub.read(p)
        del v['proof_bindings'][str(self.f.old/'admission.json')];pub.write(p,v)
        with self.assertRaisesRegex(RuntimeError,'evidence'): self.f.gate()

    def test_original_science_untouched_required(self):
        self.f.change(self.f.out/'r4_retirement.json',original_scientific_owner_untouched=False)
        with self.assertRaisesRegex(RuntimeError,'retired'): self.f.gate()

    def test_zombie_old_owner_allowed(self):
        self.f.processes[200]=dict(state='Z',start_ticks='20',command='old-controller')
        self.f.gate()

    def test_unchecked_push_record_rejected(self):
        with self.assertRaises(RuntimeError): pub.published(dict(status='PUSHED',commit='a'*40,remote_commit='a'*40,checks='FAIL'))

    def test_unknown_source_suffix_rejected(self):
        (self.f.runtime/'model.pt').write_bytes(b'unexpected')
        with self.assertRaisesRegex(RuntimeError,'artifact'):self.f.gate()


if __name__=='__main__': unittest.main()
