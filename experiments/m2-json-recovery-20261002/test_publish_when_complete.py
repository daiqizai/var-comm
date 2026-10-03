"""CPU-only publication sequencing fixtures; never launch a scientific worker."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import publish_when_complete as pub


class Fixture:
    def __init__(self,root):
        self.root=root;self.runtime=root/'runtime';self.runtime.mkdir()
        (self.runtime/'startup.py').write_text('# immutable fixture\n')
        self.proof=root/'proof.json';pub.write(self.proof,{'fixture':True})
        self.bound=pub.bindings(pub.source_files(self.runtime));self.out=root/'outputs'/pub.NAME
        self.manifest=self.out/'recovery_publication_manifest.json'
        pub.write(self.manifest,dict(status='RECOVERY_READY_FOR_PUBLICATION',source_bindings=self.bound,
            proof_bindings=pub.bindings([self.proof]),artifacts=pub.bindings([self.proof]),training_updates=0,
            policy_selection_updates=0,allowed_json_type='numpy.bool_',gate_derivation='first_200_from_frozen_screen'))
        self.pub=dict(status='PUSHED',checks='PASS',commit='a'*40,remote_commit='a'*40,source_bindings=self.bound)

    def complete(self):
        self.parent=self.root/'outputs'/pub.PARENT;self.metric=self.root/'outputs'/pub.METRICS
        for folder,pid,status in ((self.parent,100,'AUTHORIZED_TWO_METHODS_COMPLETE'),(self.metric,200,'UNIFIED_METRICS_COMPLETE')):
            pub.write(folder/'completion.json',dict(status=status,synthetic=False,training_updates=0,policy_selection_updates=0,stop=True,publication=self.pub))
            pub.write(folder/'supervisor_status.json',dict(status='COMPLETE',pid=pid,stopped_after_authorized_scope=True))
            pub.write(folder/'supervisor_launch.json',dict(pid=pid,start_ticks=str(pid)))
        final=self.root/'outputs'/pub.FINAL
        pub.write(final/'controller_launch.json',dict(pid=200,start_ticks='200'))
        pub.write(final/'controller_completion.json',dict(status='COMPLETE',pid=200,start_ticks='200',training_updates=0,
            policy_selection_updates=0,stopped_after_authorized_scope=True,source_bindings=self.bound,
            metrics_completion_sha256=pub.sha(self.metric/'completion.json')))
        r4=self.root/'outputs'/pub.R4
        for i,name in enumerate(('m2_controller_launch.json','controller_launch.json','partial_launch.json')):
            pub.write(r4/name,dict(pid=300+i,start_ticks=str(300+i)))

    def update(self,path,**fields):
        v=pub.read(path);v.update(fields);pub.write(path,v)


class RecoveryPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(Path(self.tmp.name))

    def test_gate_waits_for_missing_complete(self):
        with self.assertRaises(pub.NotReady):pub.completion_gate(self.f.root,{})

    def test_complete_and_exited_passes(self):
        self.f.complete();v=pub.completion_gate(self.f.root,{})
        self.assertEqual(v['status'],'ALL_AUTHORIZED_RESULTS_PUBLISHED_AND_OWNERS_EXITED')

    def test_alive_final_guardian_blocks(self):
        self.f.complete()
        with self.assertRaises(pub.NotReady):pub.completion_gate(self.f.root,{200:dict(state='S',start_ticks='200',command='controller')})

    def test_alive_m2_owner_blocks(self):
        self.f.complete()
        with self.assertRaises(pub.NotReady):pub.completion_gate(self.f.root,{300:dict(state='S',start_ticks='300',command='m2')})

    def test_unaccounted_worker_blocks(self):
        self.f.complete()
        with self.assertRaises(pub.NotReady):pub.completion_gate(self.f.root,{999:dict(state='S',command='/experiments/unified-metrics-20261002/runner.py')})

    def test_other_publisher_blocks(self):
        self.f.complete()
        with self.assertRaises(pub.NotReady):pub.completion_gate(self.f.root,{999:dict(state='S',command='/repo/publish.py --phase results')})

    def test_other_git_writer_blocks(self):
        with self.assertRaises(pub.NotReady):pub.no_publishers({999:dict(state='S',command='git commit -F body')},own_pid=1)

    def test_read_only_git_allowed(self):
        pub.no_publishers({999:dict(state='S',command='git status --short')},own_pid=1)

    def test_self_publisher_allowed(self):
        pub.no_publishers({1:dict(state='S',command='/runtime/publish_when_complete.py')},own_pid=1)

    def test_zombie_owners_allowed(self):
        self.f.complete();pub.completion_gate(self.f.root,{200:dict(state='Z',start_ticks='200',command='controller')})

    def test_unpushed_results_rejected(self):
        self.f.complete();v=pub.read(self.f.metric/'completion.json');v['publication']['status']='COMMITTED';pub.write(self.f.metric/'completion.json',v)
        with self.assertRaises(RuntimeError):pub.completion_gate(self.f.root,{})

    def test_changed_final_evidence_rejected(self):
        self.f.complete();self.f.update(self.f.metric/'completion.json',extra='changed')
        with self.assertRaisesRegex(RuntimeError,'guardian'):pub.completion_gate(self.f.root,{})

    def test_recovery_scope_accepts_only_exact_type(self):
        pub.recovery_gate(self.f.root,self.f.runtime,self.f.manifest)
        self.f.update(self.f.manifest,allowed_json_type='numpy.generic')
        with self.assertRaisesRegex(RuntimeError,'scope'):pub.recovery_gate(self.f.root,self.f.runtime,self.f.manifest)

    def test_runtime_changes_rejected(self):
        (self.f.runtime/'startup.py').write_text('# modified\n')
        with self.assertRaisesRegex(RuntimeError,'inventory'):pub.recovery_gate(self.f.root,self.f.runtime,self.f.manifest)

    def test_proof_changes_rejected(self):
        self.f.proof.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'proof'):pub.recovery_gate(self.f.root,self.f.runtime,self.f.manifest)

    def test_no_git_call_while_waiting(self):
        handle=Mock()
        with patch.object(pub,'acquire_lock',return_value=handle),patch.object(pub,'process_snapshot',return_value={}),patch.object(pub.subprocess,'run',side_effect=AssertionError('Git invoked before completion')):
            result=pub.main(self.f.root,runtime=self.f.runtime,manifest=self.f.manifest,once=True)
        self.assertIsNone(result);handle.close.assert_called_once()


if __name__=='__main__':unittest.main()
