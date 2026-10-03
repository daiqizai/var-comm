"""Engineering-only recovery routing and original-owner gate tests."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import history_controller as controller
import phase2_completed as recovery
from history_common import read,write,sha


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.folder=self.root/'outputs'/recovery.OLD
        self.source=self.root/'fixture.py';self.source.write_text('# engineering fixture\n')
        bound={str(self.source):sha(self.source)}
        self.queue=dict(source_bindings=bound,shared_metric_bindings=bound,input_proof_bindings=bound)
        write(self.folder/'queue_registration.json',self.queue)
        publication=dict(status='PUSHED',checks='PASS',commit='a'*40,remote_commit='a'*40,
            runtime_source_bindings=bound,inputs=bound,published_files=bound,source_bindings=bound)
        write(self.folder/'source_publication.json',publication)
        for name,pid in (('controller_launch',101),('controller_dispatch',101),('bootstrap_launch',102)):
            write(self.folder/(name+'.json'),dict(pid=pid,start_ticks=str(pid)))
        write(self.folder/'launches/phase2.json',dict(pid=103,start_ticks='103'))
        self.base_gate=dict(inputs={},commits=['a'*40])

    def tearDown(self):self.temp.cleanup()

    def gate(self,processes):
        with patch.object(controller,'original_parent_gate',return_value=self.base_gate):
            return controller.parent_gate(self.root,processes)

    def test_all_old_owners_exited_and_bound(self):
        result=self.gate({})
        self.assertFalse(result['original_r3_complete_claimed'])
        self.assertIn(str(self.folder/'bootstrap_launch.json'),result['inputs'])
        self.assertIn(str(self.folder/'launches/phase2.json'),result['inputs'])

    def test_old_controller_cannot_overlap(self):
        with self.assertRaises(controller.NotReady):self.gate({101:dict(state='S',start_ticks='101')})

    def test_bootstrap_cannot_overlap(self):
        with self.assertRaises(controller.NotReady):self.gate({102:dict(state='S',start_ticks='102')})

    def test_old_scoring_child_cannot_overlap(self):
        with self.assertRaises(controller.NotReady):self.gate({103:dict(state='S',start_ticks='103')})

    def test_pid_reuse_is_not_old_owner(self):
        self.assertTrue(self.gate({101:dict(state='S',start_ticks='999')}))

    def test_changed_old_source_refused(self):
        self.source.write_text('# changed\n')
        with self.assertRaises(RuntimeError):self.gate({})

    def test_only_last_study_can_have_new_path(self):
        queue={'phase2_recovery':{}}
        out,result=recovery.study_paths(self.root,queue,recovery.STUDY)
        self.assertEqual(out,self.root/'outputs'/recovery.NAME/recovery.STUDY)
        with self.assertRaises(RuntimeError):recovery.study_paths(self.root,queue,'NEW_UNREGISTERED_STUDY')

    def test_completed_r3_routing_is_read_only(self):
        study='OPTIONAL_H6_13DB';out=self.folder/study;result=self.root/'results'/recovery.OLD_RESULT/study
        queue={'completed_r3_studies':{study:dict(out=str(out),result=str(result))},'phase2_recovery':{}}
        self.assertEqual(recovery.study_paths(self.root,queue,study),(out,result))


if __name__=='__main__':unittest.main()
