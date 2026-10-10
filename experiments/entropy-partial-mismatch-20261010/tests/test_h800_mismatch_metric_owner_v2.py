"""CPU-only closure/identity tests; no metric model or tensor is constructed."""
from pathlib import Path
import sys
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import h800_mismatch_metric_owner_v1 as old
import h800_mismatch_metric_owner_v2 as new


class GPU0MetricTests(unittest.TestCase):
    def test_compute_and_owner_code_objects_unchanged(self):
        for name in ('scores','configuration','run','worker','engine'):
            self.assertIs(getattr(old,name).__code__,getattr(new,name).__code__,name)
        self.assertEqual(new.CAPS,old.CAPS)
        self.assertIs(new.metric,old.metric)
        self.assertEqual(new.CAPS['image_scores'],4500)

    def test_private_namespace_has_actual_gpu0_visual(self):
        self.assertEqual(new.visual.SCHEMA,'H800_RAW100_ONE_BIN_MISMATCH_VISUAL_V2_GPU0')
        self.assertEqual(old.visual.SCHEMA,'H800_RAW100_ONE_BIN_MISMATCH_VISUAL_V1')
        self.assertEqual(new.scores.__globals__['__file__'],new.__file__)
        self.assertEqual(new.engine().worker_identity.__globals__['__file__'],new.__file__)
        self.assertIn('h800_mismatch_metric_owner_v1.py',new.TOOLS)
        self.assertIn('h800_mismatch_visual_v2.py',new.TOOLS)

    def test_old_visual_path_cannot_admit_metric_stage(self):
        root=new.gate.RT/'qualification/h800_mismatch_visual_v1_attempt1'
        pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
            owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
        with mock.patch.object(new.gate,'readpin') as read,self.assertRaisesRegex(RuntimeError,'Exact raw mismatch visual attempt'):
            new.visual_closure(pins)
        read.assert_not_called()

    def test_failed_actual_gpu0_visual_wait_blocks_metrics(self):
        root=new.gate.RT/'qualification/h800_mismatch_visual_v2_attempt1'
        pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
            owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
        done=dict(status=new.visual.PASS,actual_wait=dict(success=True),actual_children_waited=True,worker_exit_codes=[0])
        with mock.patch.object(new.gate,'readpin',side_effect=[done,dict(actual_wait=True,returncode=1)]),\
            self.assertRaisesRegex(RuntimeError,'Actual raw visual owner'):
            new.visual_closure(pins)

    def test_gpu2_registered_metric_identity_rejected(self):
        with mock.patch.object(new,'original_registration',return_value=dict(target_index=2)),self.assertRaisesRegex(RuntimeError,'GPU0-only'):
            new.registration(None,None,None)
        with mock.patch.object(new,'original_registration',return_value=dict(target_index=0)):
            self.assertEqual(new.registration(None,None,None),dict(target_index=0))

    def test_cli_refuses_gpu2(self):
        args=['owner','prepare','--visual-completion','a','--visual-completion-sha256','b','--visual-owner-wait','c',
            '--visual-owner-wait-sha256','d','--device-receipt','e','--device-receipt-sha256','f','--out','g',
            '--deadline-unix','9999999999','--target-index','2']
        with mock.patch.object(sys,'argv',args),mock.patch('sys.stderr'),self.assertRaises(SystemExit) as caught:new.main()
        self.assertEqual(caught.exception.code,2)


if __name__=='__main__':unittest.main()
