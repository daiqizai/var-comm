"""CPU-only identity/admission tests for separate GPU0 raw reconstruction."""
from pathlib import Path
import sys
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import h800_mismatch_visual_v1 as old
import h800_mismatch_visual_v2 as new


class GPU0AdmissionTests(unittest.TestCase):
    def test_same_compute_code_objects_and_caps(self):
        for name in ('render_actual','received_tokens','actual_state_key','images','worker','run','visual_identity','cpu_closure'):
            self.assertIs(getattr(new,name).__code__,getattr(old,name).__code__,name)
        self.assertEqual(new.CAPS,old.CAPS)
        self.assertEqual(new.CAPS['var_render'],4500)
        self.assertEqual(new.CAPS['prior_scale'],45000)

    def test_private_namespace_and_tool_closure(self):
        self.assertEqual(new.images.__globals__['__file__'],new.__file__)
        self.assertEqual(old.images.__globals__['__file__'],old.__file__)
        self.assertNotEqual(new.SCHEMA,old.SCHEMA)
        self.assertIn('h800_mismatch_visual_v1.py',new.TOOLS)
        self.assertEqual(new.engine().worker_identity.__globals__['__file__'],new.__file__)

    def test_gpu2_registration_rejected_without_running_any_model(self):
        with mock.patch.object(new,'original_registration',return_value=dict(target_index=2)),self.assertRaisesRegex(RuntimeError,'GPU0-only'):
            new.registration(None,None,None)
        with mock.patch.object(new,'original_registration',return_value=dict(target_index=0)):
            self.assertEqual(new.registration(None,None,None),dict(target_index=0))

    def test_cli_only_admits_gpu0(self):
        args=['owner','prepare','--cpu-completion','a','--cpu-completion-sha256','b','--cpu-owner-wait','c',
            '--cpu-owner-wait-sha256','d','--device-receipt','e','--device-receipt-sha256','f','--out','g',
            '--deadline-unix','9999999999','--target-index','2']
        with mock.patch.object(sys,'argv',args),mock.patch('sys.stderr'),self.assertRaises(SystemExit) as caught:new.main()
        self.assertEqual(caught.exception.code,2)

    def test_gpu_identity_changes_image_cache_key(self):
        state=dict(kind='tokens',prefix=[[1]],partial_values=[],m=1,K=0)
        self.assertNotEqual(new.actual_state_key(state,dict(nvml_gpu_index=0)),new.actual_state_key(state,dict(nvml_gpu_index=2)))


if __name__=='__main__':unittest.main()
