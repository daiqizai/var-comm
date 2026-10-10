"""CPU-only host-binding and actual-closure tests; no scoring/model calls."""
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import h800_mismatch_metric_owner_v1 as old
import leo_mismatch_metric_owner_v3 as new


class LeoMetricTests(unittest.TestCase):
    def test_metric_math_and_resource_owner_code_objects_preserved(self):
        for name in ('scores','run','worker'):
            self.assertIs(getattr(new,name).__code__,getattr(old,name).__code__)
        self.assertIs(new.metric,old.metric)
        self.assertEqual(new.CAPS,old.CAPS)
        self.assertEqual(new.CAPS['image_scores'],4500)

    def test_current_host_dependencies_do_not_change_old_globals(self):
        self.assertIs(new.scores.__globals__['g'],new.visual.g)
        self.assertIsNot(old.g,new.g)
        self.assertIs(new.original.metric_environment.__globals__['g'],new.g)
        self.assertIs(new.original.worker.__globals__['g'],new.g)
        self.assertEqual(new.engine().worker_identity.__globals__['__file__'],new.__file__)
        self.assertIs(new.engine().device_from_request.__globals__['DeviceAdapter'],new.DeviceAdapter)

    def test_H800_path_cannot_admit_leo_metric(self):
        root=new.visual.RT/'qualification/h800_mismatch_visual_v2_attempt1'
        pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
            owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
        with mock.patch.object(new.gate,'readpin') as read,self.assertRaisesRegex(RuntimeError,'Exact raw mismatch visual attempt'):
            new.visual_closure(pins)
        read.assert_not_called()

    def test_actual_leo_failed_wait_blocks_metric(self):
        root=new.visual.RT/'qualification/leo_mismatch_visual_v3_attempt1'
        pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
            owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
        done=dict(status=new.visual.PASS,actual_wait=dict(success=True),actual_children_waited=True,worker_exit_codes=[0])
        with mock.patch.object(new.gate,'readpin',side_effect=[done,dict(actual_wait=True,returncode=1)]),\
            self.assertRaisesRegex(RuntimeError,'Actual raw visual owner'):
            new.visual_closure(pins)

    def test_metric_identity_binds_actual_leo_native_and_device(self):
        pin=dict(path='leo-new-native',sha256='c'*64);device=types.SimpleNamespace(device_identity_binding=lambda:dict(index=5,host='leo'))
        with mock.patch.dict(new.original.runtime_identity.__globals__,native_binding=lambda:(pin,[])):
            identity=new.original.runtime_identity(dict(environment=dict(path='old-projection',sha256='a'*64)),device)
        self.assertEqual(identity['base_host_native'],new.visual.g.native_pin())
        self.assertEqual(identity['metric_native_import'],pin)
        self.assertEqual(identity['device_identity'],dict(index=5,host='leo'))
        self.assertFalse(identity['old_host_numeric_identity_claimed'])

    def test_different_native_observation_schema_rejected(self):
        doc=dict(schema='H800_EP_METRIC_NATIVE_BINDING_V1',cuda_initialized=False,numerical_qualification=False)
        with mock.patch.object(new.gate,'readpin',return_value=doc),self.assertRaisesRegex(RuntimeError,'Actual leo CPU-hidden'):
            new.native_binding()

    def test_resource_refusals_have_independent_metric_directory(self):
        self.assertNotEqual(new.OUT,new.visual.OUT)
        self.assertIn('leo_mismatch_metrics_v3_attempt1',str(new.OUT))
        self.assertEqual(new.original.metric_environment.__code__,old.original.metric_environment.__code__)


if __name__=='__main__':unittest.main()
