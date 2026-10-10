"""CPU-only alternate-H800 metric control tests; no real inputs or model calls."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_metric_owner_v2 as m
from test_h800_ep_new100_metric_owner_v1 import fixture as prior_fixture

def fixture():
    pins,docs,visual,vr,worker=prior_fixture()
    old='h800_ep_new100_visual_v1_attempt1';new='h800_ep_new100_visual_v2_attempt1'
    for pin in pins.values():pin['path']=pin['path'].replace(old,new)
    docs={k.replace(old,new):v for k,v in docs.items()}
    done=docs[pins['completion']['path']];done['worker_completion']['path']=done['worker_completion']['path'].replace(old,new)
    done['schema']=worker['schema']='H800_EP_NEW100_FOUR_ARM_VISUAL_V2'
    done['status']=worker['status']='PASS_H800_EP_NEW100_ALTERNATE_H800_RECONSTRUCTIONS_ONLY'
    visual.SCHEMA=done['schema'];visual.PASS=done['status'];vr['target_index']=3
    vr['device_receipt']=dict(path=str(m.resources.DEVICE_PATH),sha256=m.resources.DEVICE_SHA)
    vr['resource_policy']=m.resources.policy(3)
    return pins,docs,visual,vr,worker

class AlternateMetricTests(unittest.TestCase):
    def closure(self,f):
        pins,docs,visual,_,_=f
        with mock.patch.object(m.gate,'readpin',side_effect=lambda p:docs[p['path']]),mock.patch.object(m.g,'sha',return_value='a'*64),\
             mock.patch.object(m,'visual_implementation',return_value=visual),mock.patch.object(m.resources,'validate_policy'):
            return m.visual_closure(pins)

    def test_closed_gpu3_visual_is_admitted_without_gpu2_requirement(self):
        f=fixture();vr,_=self.closure(f);self.assertEqual(vr['target_index'],3)

    def test_original_schema_or_failed_wait_not_accepted(self):
        for key,value in [('schema','H800_EP_NEW100_FOUR_ARM_VISUAL_V1'),('worker_exit_codes',[1])]:
            f=fixture();f[1][f[0]['completion']['path']][key]=value
            with self.assertRaises(RuntimeError):self.closure(f)
        for key in ('timeout','interrupted_or_timeout'):
            f=fixture();f[1][f[0]['owner_actual_wait']['path']][key]=True
            with self.assertRaises(RuntimeError):self.closure(f)

    def test_caps_or_foreign_device_cannot_expand_admission(self):
        f=fixture();f[3]['target_index']=4
        with self.assertRaises(RuntimeError):self.closure(f)
        f=fixture();f[3]['caps']=dict(f[3]['caps'],source_rx=601)
        with self.assertRaises(RuntimeError):self.closure(f)

    def test_score_runtime_configuration_code_objects_preserved(self):
        for name in ('configuration','scores','run','worker'):
            self.assertIs(getattr(m,name).__code__,getattr(m.base,name).__code__)
            self.assertIs(getattr(m,name).__globals__,vars(m))
        self.assertIs(m.metric,m.base.metric)
        self.assertIs(m.metric_environment,m.base.metric_environment)
        self.assertIs(m.runtime_identity,m.base.runtime_identity)
        self.assertNotEqual(m.resources,m.base.resources)
        self.assertEqual(m.base.SCHEMA,'H800_EP_NEW100_FOUR_METRIC_OWNER_V1')

    def test_original_v1_registered_attempt_blocks_new_budget(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(m.gate,'RT',Path(tmp)):
            m.previous_attempt_unexecuted()
            p=Path(tmp)/'qualification/h800_ep_new100_metrics_v1_attempt1/registered';p.mkdir(parents=True)
            with self.assertRaises(RuntimeError):m.previous_attempt_unexecuted()

    def test_prepare_target_must_match_closed_visual_before_model_or_device(self):
        f=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            a=types.SimpleNamespace(out=str(Path(tmp)/'new'),visual_completion=f[0]['completion']['path'],
                visual_completion_sha256='a'*64,visual_owner_wait=f[0]['owner_actual_wait']['path'],visual_owner_wait_sha256='b'*64,target_index=2)
            with mock.patch.object(m,'previous_attempt_unexecuted'),mock.patch.object(m.g,'inside',side_effect=Path),\
                 mock.patch.object(m,'visual_closure',return_value=(f[3],f[4])),mock.patch.object(m.resources,'DeviceAdapter') as device:
                with self.assertRaises(RuntimeError):m.prepare(a,None)
                device.assert_not_called();self.assertFalse(Path(a.out).exists())

    def test_gpu3_policy_only_changes_authorized_utilization(self):
        policy=m.resources.policy(3)
        self.assertEqual(policy['prelaunch_utilization_max'],80)
        self.assertEqual(policy['prelaunch_utilization_comparison'],'strictly_below')
        self.assertEqual(policy['prelaunch_free_bytes'],20*(1<<30))
        self.assertEqual(policy['own_allocator_cap_bytes'],16*(1<<30))
        self.assertEqual(policy['runtime_free_bytes'],4*(1<<30))
        self.assertEqual(m.CAPS,m.base.CAPS)

    def test_private_engine_uses_new_resource_without_mutating_base(self):
        before=dict(vars(m.base))
        def simple():return 1
        engine=types.SimpleNamespace(**{k:simple for k in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity')})
        pins={m.base.__file__:m.BASE_SHA,m.original.__file__:m.ORIGINAL_SHA,m.metric.__file__:m.CORE_SHA,m.resources.__file__:m.RESOURCE_SHA}
        with mock.patch.object(m.g,'sha',side_effect=lambda p:pins[str(p)]),mock.patch.object(m,'visual_implementation',return_value=types.SimpleNamespace(TOOLS=())),\
             mock.patch.object(m.original,'engine',return_value=engine):
            runner=m.engine();self.assertIs(runner.DeviceAdapter,m.resources.DeviceAdapter)
            self.assertEqual(runner.CAPS,m.base.CAPS);self.assertEqual(runner.SCHEMA,m.SCHEMA)
            for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):
                self.assertIs(getattr(runner,name).__code__,simple.__code__)
        self.assertEqual(before,dict(vars(m.base)))

    def test_runtime_identity_comes_from_actual_device(self):
        device=types.SimpleNamespace(device_identity_binding=lambda:{'physical_index':3,'UUID':'observed-only'})
        with mock.patch.object(m.original,'configuration',return_value={'runtime_identity':{'device_identity':device.device_identity_binding()}}):
            value=m.configuration({},device,fixture()[3])
        self.assertEqual(value['runtime_identity']['device_identity']['physical_index'],3)
        self.assertFalse(value['historical_score_reuse_allowed']);self.assertFalse(value['policy_selection'])

if __name__=='__main__':unittest.main()
