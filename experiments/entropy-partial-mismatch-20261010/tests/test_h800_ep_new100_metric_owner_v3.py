"""Narrow CPU tests for recovery-result binding only; no scientific calls."""
from pathlib import Path
import sys,tempfile,types,unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_metric_owner_v3 as m
from test_h800_ep_new100_metric_owner_v2 import fixture as alternate_fixture

def fixture():
    pins,docs,visual,vr,worker=alternate_fixture()
    old='h800_ep_new100_visual_v2_attempt1';new='h800_ep_new100_visual_v3_attempt1'
    for pin in pins.values():pin['path']=pin['path'].replace(old,new)
    docs={k.replace(old,new):v for k,v in docs.items()};done=docs[pins['completion']['path']]
    done['worker_completion']['path']=done['worker_completion']['path'].replace(old,new)
    done['schema']=worker['schema']=visual.SCHEMA='H800_EP_NEW100_FOUR_ARM_VISUAL_V3_RECOVERY_GPU0'
    done['status']=worker['status']=visual.PASS='PASS_H800_EP_NEW100_RECOVERY_GPU0_RECONSTRUCTIONS_ONLY'
    vr['target_index']=0;vr['resource_policy']=m.resources.policy(0)
    return pins,docs,visual,vr,worker

class RecoveryMetricTests(unittest.TestCase):
    def closure(self,f):
        pins,docs,visual,_,_=f
        with mock.patch.object(m.gate,'readpin',side_effect=lambda p:docs[p['path']]),mock.patch.object(m.g,'sha',return_value='a'*64),\
             mock.patch.object(m,'visual_implementation',return_value=visual),mock.patch.object(m.resources,'validate_policy'):
            return m.visual_closure(pins)

    def test_actual_gpu0_closed_recovery_result_admitted(self):
        f=fixture();vr,_=self.closure(f);self.assertIs(vr,f[3]);self.assertEqual(vr['target_index'],0)
        self.assertEqual(vr['resource_policy']['prelaunch_utilization_max'],50)

    def test_prior_failed_visual_or_incomplete_wait_cannot_satisfy_gate(self):
        f=fixture();f[1][f[0]['completion']['path']]['status']='STOPPED_NO_RETRY'
        with self.assertRaises(RuntimeError):self.closure(f)
        for key in ('timeout','interrupted_or_timeout'):
            f=fixture();f[1][f[0]['owner_actual_wait']['path']][key]=True
            with self.assertRaises(RuntimeError):self.closure(f)

    def test_visual_original_registration_is_executed_not_skipped(self):
        f=fixture();f[2].registration=mock.Mock(side_effect=RuntimeError('Preserved failed-attempt evidence differs'))
        with self.assertRaisesRegex(RuntimeError,'Preserved'):self.closure(f)
        f[2].registration.assert_called_once()

    def test_either_earlier_metric_registration_prevents_replay(self):
        for version in (1,2):
            with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m.gate,'RT',Path(tmp)):
                m.previous_attempt_unexecuted()
                (Path(tmp)/f'qualification/h800_ep_new100_metrics_v{version}_attempt1/registered').mkdir(parents=True)
                with self.assertRaises(RuntimeError):m.previous_attempt_unexecuted()

    def test_same_four_metric_codeobjects_and_budgets(self):
        for name in ('configuration','scores','run','worker'):
            self.assertIs(getattr(m,name).__code__,getattr(m.base,name).__code__)
            self.assertIs(getattr(m,name).__globals__,vars(m))
        self.assertIs(m.metric,m.base.metric);self.assertEqual(m.CAPS,m.base.CAPS)
        self.assertEqual(m.CAPS['model_constructions'],3);self.assertEqual(m.CAPS['image_scores'],3600)

    def test_pending_visual_pin_is_a_real_execution_block(self):
        with mock.patch.object(m,'VISUAL_SHA','PENDING'),mock.patch.object(m.g,'import_file') as loader:
            with self.assertRaises(RuntimeError):m.visual_implementation()
            loader.assert_not_called()

    def test_target_cannot_drift_from_closed_visual(self):
        f=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            a=types.SimpleNamespace(out=str(Path(tmp)/'fresh'),visual_completion=f[0]['completion']['path'],
                visual_completion_sha256='a'*64,visual_owner_wait=f[0]['owner_actual_wait']['path'],visual_owner_wait_sha256='b'*64,target_index=3)
            with mock.patch.object(m,'previous_attempt_unexecuted'),mock.patch.object(m.g,'inside',side_effect=Path),\
                 mock.patch.object(m,'visual_closure',return_value=(f[3],f[4])),mock.patch.object(m.resources,'DeviceAdapter') as device:
                with self.assertRaises(RuntimeError):m.prepare(a,None)
                device.assert_not_called()

if __name__=='__main__':unittest.main()
