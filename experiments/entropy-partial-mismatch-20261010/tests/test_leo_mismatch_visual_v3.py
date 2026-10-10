"""CPU-only recovery-budget/native-identity tests; no scientific execution."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import leo_mismatch_visual_v3 as new
import h800_mismatch_visual_v1 as old


class LeoRawRecoveryTests(unittest.TestCase):
    def failure_values(self):
        zero=dict.fromkeys(new.CAPS,0)
        return {'registered/request.json':dict(historical_image_reuse_admitted=0),
            'run_owner_actual_wait.json':dict(actual_wait=True,returncode=1,timeout=False),
            'registered/run/actual_child_wait.json':dict(actual_child_waited=True,child_exit_code=1,success=False),
            'registered/run/worker_failure.json':dict(counts=dict(caps=new.CAPS,reserved=dict(zero,model_load=1),completed=zero,unresolved=1),
                request_sha256=new.FAILURE_PINS['registered/request.json'],error="ResourceBusy('Less than20GiB GPU margin')"),
            'registered/run/calls/0001_model_load.reserved.json':dict(kind='model_load')}

    def checked(self,values):
        return lambda p,h:copy.deepcopy(values[Path(p).relative_to(new.FAILED_ROOT).as_posix()])

    def test_raw_compute_functions_are_original_code_objects(self):
        for name in ('parser','received_tokens','actual_state_key','render_actual','images','worker'):
            self.assertIs(getattr(new,name).__code__,getattr(old,name).__code__,name)
        self.assertEqual(new.CAPS,old.CAPS)

    def test_native_host_and_device_are_private_dependencies(self):
        self.assertIs(new.worker.__globals__['g'],new.g)
        self.assertIsNot(new.g,old.g)
        self.assertIs(new.engine().device_from_request.__globals__['DeviceAdapter'],new.DeviceAdapter)
        self.assertEqual(new.engine().worker_identity.__globals__['__file__'],new.__file__)

    def test_prior_unresolved_load_is_preserved_in_cumulative_budget(self):
        with mock.patch.object(new.g,'checked_json',side_effect=self.checked(self.failure_values())):
            _,budget=new.failed_attempt()
        self.assertEqual(budget['previous_model_load_attempts'],1)
        self.assertEqual(budget['previous_unresolved_model_load'],1)
        self.assertEqual(budget['additional_model_load_attempt_cap'],1)
        self.assertEqual(budget['cumulative_model_load_attempt_cap'],2)
        self.assertEqual(budget['cumulative_render_cap'],4500)
        self.assertEqual(budget['cumulative_prior_cap'],45000)
        self.assertFalse(budget['automatic_retry'])

    def test_any_previous_reconstruction_or_other_fault_rejects_recovery(self):
        for change in ('var_render','error','returncode'):
            values=self.failure_values()
            if change=='var_render':values['registered/run/worker_failure.json']['counts']['reserved']['var_render']=1
            elif change=='error':values['registered/run/worker_failure.json']['error']='unexpected fault'
            else:values['run_owner_actual_wait.json']['returncode']=-9
            with self.subTest(change=change),mock.patch.object(new.g,'checked_json',side_effect=self.checked(values)),\
                self.assertRaisesRegex(RuntimeError,'Exact paid initialization failure'):
                new.failed_attempt()

    def test_device_only_accepts_original_leo_gpu5(self):
        receipt=dict(registered_device_identity=new.original_gate.device_identity_binding())
        with mock.patch.object(new.g,'checked_json',return_value=receipt):
            device=new.DeviceAdapter(new.g.native_pin(),5)
        self.assertEqual(device.device_identity_binding(),new.original_gate.device_identity_binding())
        with self.assertRaisesRegex(RuntimeError,'Only independently rechecked'):
            new.DeviceAdapter(new.g.native_pin(),0)

    def test_h800_identity_cannot_be_substituted(self):
        receipt=dict(registered_device_identity=dict(nvml_gpu_index=0))
        with mock.patch.object(new.g,'checked_json',return_value=receipt),self.assertRaisesRegex(RuntimeError,'Original leo device'):
            new.DeviceAdapter(new.g.native_pin(),5)

    def test_original_resource_refusal_is_preserved_without_changing_threshold(self):
        error=new.original_device.ResourceBusy('Less than20GiB GPU margin');error.resource_snapshot=dict(gpu=dict(free_MiB=19000))
        device=object.__new__(new.DeviceAdapter)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(new,'OUT',Path(d)),\
            mock.patch.object(new.g,'inside',side_effect=lambda p:p),\
            mock.patch.object(new.original_device.DeviceAdapter,'resource_snapshot',side_effect=error) as original,\
            self.assertRaises(new.original_device.ResourceBusy):
            try:device.resource_snapshot(None,True)
            finally:
                files=list((Path(d)/'resource_refusals').glob('*.json'));self.assertEqual(len(files),1)
                saved=json.loads(files[0].read_text());self.assertEqual(saved['snapshot'],error.resource_snapshot)
                self.assertTrue(saved['thresholds_unchanged']);original.assert_called_once_with(None,True)


if __name__=='__main__':unittest.main()
