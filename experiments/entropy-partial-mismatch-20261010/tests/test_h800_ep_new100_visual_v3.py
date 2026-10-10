import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_visual_v3 as v


def fixture():
    zero=dict.fromkeys(v.CAPS,0);sha=v.FAILURE_PINS['registered/request.json'];error="ResourceBusy('GPU free plus verified own reserved margin')"
    return {'registered/request.json':dict(schema=v.base.SCHEMA,caps=v.CAPS,target_index=3),
        'run_owner_actual_wait.json':dict(actual_wait=True,returncode=1,interrupted_or_timeout=False),
        'registered/run/actual_child_wait.json':dict(actual_child_waited=True,child_exit_code=1,success=False),
        'registered/run/worker_failure.json':dict(request_sha256=sha,error=error,counts=dict(caps=v.CAPS,
            completed=zero,reserved=dict(zero,model_load=1),unresolved=1)),
        'registered/run/failure.json':dict(actual_child_waited=True,child_exit_code=1,status='STOPPED_NO_RETRY'),
        'registered/run/child_started.json':dict(request_sha256=sha,pid=17,owner_pid=16),
        'registered/run/calls/0001_model_load.reserved.json':dict(kind='model_load',pid=17),
        'registered/run/resource_guard_failures/4103632_1791649935351311421.json':dict(pid=17,request_sha256=sha,
            error=error,scientific_caps_changed=False,snapshot=dict(resource_guard=dict(policy=v.resources.policy(3))))}


class RecoveryTests(unittest.TestCase):
    def test_startup_only_failure_is_admitted(self):
        data=fixture();self.assertIs(v.validate_failure(data),data['registered/request.json'])

    def test_any_previous_science_or_second_load_is_rejected(self):
        for name in v.CAPS:
            data=fixture();data['registered/run/worker_failure.json']['counts']['reserved'][name]+=1
            with self.assertRaises(RuntimeError):v.validate_failure(data)

    def test_interrupted_or_unwaited_failure_is_rejected(self):
        for key in ('timeout','interrupted_or_timeout'):
            data=fixture();data['run_owner_actual_wait.json'][key]=True
            with self.assertRaises(RuntimeError):v.validate_failure(data)
        data=fixture();data['registered/run/actual_child_wait.json']['actual_child_waited']=False
        with self.assertRaises(RuntimeError):v.validate_failure(data)

    def test_failure_reservation_resource_identity_cannot_diverge(self):
        data=fixture();data['registered/run/calls/0001_model_load.reserved.json']['pid']=18
        with self.assertRaises(RuntimeError):v.validate_failure(data)
        data=fixture();data['registered/run/resource_guard_failures/4103632_1791649935351311421.json']['snapshot']['resource_guard']['policy']=v.resources.policy(0)
        with self.assertRaises(RuntimeError):v.validate_failure(data)

    def test_private_original_worker_and_owner_have_complete_globals(self):
        runner=v.engine()
        self.assertIs(v.run.__code__,v.base.run.__code__)
        self.assertIs(v.worker.__code__,v.base.worker.__code__)
        self.assertIs(v.images,v.base.images)
        self.assertIs(v.run.__globals__['registration'],v.registration)
        self.assertEqual(v.worker.__globals__['__file__'],v.__file__)
        self.assertEqual(runner.check_tools.__globals__['__file__'],v.__file__)
        for name in ('time','os','subprocess','traceback','signal','Path'):
            self.assertIn(name,v.run.__globals__)
        self.assertIs(runner.DeviceAdapter,v.resources.DeviceAdapter)
        self.assertEqual(v.CAPS,v.base.CAPS)

    def test_registration_requires_same_inputs_recovery_and_gpu0(self):
        old=dict(CPU_closed={'fixed':True});recovery={'previous_unresolved_model_load':1}
        r=dict(target_index=0,recovery_budget=recovery,CPU_closed=old['CPU_closed'])
        with patch.object(v,'_base_registration',return_value=r),patch.object(v,'failed_attempt',return_value=(old,recovery)),patch.object(v.g,'inside',side_effect=lambda p:p):
            self.assertIs(v.registration(v.OUT/'request.json','fake',None),r)
            r['target_index']=3
            with self.assertRaises(RuntimeError):v.registration(v.OUT/'request.json','fake',None)


if __name__=='__main__':unittest.main()
