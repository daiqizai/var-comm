"""CPU-only bounded pilot owner checks, without constructing actual PHY."""
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_pilot_phy_v1 as h


class PilotOwnerTests(unittest.TestCase):
    def test_all_downstream_caps_are_finite_before_CPU_work(self):
        caps=h.scientific_caps()
        self.assertEqual(caps['logical_frames'],100*3*144)
        self.assertEqual(caps['actual_packet_decodes'],2*caps['logical_frames'])
        self.assertEqual(caps['visual']['source_rx'],caps['visual']['var_render'])
        self.assertEqual(caps['visual']['prior_scale'],20*caps['logical_frames'])
        self.assertEqual(caps['metrics']['model_constructions'],3)
        self.assertEqual(caps['metrics']['lpips_alexnet_backbone_forward'],2*caps['logical_frames'])
        self.assertEqual(caps['new_confirmation_source_reads'],0)

    def test_execution_bound_is_independent_explicit_and_CPU_only(self):
        result=h.execution(time.time()+21600,21600)
        self.assertEqual(result['CPU_workers'],1);self.assertEqual(result['CPU_affinity_count'],2)
        self.assertFalse(result['CUDA_visible'])
        for seconds in (0,21601,True):
            with self.assertRaises(RuntimeError):h.execution(time.time()+30000,seconds)
        with self.assertRaises(RuntimeError):h.execution(time.time()-1,3600)

    def test_private_worker_identity_matches_newfile_not_old_owner(self):
        before=h.source.engine().worker_identity.__globals__['__file__']
        runner=h.engine()
        self.assertEqual(runner.worker_identity.__globals__['__file__'],h.__file__)
        self.assertEqual(h.source.engine().worker_identity.__globals__['__file__'],before)

    def test_source68_binding_tamper_stops_before_prepare(self):
        with mock.patch.object(h,'SOURCE_SCRIPT_SHA','0'*64),self.assertRaisesRegex(RuntimeError,'implementation changed'):
            h.engine()

    def source_pins(self):
        root=h.gate.RT/'qualification/h800_ep_pilot_source68_v1_attempt1'
        return dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
                    owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))

    def test_source68_actual_wait_failure_cannot_be_success(self):
        owner=dict(status=h.source.PASS,actual_children_waited=True,actual_wait=dict(success=True),worker_exit_codes=[0])
        wait=dict(actual_wait=True,returncode=1)
        with mock.patch.object(h.gate,'readpin',side_effect=[owner,wait]),self.assertRaisesRegex(RuntimeError,'owner wait'):
            h.source_closure(self.source_pins())

    def test_source68_different_attempt_cannot_replace_expected_receipt(self):
        pins=self.source_pins();pins['completion']['path']=pins['completion']['path'].replace('attempt1','attempt2')
        with self.assertRaisesRegex(RuntimeError,'Exact source68 attempt'):h.source_closure(pins)

    def test_source68_unresolved_reservation_stops_before_input_read(self):
        owner=dict(status=h.source.PASS,actual_children_waited=True,actual_wait=dict(success=True),worker_exit_codes=[0],
                   worker_completion=dict(path='worker.json',sha256='c'*64))
        wait=dict(actual_wait=True,returncode=0)
        worker=dict(status=h.source.PASS,counts=dict(completed=h.source.CAPS,reserved=h.source.CAPS,unresolved=1),
                    source_count=100,new68_streams=[{}]*68)
        with mock.patch.object(h.gate,'readpin',side_effect=[owner,wait,worker]),self.assertRaisesRegex(RuntimeError,'must actually close'):
            h.source_closure(self.source_pins())

    def test_CPU_registration_cannot_admit_GPU_or_metric_execution(self):
        deadline=time.time()+600
        request=dict(schema=h.SCHEMA,status='REGISTERED_NOT_EXECUTED',execution=h.execution(deadline,600),
            complete_scientific_caps=h.scientific_caps(),records=[],streams=[],phy_identity={},reuse48={},policy={},frames=[],
            source68_closed={},tool_bindings={},GPU_execution_admitted=False,metrics_execution_admitted=False,automatic_successor=False)
        for field in ('GPU_execution_admitted','metrics_execution_admitted','automatic_successor'):
            altered=dict(request);altered[field]=True
            with mock.patch.object(h.g,'inside',side_effect=Path),mock.patch.object(h.g,'checked_json',return_value=altered),\
                 mock.patch.object(h,'source_closure',return_value=([],[],{},{})),mock.patch.object(h,'phy_binding',return_value=({},{})),\
                 mock.patch.object(h.gate,'policy_and_static',return_value=({}, {}, {})),mock.patch.object(h,'reuse48',return_value={}),\
                 mock.patch.object(h.core,'frames',return_value=[]),mock.patch.object(h,'engine',return_value=types.SimpleNamespace(check_tools=lambda x:None)),\
                 self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'Only finite CPU'):
                h.registration('request.json','a'*64)

    def test_already_claimed_CPU_owner_never_launches_again(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'request.json';path.write_text('{}');(path.parent/'run').mkdir()
            class Lock:
                def __enter__(self):return self
                def __exit__(self,*a):pass
            shared=mock.Mock();shared.owner_lock.return_value=Lock()
            args=mock.Mock(request=str(path),request_sha256='a'*64)
            with mock.patch.object(h.g.sys,'platform','linux'),mock.patch.object(h.g,'inside',side_effect=Path),\
                 mock.patch.object(h,'registration',return_value=(dict(execution={}),{})),\
                 mock.patch.object(h.g,'helper',return_value=shared),mock.patch.object(h.subprocess,'Popen') as launch,\
                 self.assertRaisesRegex(RuntimeError,'already claimed'):
                h.run(args)
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
