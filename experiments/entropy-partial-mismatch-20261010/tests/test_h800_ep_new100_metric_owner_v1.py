"""Synthetic finite-owner admission tests; no SSH, real input or model calls."""
import contextlib
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_metric_owner_v1 as m
from test_ep_new100_metric_core_v1 import fake


def fixture():
    r,_,_=fake();caps=m.metric.plan.scientific_caps()['visual'];root=m.gate.RT/'qualification/h800_ep_new100_visual_v1_attempt1'
    pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
        owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
    wait=dict(actual_wait=True,success=True,returncode=0)
    d=dict(schema='H800_EP_NEW100_FOUR_ARM_VISUAL_V1',status='PASS_H800_EP_NEW100_FOUR_ARM_RECONSTRUCTIONS_ONLY',
        request_sha256='c'*64,actual_wait=wait,actual_children_waited=True,worker_exit_codes=[0],
        worker_completion=dict(path=str(root/'registered/run/worker_completion.json'),sha256='d'*64))
    counts={k:0 for k in caps};counts.update(model_load=1,source_rx=1,var_render=1,decoder_forward=1,prior_scale=20)
    vr=dict(r,population=m.metric.POPULATION,caps=caps,inputs={'policy_bundle':{'path':'p','sha256':'p'},'source_manifest':{'path':'s','sha256':'s'}})
    worker=dict(schema=d['schema'],status=d['status'],request_sha256='c'*64,quality_scores=0,
        results=dict(logical_frames=r['reconstruction_results'],counts=dict(caps=caps,reserved=dict(counts),completed=dict(counts),unresolved=0)))
    docs={pins['completion']['path']:d,pins['owner_actual_wait']['path']:dict(actual_wait=True,returncode=0),
        d['worker_completion']['path']:worker,str(root/'registered/run/actual_child_wait.json'):wait}
    visual=types.SimpleNamespace(SCHEMA=d['schema'],PASS=d['status'],CAPS=caps,engine=lambda:None,registration=lambda *a:vr)
    return pins,docs,visual,vr,worker


class MetricOwnerTests(unittest.TestCase):
    def closure(self,f):
        pins,docs,visual,_,_=f
        with mock.patch.object(m.gate,'readpin',side_effect=lambda p:docs[p['path']]),mock.patch.object(m.g,'sha',return_value='a'*64),\
             mock.patch.object(m,'visual_implementation',return_value=visual),mock.patch.object(m.resources,'validate_policy'):
            return m.visual_closure(pins)

    def test_exact_closed_visual_can_be_bound_without_scoring(self):
        f=fixture();vr,worker=self.closure(f);self.assertIs(vr,f[3]);self.assertIs(worker,f[4])

    def test_failed_or_unwaited_visual_stops_before_provider(self):
        for key,value in [('actual_children_waited',False),('worker_exit_codes',[1])]:
            f=fixture();f[1][f[0]['completion']['path']][key]=value
            with self.assertRaises(RuntimeError):self.closure(f)
        for key in ('timeout','interrupted_or_timeout'):
            f=fixture();f[1][f[0]['owner_actual_wait']['path']][key]=True
            with self.assertRaises(RuntimeError):self.closure(f)

    def test_wrong_visual_population_or_cap_rejected(self):
        f=fixture();f[3]['population']='original_holdout500'
        with self.assertRaises(RuntimeError):self.closure(f)
        f=fixture();f[3]['caps']=dict(f[3]['caps'],source_rx=1800)
        with self.assertRaises(RuntimeError):self.closure(f)

    def test_missing_logical_rows_or_unresolved_visual_refused(self):
        f=fixture();f[4]['results']['logical_frames']=f[4]['results']['logical_frames'][:-1]
        with self.assertRaises(RuntimeError):self.closure(f)
        f=fixture();f[4]['results']['counts']['unresolved']=1
        with self.assertRaises(ValueError):self.closure(f)

    def test_quality_or_source_calls_cannot_hide_in_visual_gate(self):
        f=fixture();f[4]['quality_scores']=1
        with self.assertRaises(RuntimeError):self.closure(f)
        f=fixture();f[4]['results']['counts']['completed']['encoder']=1;f[4]['results']['counts']['reserved']['encoder']=1
        with self.assertRaises(ValueError):self.closure(f)

    def test_actual_wait_and_worker_request_must_match(self):
        f=fixture();f[4]['request_sha256']='e'*64
        with self.assertRaises(RuntimeError):self.closure(f)
        f=fixture();key=next(k for k in f[1] if k.endswith('actual_child_wait.json'));f[1][key]=dict(actual_wait=True,success=False,returncode=1)
        with self.assertRaises(RuntimeError):self.closure(f)

    def test_metric_configuration_is_new_population_with_no_old_cache(self):
        vr=fixture()[3]
        with mock.patch.object(m.original,'configuration',return_value={'whole_policy':'historical_anchor'}):
            config=m.configuration({},None,vr)
        self.assertEqual(config['confirmation_population'],m.metric.POPULATION)
        self.assertEqual(config['confirmation_policy_bundle'],vr['inputs']['policy_bundle'])
        self.assertEqual(config['confirmation_source_manifest'],vr['inputs']['source_manifest'])
        self.assertFalse(config['policy_selection']);self.assertFalse(config['historical_score_reuse_allowed'])

    def test_unfrozen_visual_placeholder_blocks_import(self):
        with mock.patch.object(m,'VISUAL_SHA','PENDING'),mock.patch.object(m.g,'import_file') as loader:
            with self.assertRaises(RuntimeError):m.visual_implementation()
            loader.assert_not_called()

    def test_engine_clones_runtime_functions_and_preserves_original_module(self):
        before=dict(vars(m.original));old=m.original.engine
        def simple():return 1
        source=types.SimpleNamespace(**{k:simple for k in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity')})
        def digest(path):
            if str(path)==m.original.__file__:return m.ORIGINAL_SHA
            if str(path)==m.metric.__file__:return m.CORE_SHA
            if str(path)==m.resources.__file__:return m.RESOURCE_SHA
            raise AssertionError(str(path))
        with mock.patch.object(m.g,'sha',side_effect=digest),mock.patch.object(m,'visual_implementation',return_value=types.SimpleNamespace(TOOLS=())),mock.patch.object(m.original,'engine',return_value=source):
            runner=m.engine();self.assertEqual(runner.CAPS,m.CAPS);self.assertEqual(runner.SCHEMA,m.SCHEMA)
            self.assertIs(runner.DeviceAdapter,m.resources.DeviceAdapter)
            for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):self.assertIs(getattr(runner,name).__code__,simple.__code__)
        self.assertEqual(before,dict(vars(m.original)));self.assertIs(m.original.engine,old)

    def test_existing_run_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'request.json';path.write_text('{}');(Path(tmp)/'run').mkdir()
            a=types.SimpleNamespace(request=str(path),request_sha256='a'*64)
            r=dict(spec={},execution={});runner=types.SimpleNamespace(device_from_request=lambda r:types.SimpleNamespace(index=2))
            shared=types.SimpleNamespace(owner_lock=lambda p:contextlib.nullcontext())
            with mock.patch.object(m,'registration',return_value=r),mock.patch.object(m.g,'inside',side_effect=Path),\
                 mock.patch.object(m.g,'helper',return_value=shared),mock.patch.object(m.g.sys,'platform','linux'),mock.patch.object(m.subprocess,'Popen') as popen:
                with self.assertRaises(RuntimeError):m.run(a,runner)
                popen.assert_not_called()

if __name__=='__main__':unittest.main()
