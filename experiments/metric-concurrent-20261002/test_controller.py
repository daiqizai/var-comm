"""Synthetic CPU scheduling checks; the native fixture signals only its own parent."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from contextlib import nullcontext
import scheduling_guard as guard

HERE = Path(__file__).parent
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location('concurrent_controller_tested',HERE/'controller.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class Lock:
    def close(self): pass


class Fixture:
    def __init__(self,root):
        self.root = Path(root); self.out = self.root/'outputs'/c.NAME
        self.here = self.out/'runtime'; self.metrics = self.root/'outputs'/c.METRICS
        self.parent = self.root/'outputs'/c.PARENT
        self.oldsource = self.root/'experiments/unified-metrics-20261002'
        for p in (self.here,self.metrics,self.parent,self.oldsource): p.mkdir(parents=True)
        for p in (self.here/'controller.py',self.here/'scheduling_guard.py',self.here/'m2_controller.py',
                  self.oldsource/'supervisor.py',self.oldsource/'metric_models.py'):
            p.write_text('# ENGINEERING_SYNTHETIC_FIXTURE\n')
        self.bound = c.sources(self.here); self.oldbound = c.sources(self.oldsource)
        frozen = self.root/'frozen_guard.py'; frozen.write_text('# frozen original fixture\n')
        c.write(self.out/'admission.json',dict(status='REGISTERED_SCHEDULING_ADMISSION',allowed_roles=['metrics','m2'],
            exclusive_stages=['timing'],training_updates=0,policy_selection_updates=0,source_bindings=self.bound,
            original_guard_bindings={str(frozen):c.sha(frozen)},original_execution_bindings={str(frozen):c.sha(frozen)}))
        c.write(self.metrics/'supervisor_registration.json',dict(source_bindings=self.oldbound))
        c.write(self.metrics/'supervisor_launch.json',dict(pid=100,start_ticks='10'))
        c.write(self.metrics/'supervisor_status.json',dict(pid=100,status='WAITING_FOR_ORIGINAL_EXPERIMENTS'))
        c.write(self.metrics/'modelmanifest.json',dict(synthetic=True))
        digest = c.sha(self.metrics/'modelmanifest.json')
        c.write(self.metrics/'assets_complete.json',dict(status='ASSETS_READY',manifest_sha256=digest,environment=sys.executable))
        c.write(self.metrics/'models_qualification.json',dict(status='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS',
            modelmanifest_sha256=digest,source_bindings=self.oldbound))
        c.write(self.out/'handoff.json',dict(oldsupervisor=dict(pid=100,start_ticks='10'),new_source_bindings=self.bound,
            oldlaunch_sha256=c.sha(self.metrics/'supervisor_launch.json'),
            m2_baseline=dict(stage='m2_screen',seconds_per_source=100.21,sample_count=20),
            m2_resume_floor=dict(stage='m2_screen',sources=137)))
        self.ps = {100:dict(pid=100,start_ticks='10',state='T',command=(self.oldsource/'supervisor.py').as_posix()),
                   200:dict(pid=200,start_ticks='20',state='R',command='scientific M2'),
                   300:dict(pid=300,start_ticks='30',state='R',command=(self.here/'controller.py').as_posix())}
        c.write(self.parent/'supervisor_launch.json',dict(pid=200,start_ticks='20'))
        c.write(self.parent/'supervisor_status.json',dict(pid=200,status='RUNNING',stage='m2_calibration',worker_pid=400))
        self.retired = []
        def retire(expected,reader):
            self.retired.append(expected); self.ps.pop(expected['pid'])
        self.controller = c.Controller(self.root,self.here,pid=300,reader=self.ps.get,retire_fn=retire,
            lock_fn=lambda p:Lock(),sleep=lambda n:None,gpu=lambda:dict(used_mib=3000,total_mib=24576,free_mib=21576,
                temperature=45,thermal_slowdown=False))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.f = Fixture(self.temp.name)

    def test_takeover_signals_only_idle_metrics_owner_and_preserves_m2(self):
        f = self.f; files = [f.parent/'supervisor_launch.json',f.parent/'supervisor_status.json']
        before = [p.read_bytes() for p in files]
        f.controller.initialize(); f.controller.takeover()
        self.assertEqual(f.retired,[dict(pid=100,start_ticks='10')]); self.assertEqual(before,[p.read_bytes() for p in files])
        self.assertEqual(f.ps[200]['state'],'R')
        self.assertEqual(c.read(f.metrics/'supervisor_registration.json')['source_bindings'],f.oldbound)
        self.assertEqual(c.read(f.out/'controller_launch.json')['pid'],300)

    def previous_fixture(self):
        f = self.f; prior = f.root/'outputs/METRIC-CONCURRENT-R3-20261002'; runtime = prior/'runtime'; runtime.mkdir(parents=True)
        for n in ('controller.py','concurrent_runner.py'): (runtime/n).write_text('# R1 frozen fixture\n')
        bound = c.sources(runtime)
        c.write(prior/'admission.json',dict(source_bindings=bound))
        c.write(prior/'handoff.json',dict(new_source_bindings=bound))
        reg = dict(runtime_source_bindings=bound,original_metric_source_bindings=f.oldbound,
            original_registration_sha256=c.sha(f.metrics/'supervisor_registration.json'),
            admission_sha256=c.sha(prior/'admission.json'),handoff_sha256=c.sha(prior/'handoff.json'))
        c.write(prior/'runtime_registration.json',reg); rsha = c.sha(prior/'runtime_registration.json')
        launch = dict(pid=100,start_ticks='10',source_bindings=bound,runtime_registration_sha256=rsha)
        status = dict(pid=100,status='WAITING_FOR_CONCURRENT_ADMISSION',worker_pid=None,
            reason='M2_SLOWDOWN_OVER_20_PERCENT_THREE_INTERVALS')
        worker = dict(pid=444,start_ticks='44',owner=dict(pid=100,start_ticks='10'),source_bindings=bound,runtime_registration_sha256=rsha)
        for name,value in (('controller_launch.json',launch),('controller_status.json',status),('partial_launch.json',worker)):
            c.write(prior/name,value)
        c.write(f.metrics/'supervisor_launch.json',dict(launch,source_bindings=f.oldbound,runtime_source_bindings=bound))
        c.write(f.metrics/'supervisor_status.json',status)
        hand = c.read(f.out/'handoff.json'); hand['oldlaunch_sha256'] = c.sha(f.metrics/'supervisor_launch.json')
        hand['previous_extension'] = dict(out=str(prior),bindings={str(p):c.sha(p) for p in prior.glob('*.json')})
        c.write(f.out/'handoff.json',hand); f.ps.pop(100)
        return prior

    def test_checked_failed_r1_owner_with_none_status_worker_can_be_replaced(self):
        prior = self.previous_fixture(); f = self.f
        f.controller.initialize(); f.controller.takeover()
        self.assertEqual(f.retired,[])
        self.assertTrue((f.out/'retired_metrics_supervisor/previous_extension_partial_launch.json').exists())
        self.assertEqual(c.read(f.metrics/'supervisor_launch.json')['pid'],300)

    def test_failed_r1_latest_partial_worker_must_have_exited(self):
        self.previous_fixture(); self.f.ps[444] = dict(pid=444,start_ticks='44',state='R')
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_failed_r1_owner_must_have_exited(self):
        self.previous_fixture(); self.f.ps[100] = dict(pid=100,start_ticks='10',state='R')
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_failed_r1_source_mutation_is_rejected(self):
        prior = self.previous_fixture(); (prior/'runtime/controller.py').write_text('# changed\n')
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_failed_r1_missing_latest_launch_evidence_is_rejected(self):
        prior = self.previous_fixture(); hand = c.read(self.f.out/'handoff.json')
        del hand['previous_extension']['bindings'][str(prior/'partial_launch.json')]; c.write(self.f.out/'handoff.json',hand)
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def add_earlier_ancestry(self,prior):
        f = self.f; earlier = f.root/'outputs/METRIC-CONCURRENT-R2-20261002'; runtime = earlier/'runtime'; runtime.mkdir(parents=True)
        for n in ('controller.py','concurrent_runner.py'): (runtime/n).write_text('# frozen earlier R1 fixture\n')
        bound = c.sources(runtime)
        c.write(earlier/'admission.json',dict(source_bindings=bound)); c.write(earlier/'handoff.json',dict(new_source_bindings=bound))
        reg = dict(c.read(prior/'runtime_registration.json'),runtime_source_bindings=bound,
            admission_sha256=c.sha(earlier/'admission.json'),handoff_sha256=c.sha(earlier/'handoff.json'))
        c.write(earlier/'runtime_registration.json',reg); rsha = c.sha(earlier/'runtime_registration.json')
        c.write(earlier/'controller_launch.json',dict(pid=111,start_ticks='11',source_bindings=bound,runtime_registration_sha256=rsha))
        c.write(earlier/'controller_status.json',dict(pid=111,status='FAILED',worker_pid=None))
        c.write(earlier/'partial_launch.json',dict(pid=445,start_ticks='45',owner=dict(pid=111,start_ticks='11'),
            source_bindings=bound,runtime_registration_sha256=rsha))
        previous = dict(out=str(earlier),bindings={str(p):c.sha(p) for p in earlier.glob('*.json')})
        c.write(prior/'runtime_registration.json',dict(c.read(prior/'runtime_registration.json'),previous_extension=previous))
        rsha = c.sha(prior/'runtime_registration.json')
        for path in (prior/'controller_launch.json',prior/'partial_launch.json',f.metrics/'supervisor_launch.json'):
            c.write(path,dict(c.read(path),runtime_registration_sha256=rsha))
        hand = c.read(f.out/'handoff.json'); hand['previous_extension']['bindings'] = {str(p):c.sha(p) for p in prior.glob('*.json')}
        hand['oldlaunch_sha256'] = c.sha(f.metrics/'supervisor_launch.json'); c.write(f.out/'handoff.json',hand)
        return earlier

    def test_checked_r2_preserves_and_verifies_earlier_r1_ancestry(self):
        prior = self.previous_fixture(); earlier = self.add_earlier_ancestry(prior)
        self.f.controller.initialize(); self.f.controller.takeover()
        self.assertEqual(self.f.retired,[])
        self.assertIn('previous_extension',c.read(prior/'runtime_registration.json'))

    def test_checked_r2_rejects_mutated_earlier_r1_ancestry(self):
        prior = self.previous_fixture(); earlier = self.add_earlier_ancestry(prior)
        (earlier/'runtime/controller.py').write_text('# changed old proof\n')
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_idle_r3_predecessor_must_not_be_relabelled_failed(self):
        prior = self.previous_fixture(); status = c.read(prior/'controller_status.json'); status['status'] = 'FAILED'
        c.write(prior/'controller_status.json',status); c.write(self.f.metrics/'supervisor_status.json',status)
        hand = c.read(self.f.out/'handoff.json'); hand['previous_extension']['bindings'][str(prior/'controller_status.json')] = c.sha(prior/'controller_status.json')
        c.write(self.f.out/'handoff.json',hand)
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_idle_r3_requires_actual_slowdown_pause_reason(self):
        prior = self.previous_fixture(); status = c.read(prior/'controller_status.json'); status['reason'] = 'UNKNOWN_ERROR'
        c.write(prior/'controller_status.json',status); c.write(self.f.metrics/'supervisor_status.json',status)
        hand = c.read(self.f.out/'handoff.json'); hand['previous_extension']['bindings'][str(prior/'controller_status.json')] = c.sha(prior/'controller_status.json')
        c.write(self.f.out/'handoff.json',hand)
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_running_original_metric_owner_rejected(self):
        self.f.ps[100]['state'] = 'R'
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_reused_original_metric_pid_rejected(self):
        self.f.ps[100]['start_ticks'] = '999'
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_original_metric_worker_not_idle_rejected(self):
        c.write(self.f.metrics/'supervisor_status.json',dict(pid=100,status='RUNNING',worker_pid=222))
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_wrong_original_metric_command_rejected(self):
        self.f.ps[100]['command'] = 'unrelated app'
        with self.assertRaises(RuntimeError): self.f.controller.initialize()

    def test_runtime_change_after_launch_rejected(self):
        self.f.controller.initialize(); (self.f.here/'controller.py').write_text('# changed\n')
        with self.assertRaises(RuntimeError): self.f.controller.verify()

    def test_original_source_change_rejected(self):
        self.f.controller.initialize(); (self.f.oldsource/'supervisor.py').write_text('# changed\n')
        with self.assertRaises(RuntimeError): self.f.controller.verify()

    def test_original_registration_change_rejected(self):
        self.f.controller.initialize(); c.write(self.f.metrics/'supervisor_registration.json',dict(changed=True))
        with self.assertRaises(RuntimeError): self.f.controller.verify()

    def test_orphan_previous_metric_worker_prevents_resume(self):
        f = self.f; f.controller.initialize(); f.controller.takeover()
        f.ps.pop(300); f.ps[333] = dict(pid=333,start_ticks='33',state='R')
        c.write(f.metrics/'supervisor_status.json',dict(pid=300,worker_pid=333,worker_start_ticks='33'))
        f.controller.pid = 301; f.ps[301] = dict(pid=301,start_ticks='31',state='R')
        with self.assertRaises(RuntimeError): f.controller.takeover()

    def test_gate_reuse_and_new_region_have_distinct_effective_stages(self):
        f = self.f; path = f.parent/'m2_gate_status.json'
        c.write(path,dict(pid=400,stage='m2_gate',sources=199,time=1))
        self.assertEqual(f.controller.progress()[1]['stage'],'m2_gate_reuse')
        c.write(path,dict(pid=400,stage='m2_gate',sources=200,time=2))
        self.assertEqual(f.controller.progress()[1]['stage'],'m2_gate_new')

    def test_stale_old_worker_progress_not_used(self):
        c.write(self.f.parent/'m2_screen_status.json',dict(pid=999,stage='m2_screen',sources=137,time=1000))
        self.assertIsNone(self.f.controller.progress()[1])

    def test_metric_threads_do_not_change_scientific_process(self):
        env = self.f.controller.environment()
        self.assertEqual(env['OMP_NUM_THREADS'],'6'); self.assertIn('token_channel_efficiency',env['PYTHONPATH'])
        self.assertEqual(self.f.ps[200]['state'],'R')

    def test_slow_stage_suspension_does_not_restart_after_next_poll(self):
        f = self.f; f.controller.initialize(); point = dict(stage='m2_screen',sources=150,time=15000)
        self.assertIn('SLOWDOWN',f.controller.stage_admission_reason(point,False,True))
        self.assertIn('SLOWDOWN',f.controller.stage_admission_reason(point,False,False))

    def test_new_stage_must_finish_fresh_baseline_before_unsuspending(self):
        f = self.f; f.controller.initialize(); f.controller.suspended_stage = 'm2_screen'
        point = dict(stage='m2_gate_new',sources=210,time=30000)
        self.assertIn('SLOWDOWN',f.controller.stage_admission_reason(point,False,False))
        f.controller.slowdown.stage = 'm2_gate_new'
        self.assertIsNone(f.controller.stage_admission_reason(point,False,False))

    def test_parent_done_releases_slowdown_for_serial_one_source_fallback(self):
        f = self.f; f.controller.initialize(); f.controller.suspended_stage = 'm2_screen'
        self.assertIsNone(f.controller.stage_admission_reason(None,True,False))

    def test_cached_startup_below_checkpoint_floor_not_admitted(self):
        f = self.f; f.controller.initialize()
        for n in (0,136,137):
            self.assertIn('REPLAYING',f.controller.stage_admission_reason(dict(stage='m2_screen',sources=n),False,False))
        self.assertIsNone(f.controller.stage_admission_reason(dict(stage='m2_screen',sources=138),False,False))

    def test_gate_first_200_never_admitted_even_if_baseline_exists(self):
        f = self.f; f.controller.initialize(); f.controller.slowdown.stage = 'm2_gate_reuse'
        self.assertIn('REUSES',f.controller.stage_admission_reason(dict(stage='m2_gate_reuse',sources=199),False,False))

    def test_no_active_source_progress_waits(self):
        self.f.controller.initialize()
        self.assertIn('WAITING',self.f.controller.stage_admission_reason(None,False,False))

    def test_exclusive_timing_request_blocks_metrics_before_lease_and_models(self):
        f = self.f; f.controller.initialize(); f.controller.takeover()
        c.write(f.out/'exclusive_timing_requested.json',dict(status='EXCLUSIVE_TIMING_REQUESTED'))
        owner = c.read(f.out/'controller_launch.json')
        own = dict(pid=444,start_ticks='44',ppid=300,state='R')
        with mock.patch.object(guard,'__file__',str(f.here/'scheduling_guard.py')),mock.patch.object(guard.os,'getpid',return_value=444),mock.patch.object(guard,'process',
                side_effect=lambda pid: own if pid == 444 else f.ps.get(pid)),mock.patch.object(guard,'admission_lock',return_value=nullcontext()):
            with self.assertRaises(guard.SchedulingPause): guard.install_admission(f.root,f.out/'admission.json')
        self.assertFalse((f.out/'active_metrics.json').exists())

    def test_unpushed_publication_rejected(self):
        with self.assertRaises(RuntimeError): c.pushed(dict(status='COMMITTED',checks='PASS',commit='a'*40,remote_commit='a'*40))

    def test_memory_and_thermal_headroom(self):
        good = dict(used_mib=10000,total_mib=24576,free_mib=14576,temperature=60,thermal_slowdown=False)
        self.assertIsNone(c.resource_reason(good))
        self.assertEqual(c.resource_reason(dict(good,free_mib=6000)),'COMBINED_GPU_MEMORY_RESERVE')
        self.assertEqual(c.resource_reason(dict(good,temperature=82)),'THERMAL_HEADROOM')
        self.assertEqual(c.resource_reason(dict(good,thermal_slowdown=True)),'THERMAL_HEADROOM')


class SlowdownTests(unittest.TestCase):
    def make(self): return c.Slowdown(dict(stage='m2_screen',seconds_per_source=100,sample_count=20))
    def point(self,i,t,stage='m2_screen'): return dict(stage=stage,sources=i,time=t)

    def test_provisional_seventy_five_percent_three_consecutive_intervals(self):
        s = self.make(); self.assertFalse(s.observe(self.point(0,0),True))
        self.assertFalse(s.observe(self.point(1,176),True)); self.assertFalse(s.observe(self.point(2,352),True))
        self.assertTrue(s.observe(self.point(3,528),True))

    def test_measured_forty_six_percent_slowdown_is_allowed(self):
        s = self.make()
        for i in range(6): self.assertFalse(s.observe(self.point(i,146*i),True))

    def test_exact_limit_allowed(self):
        s = self.make()
        for i in range(6): self.assertFalse(s.observe(self.point(i,175*i),True))

    def test_loading_interval_excluded(self):
        s = self.make(); s.observe(self.point(0,0),False); s.observe(self.point(1,1000),True)
        self.assertEqual(s.bad,0); self.assertEqual(s.intervals,0)

    def test_same_status_poll_cannot_count_as_new_interval(self):
        s = self.make(); s.observe(self.point(0,0),True); p = self.point(1,176)
        s.observe(p,True)
        for _ in range(10): self.assertFalse(s.observe(p,True))
        self.assertEqual(s.bad,1)

    def test_fast_interval_resets_bad_count(self):
        s = self.make(); s.observe(self.point(0,0),True); s.observe(self.point(1,176),True)
        s.observe(self.point(2,276),True); self.assertEqual(s.bad,0)

    def test_new_gate_region_requires_three_exclusive_intervals(self):
        s = self.make()
        for i in range(3): s.observe(self.point(200+i,10*i,'m2_gate_new'),False)
        self.assertEqual(s.stage,'m2_screen')
        s.observe(self.point(203,30,'m2_gate_new'),False)
        self.assertEqual(s.stage,'m2_gate_new'); self.assertEqual(s.seconds,10)

    def test_cached_region_never_sets_new_stage_baseline_while_scoring(self):
        s = self.make()
        for i in range(6): s.observe(self.point(i,i,'m2_gate_reuse'),True)
        self.assertEqual(s.stage,'m2_screen')

    def test_invalid_baseline_rejected(self):
        with self.assertRaises(RuntimeError): c.Slowdown(dict(stage='m2_screen',seconds_per_source=0,sample_count=20))


@unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),
                     'Native Linux pidfd fixture')
class NativeTests(unittest.TestCase):
    def test_identity_bound_retirement_keeps_other_fixture_process_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory)/'supervisor.py'; script.write_text('import time\ntime.sleep(60)\n')
            parent = subprocess.Popen([sys.executable,str(script)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            scientific = subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                expected = c.identity(c.process(parent.pid)); os.kill(parent.pid,signal.SIGSTOP)
                deadline = time.time()+5
                while c.process(parent.pid)['state'] not in ('T','t'):
                    if time.time()>deadline: self.fail('Fixture parent did not stop')
                    time.sleep(.01)
                c.retire(expected); parent.wait(timeout=5)
                self.assertIsNone(scientific.poll())
            finally:
                for p in (parent,scientific):
                    if p.poll() is None: p.kill()
                    p.wait(timeout=5)


if __name__ == '__main__': unittest.main()
