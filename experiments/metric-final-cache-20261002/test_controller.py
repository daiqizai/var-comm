"""Synthetic CPU orchestration fixtures; the Linux test signals its own child."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('final_cache_controller_tested', Path(__file__).with_name('controller.py'))
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class Lock:
    def close(self): pass


class Publisher:
    def __init__(self, fixture): self.f = fixture
    def process_snapshot(self):
        # Exact R4 publisher schema: /proc parent PID is deliberately absent.
        return {pid: {k: p[k] for k in ('state', 'start_ticks', 'command') if k in p} for pid, p in self.f.ps.items()}
    def concurrent_gate(self, out, runtime, registration, completion, ps, controller_pid, controller_ticks):
        if c.read(completion).get('sources_complete') != 100: raise RuntimeError('Incomplete fixture cache')
        actual = ps.get(self.f.worker['pid'])
        if c.alive(self.f.worker, dict(actual, pid=self.f.worker['pid']) if actual is not None else None): raise RuntimeError('Scorer active')
        if controller_pid != self.f.old['pid'] or controller_ticks != self.f.old['start_ticks']:
            raise RuntimeError('Wrong nominated fixture owner')
        marker = runtime.as_posix() + '/'
        if any(pid != controller_pid and p.get('state') != 'Z' and marker in p.get('command', '').replace('\\', '/')
               for pid, p in ps.items()):
            raise RuntimeError('Unexempted R4 runtime worker remains active')
        return self.f.prior_bound, dict(status='ENGINEERING_SYNTHETIC_CACHE')
    def parent_gate(self, root, ps):
        complete = c.read(self.f.parent / 'completion.json')
        if complete.get('stop') is not True: raise RuntimeError('Scientific completion not stopped')
        c.published(complete['publication'])
        actual = ps.get(self.f.scientist['pid'])
        if c.alive(self.f.scientist, dict(actual, pid=self.f.scientist['pid']) if actual is not None else None): raise RuntimeError('Science still live')


class Fixture:
    def __init__(self, root):
        self.root = Path(root); self.out = self.root / 'outputs' / c.NAME
        self.here = self.out / 'runtime'; self.prior = self.root / 'outputs' / c.PRIOR
        self.metrics = self.root / 'outputs' / c.METRICS; self.parent = self.root / 'outputs' / c.PARENT
        self.original = self.root / 'experiments/unified-metrics-20261002'
        for folder in (self.here, self.prior / 'runtime', self.metrics, self.parent, self.original): folder.mkdir(parents=True)
        for path in (self.here / 'controller.py', self.here / 'final_runner.py', self.original / 'runner.py',
                     self.prior / 'runtime/controller.py', self.prior / 'runtime/publish_extension.py'):
            path.write_text('# ENGINEERING_SYNTHETIC_FIXTURE\n')
        self.bound, self.prior_bound, self.original_bound = c.sources(self.here), c.sources(self.prior / 'runtime'), c.sources(self.original)
        self.old = dict(pid=100, start_ticks='10'); self.worker = dict(pid=101, start_ticks='11')
        self.scientist = dict(pid=200, start_ticks='20'); self.m2owner = dict(pid=201, start_ticks='21')
        c.write(self.metrics / 'supervisor_registration.json', dict(source_bindings=self.original_bound))
        original_sha = c.sha(self.metrics / 'supervisor_registration.json')
        c.write(self.prior / 'runtime_registration.json', dict(status='REGISTERED', runtime_source_bindings=self.prior_bound,
            original_metric_source_bindings=self.original_bound, original_registration_sha256=original_sha))
        rsha = c.sha(self.prior / 'runtime_registration.json')
        launch = dict(**self.old, source_bindings=self.prior_bound, runtime_registration_sha256=rsha)
        c.write(self.prior / 'controller_launch.json', launch)
        c.write(self.metrics / 'supervisor_launch.json', dict(launch, source_bindings=self.original_bound,
            runtime_source_bindings=self.prior_bound))
        idle = dict(status='WAITING_FOR_ORIGINAL_COMPLETION', pid=100, worker_pid=None)
        c.write(self.prior / 'controller_status.json', idle); c.write(self.metrics / 'supervisor_status.json', idle)
        c.write(self.prior / 'concurrent_registration.json', dict(synthetic=True, source_bindings=self.prior_bound))
        c.write(self.prior / 'partial_completion.json', dict(status='SEALED_CACHE_SNAPSHOT_COMPLETE', sources_complete=100))
        c.write(self.prior / 'status.json', dict(status='SEALED_STUDIES_COMPLETE'))
        c.write(self.prior / 'admission.json', dict(status='ENGINEERING_SYNTHETIC_ADMISSION'))
        c.write(self.prior / 'partial_launch.json', dict(**self.worker, source_bindings=self.prior_bound))
        self.speed = self.root / 'experiments/metric-speed-20261002/controller.py'
        self.speed.parent.mkdir(parents=True); self.speed.write_text('# ENGINEERING original controller\n')
        self.set_scientific_profile()
        c.write(self.metrics / 'modelmanifest.json', dict(synthetic=True))
        c.write(self.metrics / 'assets_complete.json', dict(status='ASSETS_READY',
            manifest_sha256=c.sha(self.metrics / 'modelmanifest.json'), environment=sys.executable))
        c.write(self.out / 'handoff.json', dict(new_source_bindings=self.bound, previous_owner=self.old,
            previous_launch_sha256=c.sha(self.prior / 'controller_launch.json'), previous_runtime_registration_sha256=rsha))
        self.ps = {100: dict(**self.old, state='R', command=(self.prior / 'runtime/controller.py').as_posix()),
            200: dict(**self.scientist, state='R', ppid=201, command=' '.join(self.science_command)),
            201: dict(**self.m2owner, state='R', command=' '.join(self.science_owner_command)),
            300: dict(pid=300, start_ticks='30', state='R', command=(self.here / 'controller.py').as_posix())}
        self.retired = []; self.jobs = []
        def retire(expected, reader, recheck, sleep, now):
            recheck(); self.ps[expected['pid']]['state'] = 'T'; recheck()
            self.retired.append(dict(expected)); self.ps.pop(expected['pid'])
        self.controller = c.Controller(self.root, self.here, pid=300, publisher=Publisher(self), reader=self.ps.get,
            lock_fn=lambda p: Lock(), sleep=lambda n: None, retire_fn=retire)

    def set_scientific_profile(self):
        self.science_owner_command = [sys.executable.replace('\\', '/'), '-u',
            (self.prior / 'runtime/m2_controller.py').as_posix(), '--root', self.root.as_posix()]
        self.science_command = [sys.executable.replace('\\', '/'), '-u',
            (self.prior / 'runtime/scheduled_process.py').as_posix(), '--root', self.root.as_posix(),
            '--admission', (self.prior / 'admission.json').as_posix(), '--stage', 'calibration']
        c.write(self.prior / 'admission.json', dict(status='ENGINEERING_SYNTHETIC_ADMISSION', source_bindings=self.prior_bound,
            original_execution_bindings={str(self.speed): c.sha(self.speed)}))
        c.write(self.prior / 'm2_handoff.json', dict(status='ENGINEERING_SYNTHETIC_M2_HANDOFF'))
        launch = dict(**self.m2owner, command=self.science_owner_command, source_bindings=self.prior_bound,
            admission_sha256=c.sha(self.prior / 'admission.json'), handoff_sha256=c.sha(self.prior / 'm2_handoff.json'),
            scientific_controller=str(self.speed), scientific_controller_sha256=c.sha(self.speed))
        c.write(self.prior / 'm2_controller_launch.json', launch)
        c.write(self.parent / 'supervisor_launch.json', dict(launch, scheduling_only=True,
            scheduling_extension_launch=str(self.prior / 'm2_controller_launch.json'),
            scheduling_extension_launch_sha256=c.sha(self.prior / 'm2_controller_launch.json')))
        c.write(self.parent / 'supervisor_status.json', dict(status='RUNNING', pid=self.m2owner['pid'],
            stage='m2_calibration', worker_pid=self.scientist['pid'], worker_start_ticks=self.scientist['start_ticks']))
        c.write(self.prior / 'active_m2.json', dict(status='ACTIVE', role='m2', **self.scientist, owner=self.m2owner,
            admission_sha256=c.sha(self.prior / 'admission.json'), source_bindings=self.prior_bound, exclusive=False))

    def scientific_done(self):
        self.ps.pop(200, None); self.ps.pop(201, None)
        c.write(self.parent / 'completion.json', dict(status='AUTHORIZED_TWO_METHODS_COMPLETE', stop=True,
            publication=dict(status='PUSHED', checks='PASS', commit='a' * 40, remote_commit='a' * 40)))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.f = Fixture(self.temp.name)

    def test_idle_takeover_preserves_scientific_queue_and_original_registration(self):
        f = self.f; paths = [f.parent / 'supervisor_launch.json', f.parent / 'supervisor_status.json',
                            f.metrics / 'supervisor_registration.json', f.prior / 'runtime_registration.json']
        before = [p.read_bytes() for p in paths]
        f.controller.initialize(); f.controller.takeover()
        self.assertEqual(f.retired, [f.old]); self.assertEqual(before, [p.read_bytes() for p in paths])
        self.assertEqual(f.ps[200]['state'], 'R'); self.assertEqual(f.ps[201]['state'], 'R')
        self.assertEqual(c.read(f.metrics / 'supervisor_launch.json')['pid'], 300)
        proof = c.read(f.out / 'r4_retirement.json')['proof_bindings']; c.verify(proof)
        self.assertEqual(len(proof), 8)

    def test_incomplete_cache_and_nonidle_owner_not_ready(self):
        f = self.f; f.controller.initialize()
        c.write(f.prior / 'partial_completion.json', dict(sources_complete=99))
        self.assertFalse(f.controller.idle_ready())
        c.write(f.prior / 'partial_completion.json', dict(sources_complete=True))
        self.assertFalse(f.controller.idle_ready())
        c.write(f.prior / 'partial_completion.json', dict(sources_complete=100))
        c.write(f.prior / 'controller_status.json', dict(status='CONCURRENT_SCORING', pid=100, worker_pid=101))
        self.assertFalse(f.controller.idle_ready())

    def test_failed_owner_is_not_disguised_as_idle(self):
        f = self.f; f.controller.initialize(); c.write(f.prior / 'controller_status.json', dict(status='FAILED'))
        with self.assertRaises(RuntimeError): f.controller.idle_ready()
        self.assertEqual(f.retired, [])

    def test_running_orphan_scorer_blocks_retirement(self):
        f = self.f; f.controller.initialize(); f.ps[101] = dict(**f.worker, state='R', command='scorer')
        with self.assertRaises(RuntimeError): f.controller.idle_proof()
        self.assertEqual(f.retired, [])

    def test_only_two_registered_scientific_peers_pass_strict_runtime_marker_scan(self):
        f = self.f; f.controller.initialize()
        ps = f.controller.publisher.process_snapshot(); self.assertNotIn('ppid', ps[200])
        self.assertEqual(f.controller.scientific_peers(ps), {200, 201})
        f.controller.idle_proof()
        f.ps[444] = dict(pid=444, start_ticks='44', state='R', command=(f.prior / 'runtime/concurrent_runner.py').as_posix())
        with self.assertRaises(RuntimeError): f.controller.idle_proof()

    def test_scientific_source_pid_and_parent_command_tampering_never_exempted(self):
        f = self.f; f.controller.initialize()
        f.ps[200]['ppid'] = 999
        with self.assertRaises(RuntimeError): f.controller.idle_proof()
        f.ps[200]['ppid'] = 201
        lease = c.read(f.prior / 'active_m2.json'); lease['source_bindings'] = f.bound
        c.write(f.prior / 'active_m2.json', lease)
        with self.assertRaises(RuntimeError): f.controller.idle_proof()
        f.set_scientific_profile(); f.ps[201]['command'] += ' --unregistered'
        with self.assertRaises(RuntimeError): f.controller.idle_proof()

    def test_idle_heartbeat_can_change_before_stop_and_final_proof_is_frozen_after_stop(self):
        f = self.f; f.controller.initialize()
        def retire(expected, reader, recheck, sleep, now):
            state = c.read(f.prior / 'controller_status.json'); state['time'] = 101
            c.write(f.prior / 'controller_status.json', state)
            c.write(f.metrics / 'supervisor_status.json', dict(state, time=100))
            recheck()  # Distinct heartbeats retain the same semantic idle state.
            state['time'] = 102; c.write(f.prior / 'controller_status.json', state)
            f.ps[100]['state'] = 'T'; recheck(); f.ps.pop(100)
        f.controller.retire_fn = retire; f.controller.takeover()
        proof = c.read(f.out / 'r4_retirement.json')['proof_bindings']
        self.assertEqual(proof[str(f.prior / 'controller_status.json')], c.sha(f.prior / 'controller_status.json'))
        archived = f.out / 'retired_r4_owner/r4_controller_status.json'
        self.assertEqual(c.read(archived)['time'], 102)

    def test_real_worker_or_final_transition_after_stop_invalidates_idle_proof(self):
        f = self.f; f.controller.initialize()
        def retire(expected, reader, recheck, sleep, now):
            recheck(); f.ps[100]['state'] = 'T'
            c.write(f.prior / 'controller_status.json', dict(status='RUNNING', pid=100, worker_pid=444, stage='publish'))
            try: recheck()
            except RuntimeError:
                f.ps[100]['state'] = 'R'; raise
            self.fail('Final-stage transition was accepted')
        f.controller.retire_fn = retire
        with self.assertRaises(RuntimeError): f.controller.takeover()
        self.assertEqual(f.ps[100]['state'], 'R')
        self.assertFalse((f.out / 'r4_retirement.json').exists())

    def test_owner_pid_reuse_and_wrong_handoff_fail(self):
        f = self.f; hand = c.read(f.out / 'handoff.json'); hand['previous_owner']['start_ticks'] = '999'
        c.write(f.out / 'handoff.json', hand)
        with self.assertRaises(RuntimeError): f.controller.initialize()

    def test_both_new_and_original_sources_are_immutable(self):
        f = self.f; f.controller.initialize()
        (f.here / 'late.py').write_text('# invalid hot edit\n')
        with self.assertRaises(RuntimeError): f.controller.verify()
        (f.here / 'late.py').unlink(); (f.original / 'runner.py').write_text('# invalid original edit\n')
        with self.assertRaises(RuntimeError): f.controller.verify()

    def test_launch_history_matches_live_launch_bytes(self):
        f = self.f; f.controller.initialize(); launch = c.read(f.out / 'controller_launch.json')
        self.assertTrue(Path(launch['history_path']).is_absolute())
        self.assertEqual(Path(launch['history_path']).read_bytes(), (f.out / 'controller_launch.json').read_bytes())

    def test_resume_retired_owner_never_signals_twice(self):
        f = self.f; f.controller.initialize(); f.controller.takeover()
        f.ps.pop(300); f.ps[301] = dict(pid=301, start_ticks='31', state='R', command='new fixture guardian')
        new = c.Controller(f.root, f.here, pid=301, publisher=Publisher(f), reader=f.ps.get,
            lock_fn=lambda p: Lock(), sleep=lambda n: None, retire_fn=lambda *a: self.fail('Second retirement'))
        new.initialize(); new.takeover()
        self.assertEqual(c.read(f.metrics / 'supervisor_launch.json')['pid'], 301)
        self.assertEqual(len(f.retired), 1)

    def test_previous_controller_orphan_final_worker_blocks_resume(self):
        f = self.f; c.write(f.out / 'controller_status.json', dict(status='FAILED', worker_pid=444, worker_start_ticks='44'))
        f.ps[444] = dict(pid=444, start_ticks='44', state='R')
        with self.assertRaises(RuntimeError): f.controller.initialize()

    def test_completion_waits_for_m2_owner_and_actual_scientific_exit(self):
        f = self.f; f.controller.initialize(); f.controller.takeover()
        self.assertFalse(f.controller.parent_ready())
        c.write(f.parent / 'completion.json', dict(status='AUTHORIZED_TWO_METHODS_COMPLETE'))
        self.assertFalse(f.controller.parent_ready())
        f.scientific_done(); self.assertTrue(f.controller.parent_ready())

    def test_unknown_job_failure_is_not_retried(self):
        f = self.f; f.controller.initialize()
        class Child:
            pid = 444
            def wait(self): return 75
        f.ps[444] = dict(pid=444, start_ticks='44', state='R')
        calls = []
        def spawn(*a, **k): calls.append(a); return Child()
        f.controller.spawn = spawn
        with self.assertRaises(RuntimeError): f.controller.job('fixture', f.here / 'final_runner.py')
        self.assertEqual(len(calls), 1)

    def test_final_pipeline_order_and_bound_union(self):
        f = self.f; f.controller.initialize(); f.controller.takeover(); f.scientific_done()
        result = f.root / 'results/unified_metrics_20261002'; result.mkdir(parents=True)
        proof_file = result / 'synthetic_proof.txt'; proof_file.write_text('ENGINEERING_SYNTHETIC')
        proof = {str(proof_file): c.sha(proof_file)}
        def pub(bound): return dict(status='PUSHED', checks='PASS', commit='a' * 40, remote_commit='a' * 40,
            source_bindings=bound, runtime_source_bindings=bound)
        def job(name, script, args=(), cpu=True):
            f.jobs.append(name)
            if name == 'publish_r4_extension': c.write(f.prior / 'extension_source_publication.json', pub(f.prior_bound))
            elif name == 'publish_final_cache_extension': c.write(f.out / 'extension_source_publication.json', pub(f.bound))
            elif name == 'publish_original_metric_source': c.write(f.metrics / 'source_publication.json', pub(f.original_bound))
            elif name == 'assemble_verified_cache_and_replay_uncached':
                self.assertFalse(cpu); self.assertIn(str(f.prior), list(map(str, args)))
                c.write(f.metrics / 'scoring_completion.json', dict(status='COMPLETE', synthetic=False,
                    training_updates=0, policy_selection_updates=0, parity_passed=True,
                    source_bindings=dict(f.original_bound, **f.prior_bound, **f.bound), inputs=proof, outputs=proof))
            elif name == 'original_paired_analysis':
                c.write(result / 'metrics_analysis_completion.json', dict(status='COMPLETE', inputs=proof, outputs=proof))
            elif name == 'original_result_publication':
                c.write(f.metrics / 'completion.json', dict(status='UNIFIED_METRICS_COMPLETE', publication=pub(f.original_bound)))
            else: self.fail('Unexpected stage ' + name)
        f.controller.job = job
        with mock.patch.object(c.subprocess, 'check_output', return_value=''):
            f.controller.final()
        self.assertEqual(f.jobs, ['publish_r4_extension', 'publish_final_cache_extension', 'publish_original_metric_source',
            'assemble_verified_cache_and_replay_uncached', 'original_paired_analysis', 'original_result_publication'])
        self.assertTrue(c.read(f.out / 'controller_completion.json')['stopped_after_authorized_scope'])
        f.jobs.clear(); f.controller.final(); self.assertEqual(f.jobs, [])


class RetirementTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'native Linux pidfd required')
    def test_native_retirement_only_fixture_child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        try:
            actual = c.process(child.pid); expected = c.identity(actual); checks = []
            c.retire_idle(expected, c.process, lambda: checks.append(True))
            child.wait(timeout=5); self.assertEqual(len(checks), 2)
            self.assertNotEqual(child.returncode, 0)
        finally:
            if child.poll() is None: child.terminate(); child.wait(timeout=5)

    def test_stop_recheck_failure_continues_owner_without_term(self):
        ps = dict(pid=100, start_ticks='10', state='R'); calls = []; count = [0]
        stop, cont = getattr(signal, 'SIGSTOP', 19), getattr(signal, 'SIGCONT', 18)
        def check():
            count[0] += 1
            if count[0] == 2: raise RuntimeError('State changed at boundary')
        def send(fd, sig, *a):
            calls.append(sig)
            if sig == stop: ps['state'] = 'T'
            elif sig == cont: ps['state'] = 'R'
        with mock.patch.object(c.os, 'pidfd_open', return_value=99, create=True), \
                mock.patch.object(c.signal, 'pidfd_send_signal', side_effect=send, create=True), mock.patch.object(c.os, 'close'), \
                mock.patch.object(c.signal, 'SIGSTOP', stop, create=True), mock.patch.object(c.signal, 'SIGCONT', cont, create=True):
            with self.assertRaises(RuntimeError): c.retire_idle(c.identity(ps), lambda p: ps, check)
        self.assertEqual(calls, [stop, cont]); self.assertNotIn(signal.SIGTERM, calls)


if __name__ == '__main__': unittest.main()
