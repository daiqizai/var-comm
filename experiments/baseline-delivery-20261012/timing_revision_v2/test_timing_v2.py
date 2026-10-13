"""Synthetic CPU checks only; no weights, PHY, channel or neural calls."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
from timing_audit_v2 import AuditClock, AuditMeter, adapt_case, static_catalogue, wrap_function


class Counter:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def advance(self, amount): self.value += amount


def synthetic_hash(value):
    return 'retained:' + str(value)


class Tests(unittest.TestCase):
    def test_exact_exclusion_and_nested_no_double_count(self):
        timer = Counter(); clock = AuditClock(timer=timer); clock.active = True
        def inner(): timer.advance(3); return 'original-value'
        def outer():
            timer.advance(2)
            return clock.call('inner', inner)
        self.assertEqual(clock.call('outer', outer), 'original-value')
        self.assertEqual(clock.excluded, 5)
        self.assertEqual(clock.now(), 0)
        self.assertEqual(set(clock.events), {'outer'})

    def test_exception_retains_spent_audit_time(self):
        timer = Counter(); clock = AuditClock(timer=timer); clock.active = True
        def bad(): timer.advance(7); raise ValueError('synthetic')
        with self.assertRaises(ValueError): clock.call('bad', bad)
        self.assertEqual(clock.excluded, 7)
        self.assertEqual(clock.depth, 0)

    def test_protocol_work_and_sync_remain_timed(self):
        timer = Counter()
        def sync(): timer.advance(1)
        clock = AuditClock(sync, timer); clock.active = True; AuditMeter.clock = clock
        meter = AuditMeter(sync)
        def endpoint():
            timer.advance(13)  # synthetic CRC/decoder work, never excluded
            clock.call('pure_audit', lambda: timer.advance(5))
            timer.advance(11)
        meter.call('TX_total', endpoint)
        self.assertEqual(meter.wall_seconds['TX_total'], 31)
        self.assertEqual(meter.audit_seconds['TX_total'], 5)
        self.assertEqual(meter.seconds['TX_total'], 26)

    def test_real_value_wrapper_and_inactive_no_exclusion(self):
        module = SimpleNamespace(__name__='fake_audit_module', digest=synthetic_hash)
        clock = AuditClock()
        entry = wrap_function(module, 'digest', clock, 'synthetic only')
        self.assertEqual(module.digest(7), 'retained:7')
        self.assertTrue(entry['output_values_preserved'])
        self.assertEqual(clock.events, {})

    def test_catalogue_mutation_rejected_outside_window(self):
        class Catalogue:
            count = 0; value = 'frozen'
            def verify(self):
                self.count += 1
                assert self.value == 'frozen'
        cat = Catalogue(); clock = AuditClock(); check = static_catalogue(cat, clock)
        check(); before = cat.count
        clock.active = True; cat.verify(); self.assertEqual(cat.count, before)
        with self.assertRaises(AssertionError): check()
        clock.active = False; cat.value = 'mutated'
        with self.assertRaises(AssertionError): check()

    def test_original_case_adapter_preserves_fake_outputs_and_reports_wall(self):
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location('_v2_test_original_endpoints', root/'timing_endpoints.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        clock = AuditClock(); checks = []
        adapted, proof = adapt_case(module, clock, lambda: checks.append('outside'))
        t = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        class Endpoint:
            def __init__(self): self.t = t
            def encode(self, pixels, snr, counter, meter):
                return np.ones((1024,2)), dict(tokens=None, attempts=[])
            def noise(self, wave, source_id, snr): return wave.copy()
            def receive(self, observed, snr, counter, meter, event_id):
                return np.full((3,256,256), .5, dtype=np.float32), dict(status='FAKE', header=dict(header_ok=False), body=None)
        class Ledger:
            state = dict(cases=[])
            def begin(self, case): self.state['cases'].append(case)
            def finish(self, identity, timings): self.completed = (identity, timings)
        ledger = Ledger()
        identity, timings = adapted(Endpoint(), np.zeros((3,256,256),dtype=np.uint8),
            dict(snr_db=4), 'synthetic-source', 0, ledger)
        self.assertEqual(identity['status'], 'FAKE')
        self.assertEqual(checks, ['outside','outside'])
        self.assertEqual(timings['audit_excluded_seconds'], 0)
        self.assertEqual(timings['tx_seconds'], timings['raw_wall_tx_seconds'])
        self.assertTrue(proof['inverse_AST_equal'])
        self.assertFalse(clock.active)

    def test_shared_resource_gate_allows_cpu_analysis_but_blocks_repository_gpu(self):
        import timing_owner_v2 as owner
        reg = dict(GPU_index=0, initial_GPU_utilization_max_percent=25)
        gpu = '0, GPU-fake, H800, 80000, 20000, 0, 0\n'
        apps = 'GPU-fake, 222, python, 5000\n'
        with mock.patch.object(owner, 'RUN', SimpleNamespace(exists=lambda: False)), \
             mock.patch.object(owner, 'read', return_value=reg), \
             mock.patch.object(owner, 'coordinated', return_value=dict(released=True)), \
             mock.patch.object(owner.subprocess, 'check_output', side_effect=[gpu, apps]), \
             mock.patch.object(owner, 'old_snapshot', return_value=dict(gpu_pids=[222], conflicting_repository_python_pids=[111])), \
             mock.patch.object(Path, 'read_bytes', return_value=(str(owner.ROOT/'experiments/entropy-main-common500-20261013/analyze.py')+'\0').encode()):
            result = owner.resource_snapshot(owner.ROOT)
            self.assertTrue(result['admitted'])
            self.assertEqual(result['observed_repository_python_pids'], [111])
        with mock.patch.object(owner, 'RUN', SimpleNamespace(exists=lambda: False)), \
             mock.patch.object(owner, 'read', return_value=reg), \
             mock.patch.object(owner, 'coordinated', return_value=dict(released=True)), \
             mock.patch.object(owner.subprocess, 'check_output', side_effect=[gpu, apps]), \
             mock.patch.object(owner, 'old_snapshot', return_value=dict(gpu_pids=[222], conflicting_repository_python_pids=[222])), \
             mock.patch.object(Path, 'read_bytes', return_value=b'python\0source.py\0'):
            result = owner.resource_snapshot(owner.ROOT)
            self.assertFalse(result['admitted'])
            self.assertEqual(result['conflicting_repository_python_pids'], [222])

    def test_missing_e1_closure_cannot_release_timing(self):
        import after_emain
        with mock.patch.object(Path, 'exists', return_value=False):
            self.assertIsNone(after_emain.ready_e1())

    def test_failed_e1_child_cannot_release_timing(self):
        import after_emain
        def exists(path):
            return path.name in ('scoring_completion.json','actual_wait_scoring.json')
        values = [dict(status='COMPLETE_EMAIN4500_SCORING',logical_frames=4500),
                  dict(actual_child_waited=True,exit_code=1,returncode=1)]
        with mock.patch.object(Path, 'exists', exists), mock.patch.object(after_emain, 'read', side_effect=values):
            with self.assertRaises(AssertionError): after_emain.ready_e1()


if __name__ == '__main__': unittest.main()
