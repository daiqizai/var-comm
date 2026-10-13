"""Synthetic CPU checks only; no weights, PHY, channel or neural calls."""
from pathlib import Path
import importlib.util
import sys
import unittest
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


if __name__ == '__main__': unittest.main()
