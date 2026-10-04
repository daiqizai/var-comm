"""CPU checks of source ownership, exact parity and engineering mode selection."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import multiworker_owner as m


def done(indices, start=100., end=110.):
    return dict(status='M1_SOURCE_WORKER_COMPLETE', rows_sha256_by_source={str(i): 'hash%d' % i for i in indices},
                compute_started_unix=start, compute_finished_unix=end, peak_cuda_reserved_bytes=1234)


def trial(count, rate, startup=10., status='EXACT_FULL_ROWS_PARITY_PASS'):
    return dict(workers=count, sources_per_second=rate, startup_and_exit_seconds=startup, status=status)


class OwnershipTests(unittest.TestCase):
    def test_source_partitions_are_exclusive_and_complete(self):
        for count in (1, 2, 4):
            groups = m.partition(list(range(21, 1000)), count)
            values = [i for group in groups for i in group]
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual(sorted(values), list(range(21, 1000)))

    def test_partition_rejects_unsupported_or_duplicate_sources(self):
        for indices, workers in (([0, 0], 2), ([1000], 2), ([True], 2), ([0, 1], 3), ([], 2)):
            with self.assertRaises(RuntimeError):
                m.partition(indices, workers)

    def test_small_remaining_tail_does_not_launch_empty_workers(self):
        self.assertEqual(m.partition([998, 999], 4), [[998], [999]])

    def test_checkpoints_require_original_binding_payload_hash_and_full_grid(self):
        rows = [dict(N=2048, source_index=0, phy_family='QPSK', snr_db=1, action_id='a%d' % i, noise_seed=4101)
                for i in range(2070)]
        cp = dict(binding='original', source_index=0, rows=rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.json'
            m.write(path, dict(cp, payload_sha256=m.identity(cp)))
            self.assertEqual(m.checkpoint_rows(path, 'original', 0), m.identity(rows))
            with self.assertRaises(RuntimeError):
                m.checkpoint_rows(path, 'changed', 0)
            changed = dict(cp, rows=rows[:-1])
            m.write(path, dict(changed, payload_sha256=m.identity(changed)))
            with self.assertRaises(RuntimeError):
                m.checkpoint_rows(path, 'original', 0)

    def test_parity_uses_complete_six_source_row_digests(self):
        refs = {i: 'hash%d' % i for i in range(6)}
        value = m.summarize_trial(2, [done([0, 2, 4]), done([1, 3, 5], 102, 111)], refs, 20, [])
        self.assertEqual(value['rows'], 12420)
        self.assertEqual(value['steady_seconds'], 11)
        self.assertEqual(value['startup_and_exit_seconds'], 9)
        self.assertAlmostEqual(value['sources_per_second'], 6 / 11)

    def test_parity_rejects_one_changed_or_duplicated_source(self):
        refs = {i: 'hash%d' % i for i in range(6)}
        bad = done([1, 3, 5]); bad['rows_sha256_by_source']['3'] = 'wrong'
        for completion in ([done([0, 2, 4]), bad], [done([0, 2, 4]), done([0, 1, 3, 5])]):
            with self.assertRaises(RuntimeError):
                m.summarize_trial(2, completion, refs, 20, [])

    def test_missing_worker_or_invalid_timing_is_rejected(self):
        refs = {i: 'hash%d' % i for i in range(6)}
        for count, completion, elapsed in ((2, [done(list(range(6)))], 20),
                                           (1, [done(list(range(6)), 100, 100)], 20),
                                           (1, [done(list(range(6)))], 5)):
            with self.assertRaises(RuntimeError):
                m.summarize_trial(count, completion, refs, elapsed, [])

    def test_below_fifteen_percent_improvement_resumes_original(self):
        selected = m.choose_mode([trial(1, 1.), trial(2, 1.149)], 500)
        self.assertEqual(selected['selected_workers'], 1)
        self.assertEqual(selected['reason'], 'resume_original_serial')

    def test_qualified_speedup_selects_fastest_amortized_mode(self):
        selected = m.choose_mode([trial(1, 1.), trial(2, 1.7), trial(4, 2.3, 20)], 500)
        self.assertEqual(selected['selected_workers'], 4)
        self.assertAlmostEqual(selected['speedup'], 2.3)

    def test_extra_startup_can_outweigh_faster_steady_rate(self):
        selected = m.choose_mode([trial(1, 1., 1), trial(2, 1.8, 50)], 10)
        self.assertEqual(selected['selected_workers'], 1)

    def test_failed_parity_cannot_qualify_a_fast_mode(self):
        with self.assertRaises(RuntimeError):
            m.choose_mode([trial(1, 1.), trial(2, 8., status='PARITY_FAILED')], 500)

    def test_original_cli_is_blocked_while_coordinator_holds_original_lock(self):
        owner = m.Coordinator.__new__(m.Coordinator)
        owner.original_lock = object()
        with self.assertRaisesRegex(RuntimeError, 'Release coordinator'):
            owner.original_stage('calibrate')

    def test_cleanup_signals_only_owned_children(self):
        class Child:
            def __init__(self): self.returncode = None; self.terminated = False; self.killed = False
            def poll(self): return self.returncode
            def terminate(self): self.terminated = True; self.returncode = -15
            def kill(self): self.killed = True; self.returncode = -9
        own, external = Child(), Child()
        owner = m.Coordinator.__new__(m.Coordinator); owner.children = [own]
        owner.stop_children()
        self.assertTrue(own.terminated)
        self.assertFalse(external.terminated)
        self.assertFalse(own.killed)

    def test_gpu_rejects_unknown_process_without_signalling_it(self):
        owner = m.Coordinator.__new__(m.Coordinator); owner.config = {}; owner.departed_gpu = {}
        with patch.object(m.subprocess, 'check_output', side_effect=['3000,24000,50,65,Not Active', '999\n']):
            with self.assertRaisesRegex(RuntimeError, 'Unknown GPU process'):
                owner.gpu()

    def test_hardware_failures_block_increasing_concurrency(self):
        owner = m.Coordinator.__new__(m.Coordinator); owner.config = {}; owner.departed_gpu = {}
        for sample, reason in [('3000,24000,50,86,Not Active', 'thermal'),
                               ('3000,24000,50,65,Active', 'thermal'),
                               ('23000,24000,50,65,Not Active', 'memory')]:
            with patch.object(m.subprocess, 'check_output', side_effect=[sample, '']):
                with self.assertRaisesRegex(RuntimeError, reason):
                    owner.gpu()


if __name__ == '__main__':
    unittest.main()
