"""Transparent fake-backend and temporary-ledger tests only; no LDPC/GPU work."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from h64_catalog import PROTOCOL, all_buckets, catalogue
from h_coarse_driver import (generator as coarse_generator, measure_cell as coarse_measure,
                            public_counter as coarse_counter, summarize_rows, atomic, csv_write, sha)
from h_prescreen import validate_body, apply_refinement, Evidence
from h_refine_driver import (ADDITIONAL, ORIGINAL, generator, public_counter,
    measure_cell, merge_cell, audit_ledger, check_packet_proofs, validate_requests, event_id, read_packets)
from test_h64_core import FakeBackend, RecordingLedger
from test_h_prescreen import fixture_catalog, fixture_cell


LEDGER_PATH = HERE.parent/'runner'/'budget_ledger.py'
if not LEDGER_PATH.exists():
    LEDGER_PATH = HERE/'budget_ledger.py'  # Deployed runtime modules share one directory.
spec = importlib.util.spec_from_file_location('h_refine_test_budget', LEDGER_PATH)
ledger_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(ledger_module)


class RefinementTests(unittest.TestCase):
    def bucket(self):
        b = all_buckets()[0]
        b.update(admission='ADMITTED', layout=FakeBackend().plan(b['k'], b['n'], b['q']))
        return b

    def ledger(self, path, capacity=10):
        identity = dict(pid=1, start_ticks=77, uid=1002, argv=['test_only'])
        return ledger_module.BudgetLedger(path, 'test_budget_SHA', 'H',
            dict(coarse=4, refine=capacity, development=2), worker_identity=identity,
            identity_reader=lambda pid: identity)

    def cell(self, b, snr=19):
        return dict(bucket=b, layout_index=0, old=dict(layout_id=b['layout']['layout_id'], snr_db=snr))

    def test_namespaces_indices_counters_disjoint_and_repeatable(self):
        lid = 'a_layout'
        a = generator(PROTOCOL, lid, 13, 2048, 'noise').standard_normal(400)
        np.testing.assert_array_equal(a, generator(PROTOCOL, lid, 13, 2048, 'noise').standard_normal(400))
        self.assertFalse(np.array_equal(a, coarse_generator(PROTOCOL, lid, 13, 0, 'noise').standard_normal(400)))
        for changed in [(lid, 13, 2049, 'noise'), (lid, 19, 2048, 'noise'), (lid, 13, 2048, 'payload')]:
            self.assertFalse(np.array_equal(a, generator(PROTOCOL, *changed).standard_normal(400)))
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            generator(PROTOCOL, lid, 13, 2047, 'noise')
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            generator(PROTOCOL, lid, 16, 2048, 'noise')
        old = {coarse_counter(i, s, j) for i in range(8) for s in (13, 16, 19) for j in range(2048)}
        new = {public_counter(i, s, j) for i in range(8) for s in (13, 19) for j in range(2048, 8192)}
        self.assertEqual(len(new), 8*2*6144)
        self.assertFalse(old & new)

    def test_paid_body_only_and_exact_actual_ledger_proofs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ledger.sqlite'; ledger = self.ledger(path); b = self.bucket()
            outer = self
            class PrechargeChecked(FakeBackend):
                def decode(self, *args):
                    outer.assertEqual(ledger.snapshot()['phase_charged']['refine'], self.calls+1)
                    return super().decode(*args)
            backend = PrechargeChecked(); journal = []
            rows, summary = measure_cell(backend, ledger, b, 0, 19, lambda: None, journal.append, blocks=3)
            self.assertEqual(len(journal), 3); self.assertEqual(backend.calls, 3)
            self.assertEqual([r['index'] for r in rows], [2048, 2049, 2050])
            snapshot = ledger.assert_quiescent()
            self.assertEqual(snapshot['phase_charged']['refine'], 3)
            self.assertEqual(snapshot['counts'][0]['kind'], 'body')
            proof, actual = audit_ledger(path, [self.cell(b)], blocks=3)
            self.assertEqual(proof['event_count'], 3); check_packet_proofs(rows, actual)
            broken = copy.deepcopy(rows); broken[0]['correct'] = 1-broken[0]['correct']
            with self.assertRaisesRegex(RuntimeError, 'actual paid result'):
                check_packet_proofs(broken, actual)
            # Reconstructing completed exact evidence invokes no extra callback.
            replay, repeated = measure_cell(backend, ledger, b, 0, 19, lambda: None, blocks=3)
            self.assertEqual(replay, rows); self.assertEqual(repeated, summary); self.assertEqual(backend.calls, 3)

    def test_failure_stays_charged_and_cannot_reenter(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = self.ledger(Path(folder)/'ledger.sqlite'); backend = FakeBackend(); backend.raise_decode = True
            with self.assertRaisesRegex(RuntimeError, 'test decoder failure'):
                measure_cell(backend, ledger, self.bucket(), 0, 19, lambda: None, blocks=2)
            self.assertEqual(ledger.snapshot()['phase_charged']['refine'], 1)
            self.assertEqual(ledger.snapshot()['counts'][0]['status'], 'FAILED')
            backend.raise_decode = False
            with self.assertRaisesRegex(ledger_module.BudgetError, 'no silent retry'):
                measure_cell(backend, ledger, self.bucket(), 0, 19, lambda: None, blocks=2)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(ledger.snapshot()['phase_charged']['refine'], 1)

    def test_stop_before_next_packet_preserves_first_paid_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = self.ledger(Path(folder)/'ledger.sqlite'); backend = FakeBackend(); captured = []
            def boundary():
                if captured:
                    raise RuntimeError('requested STOP')
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                measure_cell(backend, ledger, self.bucket(), 0, 19, boundary, captured.append, blocks=3)
            self.assertEqual(backend.calls, 1); self.assertEqual(len(captured), 1)
            self.assertEqual(ledger.assert_quiescent()['phase_charged']['refine'], 1)

    def test_phase_exhaustion_does_not_borrow_reserved_quota(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = self.ledger(Path(folder)/'ledger.sqlite', capacity=1); backend = FakeBackend()
            with self.assertRaisesRegex(ledger_module.BudgetError, 'budget exhausted'):
                measure_cell(backend, ledger, self.bucket(), 0, 19, lambda: None, blocks=2)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(ledger.snapshot()['phase_remaining']['development'], 2)

    def test_wrong_accepted_is_U_including_parser_invalid(self):
        class WrongAccepted(RecordingLedger):
            def decode_once(self, *args):
                r = super().decode_once(*args)
                r['crc_accepted'] = True; r['parser_accepted'] = False; r['decoded_bits'][20] ^= 1
                return r
        rows, summary = measure_cell(FakeBackend(), WrongAccepted(), self.bucket(), 0, 19, lambda: None, blocks=2)
        self.assertEqual(summary['correct'], 0)
        self.assertEqual(summary['undetected'], 2); self.assertEqual(summary['parser_invalid'], 2)
        self.assertEqual(summary['rejected'], 0)

    def test_merge_preserves_original_samples_counts_and_exact_energy_quantiles(self):
        b = self.bucket()
        old_rows, old = coarse_measure(FakeBackend(), RecordingLedger(), b, 0, 19, lambda: None, blocks=2)
        new_rows, added = measure_cell(FakeBackend(), RecordingLedger(), b, 0, 19, lambda: None, blocks=3)
        original = copy.deepcopy(old_rows)
        merged = merge_cell(old_rows, new_rows, old, added, original_count=2, additional_count=3)
        self.assertEqual(old_rows, original); self.assertEqual(merged['trials'], 5)
        for key in ('correct', 'rejected', 'undetected'):
            self.assertEqual(merged[key], old[key]+added[key])
        energies = [r['body_energy'] for r in old_rows+new_rows]
        self.assertAlmostEqual(merged['body_energy']['p95'], float(np.quantile(energies, .95)))
        self.assertAlmostEqual(merged['body_energy']['mean'], float(np.mean(energies)))
        self.assertEqual(merged['original_counts']['trials'], 2)
        broken = copy.deepcopy(new_rows); broken[0]['event_id'] = old_rows[0]['event_id']
        with self.assertRaisesRegex(RuntimeError, 'sample identity'):
            merge_cell(old_rows, broken, old, added, original_count=2, additional_count=3)
        broken_summary = copy.deepcopy(added); broken_summary['correct'] += 1
        with self.assertRaisesRegex(RuntimeError, 'counts differ'):
            merge_cell(old_rows, new_rows, old, broken_summary, original_count=2, additional_count=3)

    def test_only_frozen_one_or_two_cells_no16_or_replacement(self):
        cat = fixture_catalog()
        coarse = validate_body([fixture_cell(b, s) for b in cat['buckets'] for s in (13, 16, 19)], cat)
        def request(index, snr, rank):
            b = cat['buckets'][index]
            return dict(layout_id=b['layout']['layout_id'], snr_db=snr, q=b['q'], nominal_rate=b['nominal_rate'],
                        refinement_rank=rank, additional_body_calls=6144, original_trials=2048, required_merged_trials=8192)
        req = dict(status='H_REFINEMENT_REQUEST_FROZEN', requests=[request(0, 13, 1), request(7, 19, 2)],
                   requested_body_calls=12288, maximum_additional_body_calls=12288)
        self.assertEqual([x['layout_index'] for x in validate_requests(req, cat, coarse)], [0, 7])
        one = dict(req, requests=req['requests'][:1], requested_body_calls=6144)
        self.assertEqual(len(validate_requests(one, cat, coarse)), 1)
        bad = copy.deepcopy(req); bad['requests'][1] = request(1, 16, 2)
        with self.assertRaisesRegex(RuntimeError, 'SNR'):
            validate_requests(bad, cat, coarse)
        bad = copy.deepcopy(req); bad['requests'][1] = request(0, 13, 2)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            validate_requests(bad, cat, coarse)
        bad = copy.deepcopy(req); bad['requests'][0]['additional_body_calls'] = 8192
        with self.assertRaisesRegex(RuntimeError, 'count'):
            validate_requests(bad, cat, coarse)

    def test_exact8192_merge_matches_frozen_final_prescreen_schema(self):
        b = self.bucket(); lid = b['layout']['layout_id']; snr = 19
        def rows(start, count, namespace):
            result = []
            for j in range(start, start+count):
                correct, undetected = int(j % 7 != 0), int(j % 21 == 0)
                rejected = 1-correct-undetected
                result.append(dict(index=j, event_id=f'HCOARSE:{lid}:{snr}:{j}' if namespace == 'coarse' else event_id(lid, snr, j),
                    public_frame_counter=coarse_counter(0, snr, j) if namespace == 'coarse' else public_counter(0, snr, j),
                    correct=correct, rejected=rejected, undetected=undetected, crc_accepted=correct+undetected,
                    parser_invalid=undetected, body_energy=1912.+(j % 11)))
            return result
        old_rows, new_rows = rows(0, 2048, 'coarse'), rows(2048, 6144, 'refine')
        base = dict(layout_id=lid, layout_index=0, snr_db=snr,
                    **{k:b[k] for k in ('resource_id', 'q', 'nominal_rate', 'k', 'n', 'source_capacity')})
        old, additional = dict(base, **summarize_rows(old_rows)), dict(base, **summarize_rows(new_rows))
        merged = merge_cell(old_rows, new_rows, old, additional)
        self.assertEqual(merged['trials'], 8192)
        self.assertEqual(merged['additional_counts']['trials'], 6144)
        self.assertEqual(merged['original_counts']['trials'], 2048)
        valid = validate_body([merged], {'buckets':[b]}, trials=8192, snrs=(snr,))
        self.assertEqual(valid[lid, snr], merged)
        for field in ('correct', 'rejected', 'undetected'):
            self.assertEqual(merged[field], old[field]+additional[field])

    def test_production_csv_summary_and_completion_shape_roundtrip_to_final_prescreen(self):
        """All packet values are synthetic; real production serializers and parsers.

        One transparent fake decode supplies each phase's complete row schema.
        Replicated test rows have fresh public identities and exact count totals;
        they are never labelled independent measured physical observations.
        """
        b = self.bucket(); lid = b['layout']['layout_id']; snr = 19
        seed_old, old_base = coarse_measure(FakeBackend(), RecordingLedger(), b, 0, snr, lambda: None, blocks=1)
        seed_new, new_base = measure_cell(FakeBackend(), RecordingLedger(), b, 0, snr, lambda: None, blocks=1)
        def expand(template, start, count, old_phase):
            rows = []
            for i in range(start, start+count):
                c, u = int(i % 7 != 0), int(i % 21 == 0)
                rows.append(dict(template, index=i,
                    event_id=f'HCOARSE:{lid}:{snr}:{i}' if old_phase else event_id(lid, snr, i),
                    public_frame_counter=coarse_counter(0, snr, i) if old_phase else public_counter(0, snr, i),
                    correct=c, rejected=1-c-u, undetected=u, crc_accepted=c+u, parser_invalid=u,
                    body_energy=1912.+(i % 11)))
            return rows
        old_rows = expand(seed_old[0], 0, 2048, True)
        added_rows = expand(seed_new[0], 2048, 6144, False)
        old = dict(old_base, **summarize_rows(old_rows))
        additional = dict(new_base, **summarize_rows(added_rows))
        additional['last_index'] = 8191
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            # Exact directory convention emitted by production coarse worker0.
            old_dir = d/'coarse'/'worker_0'/'cells'/'layout0_snr19'
            new_dir = d/'refine'/'worker_0'
            csv_write(old_dir/'packets.csv', old_rows); atomic(old_dir/'summary.json', old)
            csv_write(new_dir/'packets.csv', added_rows); atomic(new_dir/'additional_summary.json', additional)
            old_paths = {str(old_dir/n): sha(old_dir/n) for n in ('packets.csv', 'summary.json')}
            new_paths = {str(new_dir/n): sha(new_dir/n) for n in ('packets.csv', 'additional_summary.json')}
            old_completion = old_dir/'completion.json'
            atomic(old_completion, dict(status='H_COARSE_CELL_COMPLETE', testing_only=True,
                registration_sha256='synthetic_original', budget_registration_sha256='synthetic_budget',
                worker_index=0, layout_index=0, snr_db=19, trials=2048, source_bindings={}, outputs=old_paths))
            loaded_old = json.loads((old_dir/'summary.json').read_text())
            value = merge_cell(read_packets(old_dir/'packets.csv'), read_packets(new_dir/'packets.csv'),
                               loaded_old, json.loads((new_dir/'additional_summary.json').read_text()))
            request_path = d/'prescreen'/'refinement_requests.json'
            atomic(request_path, dict(status='H_REFINEMENT_REQUEST_FROZEN', requests=[dict(layout_id=lid, snr_db=19)]))
            old_prescreen = d/'prescreen'/'completion.json'
            atomic(old_prescreen, dict(status='H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED', testing_only=True,
                 input_bindings={str(old_completion): sha(old_completion), **old_paths},
                 outputs={str(request_path): sha(request_path)}))
            table = d/'refine'/'refined_cells.json'; atomic(table, [value])
            done = d/'refine'/'completion.json'
            atomic(done, dict(status='H_REFINEMENT_COMPLETE', testing_only=True,
                 registration_sha256='synthetic_refine', refinement_requests_sha256=sha(request_path),
                 additional_body_decode_events=6144, header_decode_events=0,
                 outputs={str(table): sha(table), **new_paths}))
            cfg = dict(coarse_prescreen_completion=str(old_prescreen), refinement_requests=str(request_path),
                       refinement_completion=str(done), refined_cells=str(table))
            result = apply_refinement(Evidence(), cfg, {'buckets':[b]}, {(lid, snr): old})
            self.assertEqual(result[lid, snr]['trials'], 8192)
            self.assertEqual(result[lid, snr]['original_counts']['trials'], 2048)
            self.assertEqual(result[lid, snr]['additional_counts']['trials'], 6144)
            self.assertEqual(result[lid, snr]['body_energy']['p95'], value['body_energy']['p95'])
            # Final prescreen follows bound original evidence and catches changed bytes.
            (old_dir/'packets.csv').write_text('changed\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'Changed binding'):
                apply_refinement(Evidence(), cfg, {'buckets':[b]}, {(lid, snr): old})

    def test_merge_ledger_rejects_extra_cells_and_unresolved_events(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ledger.sqlite'; ledger = self.ledger(path); b = self.bucket()
            measure_cell(FakeBackend(), ledger, b, 0, 19, lambda: None, blocks=1)
            audit_ledger(path, [self.cell(b)], blocks=1)
            # Different SNR is real additional work but is outside the frozen request.
            measure_cell(FakeBackend(), ledger, b, 0, 13, lambda: None, blocks=1)
            with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
                audit_ledger(path, [self.cell(b)], blocks=1)
            # Per-worker audit may ignore the other registered worker's cell.
            proof, _ = audit_ledger(path, [self.cell(b), self.cell(b, 13)], selected_index=0, blocks=1)
            self.assertEqual(proof['event_count'], 1)
            with ledger.transaction() as conn:
                conn.execute("UPDATE events SET status='RESERVED' WHERE event_id=?", (event_id(b['layout']['layout_id'], 19, 2048),))
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                audit_ledger(path, [self.cell(b), self.cell(b, 13)], selected_index=0, blocks=1)


if __name__ == '__main__':
    unittest.main()
