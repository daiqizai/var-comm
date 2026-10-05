"""Driver gates and receipt accounting; no physical backend or GPU execution."""
from pathlib import Path
from contextlib import closing
import copy
import hashlib
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE.parent/'phy_codec').exists():
    sys.path.insert(0, str(HERE.parent/'phy_codec'))
import h_payload_driver as d


def shortlist():
    rows = []
    for snr in (13, 19):
        for arm in ('H16-R', 'H16-A', 'H64-R', 'H64-A'):
            rows.append(dict(candidate_id='HWHOLE:'+d.digest(arm), arm=arm, snr_db=snr, target_m=7,
                             K=0, q=4 if arm.startswith('H16') else 6, nominal_rate='1/2', slot=len(rows)))
    return dict(status='H_EXPECTED_PSNR_SHORTLIST_FROZEN', ready_for_real_calibration=True,
                source_ids=[f'source{i}' for i in range(200)], whole_candidates=rows)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.book = shortlist()
        self.ctx = dict(candidates=self.book['whole_candidates'], shortlist=self.book,
            contract={'execution_registration_sha256': 'a'*64}, protocol={}, catalogue={},
            cfg=dict(out=str(self.path/'out'), registration=str(self.path/'registration.json'),
                     cpu_affinities=[[0, 1], [2, 3]], max_worker_seconds=100,
                     stop_file=str(self.path/'STOP'), source_completion=str(self.path/'source/completion.json'),
                     S1_assets_completion=str(self.path/'S1/completion.json')),
            reg=dict(input_bindings={}, source_bindings={}), source=dict(outputs={}, registration_sha256='old'),
            assets=dict(outputs={}))

    def registered_fixture(self):
        p = self.path/'registered'; p.mkdir()
        def save(name, value):
            path = p/name; d.atomic(path, value); return str(path)
        cfg = copy.deepcopy(self.ctx['cfg'])
        cfg.update(root=str(p/'repo'), runtime_dir=str(p/'runtime'), legacy_runtime=str(p/'legacy'),
                   registration=str(p/'execution.json'), workers=2, cpu_affinities=[[0, 1], [2, 3]])
        budget = dict(status='FROZEN', branch='H', total_cap=200000,
                      phase_limits=dict(initial_true200=20000, development=180000))
        cfg['phase_limits'] = budget['phase_limits']
        cfg['budget_registration'] = save('budget.json', budget)
        cfg['budget_registration_sha256'] = d.sha(cfg['budget_registration'])
        cfg['ledger'] = str(p/'old.sqlite'); Path(cfg['ledger']).touch()
        cfg['ledger_module'] = save('ledger_module.py', {})
        cfg['protocol'] = save('protocol.json', dict(status='FROZEN_BEFORE_DATA', schema='H_CODEC_PROTOCOL_V1'))
        cfg['engineering_contract'] = save('contract.json', dict(status='H_PAYLOAD_CPU_ENGINEERING_SEALED'))
        cfg['catalogue'] = save('catalogue.json', {})
        cfg['reference_qualification'] = save('reference_qual.json', {})
        cfg['qualification_completion'] = save('qual.json', dict(status='H_PHY_QUALIFICATION_PASS',
            registration_sha256='source-reg', budget_registration_sha256=cfg['budget_registration_sha256'],
            outputs={cfg['catalogue']: d.sha(cfg['catalogue'])}, source_bindings={}))
        source_outputs, asset_outputs = {}, {}
        for i, sid in enumerate(self.book['source_ids']):
            for folder, outputs in [('source', source_outputs), ('S1/export-assets', asset_outputs)]:
                path = save(f'{folder}/source_checkpoints/{i:04d}.json', dict(source_id=sid, source_index=i, outputs={}))
                outputs[path] = d.sha(path)
        cfg['source_completion'] = save('source/completion.json', dict(status='H_SOURCE200_COMPLETE',
            registration_sha256='source-reg', source_count=200, development_used=False, holdout_used=False, outputs=source_outputs))
        cfg['S1_assets_completion'] = save('S1/export-assets/completion.json', dict(status='S1_EXPORT_ASSETS_COMPLETE',
            registration_sha256='s1-reg', source_count=200, outputs=asset_outputs))
        cfg['S1_completion'] = save('S1/completion.json', dict(status='S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION',
            registration_sha256='s1-reg', outputs={cfg['S1_assets_completion']: d.sha(cfg['S1_assets_completion'])},
            input_bindings={}, source_bindings={}))
        cfg['source200'] = save('source200.json', {'source_ids': self.book['source_ids']})
        cfg['shortlist'] = save('shortlist.json', dict(self.book, registration_sha256='prescreen-reg'))
        basis = {key: dict(path=cfg[name], sha256=d.sha(cfg[name])) for key, name in [
            ('protocol', 'protocol'), ('catalogue', 'catalogue'), ('budget_registration', 'budget_registration'),
            ('source200', 'source200'), ('qualification_completion', 'qualification_completion'),
            ('source_dir_completion', 'source_completion')]}
        cfg['prescreen_completion'] = save('prescreen.json', dict(status='H_PRESCREEN_COMPLETE_FINAL',
            ready_for_real_calibration=True, registration_sha256='prescreen-reg', prescreen_basis=basis,
            outputs={cfg['shortlist']: d.sha(cfg['shortlist'])}, input_bindings={}, source_bindings={}))
        source_bindings = {}
        for name in ('h_payload_driver.py', 'h_payload_cpu.py'):
            path = str(HERE/name); source_bindings[path] = d.sha(path)
        paths = [Path(cfg['runtime_dir'])/n for n in ('h64_catalog.py', 'h64_phy.py', 'h64_backend.py', 'h64_source.py')]
        paths += [Path(cfg['legacy_runtime'])/n for n in ('ldpc_backend.py', 'uep_phy.py', 'uep_common.py')]
        paths += [Path(cfg['root'])/'src/var_comm'/n for n in ('scale_channel.py', 'token_trellis.cpp')]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('test fixture, never executed', encoding='utf8')
            source_bindings[str(path)] = d.sha(path)
        config_path = save('config.json', cfg)
        fields = ('protocol', 'engineering_contract', 'budget_registration', 'ledger_module', 'catalogue',
                  'reference_qualification', 'qualification_completion', 'shortlist', 'prescreen_completion',
                  'source_completion', 'S1_completion', 'S1_assets_completion', 'source200')
        inputs = {cfg[key]: d.sha(cfg[key]) for key in fields}
        inputs[config_path] = d.sha(config_path)
        reg = dict(status='H_EXECUTION_REVISION_REGISTERED', branch='H', allowed_stage_ids=['initial_true200'],
                   phase_limits=cfg['phase_limits'], input_bindings=inputs, source_bindings=source_bindings)
        d.atomic(cfg['registration'], reg)
        return config_path, cfg, reg

    def test_full_original_receipt_chain_and_no_registration_hash_cycle(self):
        path, cfg, reg = self.registered_fixture()
        context = d.load_registered(path)
        self.assertEqual(context['contract']['execution_registration_sha256'], d.sha(cfg['registration']))
        self.assertNotIn('execution_registration_sha256', d.read(cfg['engineering_contract']))
        self.assertEqual(context['budget']['phase_limits'], context['reg']['phase_limits'])
        self.assertEqual(len(context['shortlist']['source_ids']), 200)
        # Change the actual file without rebinding it: must fail before any PHY import.
        Path(cfg['engineering_contract']).write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(RuntimeError, 'Changed bound'):
            d.load_registered(path)

    def test_execution_scope_full_budget_and_source_screening_basis_are_enforced(self):
        path, cfg, reg = self.registered_fixture()
        changed = dict(reg, allowed_stage_ids=['qualification'])
        d.atomic(cfg['registration'], changed)
        with self.assertRaisesRegex(RuntimeError, 'does not authorize'):
            d.load_registered(path)
        changed = dict(reg, phase_limits=dict(reg['phase_limits'], development=179999))
        d.atomic(cfg['registration'], changed)
        with self.assertRaisesRegex(RuntimeError, 'phase limits'):
            d.load_registered(path)
        screen = d.read(cfg['prescreen_completion'])
        screen['prescreen_basis']['source_dir_completion']['sha256'] = '0'*64
        d.atomic(cfg['prescreen_completion'], screen)
        reg['input_bindings'][cfg['prescreen_completion']] = d.sha(cfg['prescreen_completion'])
        d.atomic(cfg['registration'], reg)
        with self.assertRaisesRegex(RuntimeError, 'source_dir_completion'):
            d.load_registered(path)

    def test_exact_disjoint_source_shards_and_whole_scope(self):
        self.assertEqual(set(d.source_indices(0)) | set(d.source_indices(1)), set(range(200)))
        self.assertFalse(set(d.source_indices(0)) & set(d.source_indices(1)))
        self.assertEqual(len(d.check_shortlist(self.book)), 8)
        full = copy.deepcopy(self.book)
        for row in self.book['whole_candidates']:
            extra = dict(row, candidate_id=row['candidate_id']+'other', slot=row['slot']+8)
            full['whole_candidates'].append(extra)
        self.assertEqual(len(d.check_shortlist(full))*200*3*2, 19200)

    def test_provisional_missing_arm_duplicate_slot_and_third_candidate_rejected(self):
        cases = []
        b = copy.deepcopy(self.book); b['ready_for_real_calibration'] = False; cases.append(b)
        b = copy.deepcopy(self.book); b['whole_candidates'].pop(); cases.append(b)
        b = copy.deepcopy(self.book); b['whole_candidates'][1]['slot'] = 0; cases.append(b)
        b = copy.deepcopy(self.book)
        for slot in (8, 9):
            b['whole_candidates'].append(dict(b['whole_candidates'][0], candidate_id=str(slot), slot=slot))
        cases.append(b)
        for b in cases:
            with self.subTest(case=b), self.assertRaises(RuntimeError):
                d.check_shortlist(b)

    def test_receipt_checks_original_output_bytes(self):
        asset = self.path/'asset.json'; asset.write_text('{}', encoding='utf8')
        receipt = self.path/'completion.json'
        d.atomic(receipt, dict(status='COMPLETE', outputs={str(asset): d.sha(asset)}))
        self.assertEqual(d.checked_receipt(receipt, 'COMPLETE')['status'], 'COMPLETE')
        asset.write_text('{"changed":true}', encoding='utf8')
        with self.assertRaisesRegex(RuntimeError, 'Changed bound'):
            d.checked_receipt(receipt, 'COMPLETE')

    def test_collect_source_pairs_all_candidates_three_seeds_without_reselection(self):
        cp = str(self.path/'source/source_checkpoints/0000.json')
        old = str(self.path/'S1/source_checkpoints/0000.json')
        self.ctx['source']['outputs'][cp] = 'b'*64
        self.ctx['assets']['outputs'][old] = 'c'*64
        calls, records = [], []
        class FakeCore:
            @staticmethod
            def load_source_assets(*args, **kwargs):
                return dict(scales='TX_ONLY', arithmetic_bits='TX_BITS_ONLY', source_bindings={})
            @staticmethod
            def run_frame(**kwargs):
                calls.append(kwargs)
                return dict(candidate=kwargs['candidate']['candidate_id'], seed=kwargs['noise_seed'])
        result = d.collect_source(self.ctx, FakeCore, 'backend', 'header', 'existing_ledger', 0, lambda: None, records.append)
        self.assertEqual(len(result['frames']), 24)
        self.assertEqual(result['frames'], records)
        self.assertEqual({(x['candidate']['slot'], x['noise_seed']) for x in calls},
                         {(s, seed) for s in range(8) for seed in (6101, 6102, 6103)})
        self.assertTrue(all(x['ledger'] == 'existing_ledger' and x['source_id'] == 'source0' for x in calls))

    def test_safe_boundary_stops_before_next_fake_frame(self):
        self.ctx['source']['outputs'][str(self.path/'source/source_checkpoints/0000.json')] = 'b'*64
        self.ctx['assets']['outputs'][str(self.path/'S1/source_checkpoints/0000.json')] = 'c'*64
        records = []
        class FakeCore:
            load_source_assets = staticmethod(lambda *a, **kw: dict(scales=[], arithmetic_bits={}, source_bindings={}))
            run_frame = staticmethod(lambda **kw: {'source_id': kw['source_id']})
        def boundary():
            if records:
                raise RuntimeError('STOP')
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            d.collect_source(self.ctx, FakeCore, None, None, None, 0, boundary, records.append)
        self.assertEqual(len(records), 1)

    def database(self, rows, count=None):
        path = self.path/'ledger.sqlite'
        with closing(sqlite3.connect(path)) as c, c:
            c.execute('CREATE TABLE events(event_id,phase,kind,status,result,result_sha)')
            c.execute('CREATE TABLE counters(phase,charged)')
            c.executemany('INSERT INTO events VALUES(?,?,?,?,?,?)', rows)
            c.execute('INSERT INTO counters VALUES(?,?)', (d.PHASE, len(rows) if count is None else count))
        return path

    def ledger_row(self, event, kind='header', status='COMPLETE', value=None):
        value = value or {'header_ok': False}
        return (event, d.PHASE, kind, status, d.canonical(value), d.digest(value))

    def test_actual_ledger_matches_trace_hashes_and_counts(self):
        event = 'H:initial_true200:c:snr13:src0000:seed6101:header'
        row = self.ledger_row(event)
        path = self.database([row])
        expected = {event: dict(kind='header', result_sha256=row[-1])}
        result = d.audit_ledger(path, expected)
        self.assertEqual(result['paid_events'], 1)
        # Re-auditing a completed result is read-only and never increases charges.
        self.assertEqual(d.audit_ledger(path, expected), result)
        with closing(sqlite3.connect(path)) as c, c:
            self.assertEqual(c.execute('SELECT charged FROM counters').fetchone()[0], 1)
            c.execute('UPDATE events SET status="RESERVED"')
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            d.audit_ledger(path, expected)

    def test_ledger_wrong_result_missing_event_or_wrong_counter_cannot_complete(self):
        event = 'H:initial_true200:c:snr13:src0000:seed6101:header'
        row = self.ledger_row(event)
        path = self.database([row], count=2)
        expected = {event: dict(kind='header', result_sha256=row[-1])}
        with self.assertRaisesRegex(RuntimeError, 'counter'):
            d.audit_ledger(path, expected)
        with self.assertRaisesRegex(RuntimeError, 'actual paid'):
            d.audit_ledger(path, {event: dict(kind='header', result_sha256='0'*64)})
        with closing(sqlite3.connect(path)) as c, c:
            c.execute('DELETE FROM events')
        with self.assertRaisesRegex(RuntimeError, 'Missing paid'):
            d.audit_ledger(path, expected)

    def test_worker_only_ignores_other_source_partition_merge_rejects_extra(self):
        a = 'H:initial_true200:c:snr13:src0000:seed6101:header'
        b = 'H:initial_true200:c:snr13:src0001:seed6101:header'
        rows = [self.ledger_row(a), self.ledger_row(b)]
        path = self.database(rows)
        expected = {a: dict(kind='header', result_sha256=rows[0][-1])}
        self.assertEqual(d.audit_ledger(path, expected, 0)['paid_events'], 1)
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            d.audit_ledger(path, expected)

    def test_failure_receipt_and_attempt_preserved_then_reentry_blocked(self):
        cfg = self.ctx['cfg']; out = Path(cfg['out'])/'worker_0'
        core = object()
        with patch.object(d, 'load_registered', return_value=self.ctx), \
             patch.object(d, 'exclusive', lambda p: __import__('contextlib').nullcontext()), \
             patch.object(d, 'configure_cpu'), patch.object(d, 'build_runtime', return_value=(core, None, None)), \
             patch.object(d, 'make_ledger', return_value=types.SimpleNamespace(worker={'pid': 123, 'start_ticks': 1, 'uid': 1002, 'argv': ['fake']})), \
             patch.object(d, 'collect_source', side_effect=RuntimeError('scripted failure after reservation')) as collect:
            with self.assertRaisesRegex(RuntimeError, 'reservation'):
                d.run_worker('config', 0)
            self.assertTrue((out/'attempt.json').is_file())
            failure = (out/'failure.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'Previous failure'):
                d.run_worker('config', 0)
            self.assertEqual((out/'failure.json').read_bytes(), failure)
            self.assertEqual(collect.call_count, 1)

    def test_partial_attempt_without_receipt_is_not_resumed(self):
        out = Path(self.ctx['cfg']['out'])/'worker_0'; out.mkdir(parents=True)
        d.atomic(out/'attempt.json', {'status': 'STARTED'})
        with patch.object(d, 'load_registered', return_value=self.ctx), \
             patch.object(d, 'exclusive', lambda p: __import__('contextlib').nullcontext()), \
             patch.object(d, 'build_runtime') as build:
            with self.assertRaisesRegex(RuntimeError, 'Partial attempt'):
                d.run_worker('config', 0)
            build.assert_not_called()
        self.assertTrue((out/'failure.json').is_file())


if __name__ == '__main__':
    unittest.main()
