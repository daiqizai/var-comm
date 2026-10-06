import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import main_raw64_cpu_runner as r
from main_raw64_ledger import Ledger, digest


def put(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value), encoding='utf-8')
    return str(p)


def identity(pid=10):
    return dict(pid=pid, start_ticks=50, uid=1002, argv=['python', 'worker'], ppid=1, state='R')


def body():
    return dict(correct=True, rejected=False, undetected=False, crc_accept=True, payload_equal=True, actual_E=2)


def fixture(root):
    old = Path(put(root / 'old.json', {'status': 'original evidence'}))
    plan = {'schema': 'MAIN_RAW64_CPU_PLAN_V1', 'status': 'FROZEN', 'namespace': 'TEST_NEW_NAMESPACE', 'catalogue_digest': 'a' * 64, 'snrs_db': [1, 4, 7, 10, 13, 19], 'source_images_read': False, 'quality_ranked': False, 'all_body_phy_keys': ['wire'], 'header_phy_key': 'header',
        'qualification': {'mode': '4_noiseless_4_awgn60', 'noise_namespace': 'MAIN_RAW64_QUALIFICATION_V1', 'body_cases': 8, 'body_phy_keys': ['wire'], 'reused_body_phy_keys': [], 'header_profile_ids': [1], 'snr_db': 60},
        'proxy': {'random_namespace': 'MAIN_RAW64_PROXY_V1', 'blocks_per_cell': 256, 'cells': [{'phy_key': 'wire', 'snr_db': 19}], 'reused_cells': [{'phy_key': 'wire', 'snr_db': s, 'n_blocks': 8192, 'n_correct': 8000, 'n_reject': 192, 'n_undetected': 0, 'row_sha256': 'c'*64, 'evidence': {'path': str(old), 'sha256': r.sha(old)}} for s in [1, 4, 7, 10, 13]]}, 'phase_caps': {'qualification': 9, 'proxy': 256}}
    plan_path = put(root / 'plan.json', plan)
    reuse = put(root / 'reuse.json', {'status': 'MAIN_RAW64_REUSE_ADMITTED', 'normal_owners_verified': True, 'source_images_read': False, 'plan_sha256': r.sha(plan_path)})
    science = put(root / 'science.json', {'schema': 'MAIN_RAW12_RASTER_KEEP_64QAM_SCIENCE_V1', 'status': 'FROZEN_BEFORE_NEW_PHY_QUALITY_OR_SELECTION', 'scope': {'N': 1024, 'SNRs': [1, 4, 7, 10, 13, 19], 'order': 'raster', 'failure_rule': 'KEEP'}, 'resources': {'proxy_samples_per_missing_cell': 256, 'qualification_per_new_wire_body_calls': 8, 'wall_clock_ceiling_days': 4}})
    adapter = root / 'adapter.py'; adapter.write_text('def create(cfg):\n raise RuntimeError("Test adapter must not execute")\n')
    catalogue = put(root / 'catalogue.json', {'catalogue_digest': plan['catalogue_digest']})
    sources = {str(Path(r.__file__).absolute()): r.sha(r.__file__), str(Path(r.__file__).with_name('main_raw64_ledger.py').absolute()): r.sha(Path(r.__file__).with_name('main_raw64_ledger.py')), str(adapter): r.sha(adapter)}
    cfg = dict(schema=r.SCHEMA, root=str(root), execution_dir=str(root / 'execution'), out=str(root / 'results'), ledger=str(root / 'ledger.sqlite'), stop_file=str(root / 'STOP'), python=sys.executable, deadline_unix=time.time()+3000, max_seconds=300, resources=dict(workers=2, threads=2, nice=15, affinities=[[0, 1], [2, 3]]), source_bindings=sources, input_bindings={p: r.sha(p) for p in [plan_path, reuse, science, catalogue, str(old)]}, science_registration=science, plan=plan_path, reuse_admission=reuse, adapter_module=str(adapter), adapter_config={'catalogue': catalogue}, pythonpath=[str(Path(r.__file__).parent)])
    request = put(root / 'request.json', cfg)
    return cfg, plan, request


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.p = Path(self.tmp.name) / 'new.sqlite'
        self.caps = {'qualification': 2, 'proxy': 1}; self.l = Ledger.create(self.p, 'a' * 64, self.caps)

    def event(self, key='one', phase='qualification'):
        return dict(event_id=key, phase=phase, kind='body', phy_key='wire')

    def test_precharge_exact_reuse_and_changed_request(self):
        calls = []
        def callback():
            calls.append(1); self.assertEqual(self.l.snapshot()['charged'], 1); self.assertEqual(self.l.snapshot()['unresolved'], 1); return body()
        self.assertEqual(self.l.decode_once(self.event(), {'x': 1}, callback, identity()), body())
        self.assertEqual(self.l.decode_once(self.event(), {'x': 1}, callback, identity()), body()); self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(ValueError, 'Changed event'):
            self.l.decode_once(self.event(), {'x': 2}, callback, identity())

    def test_atomic_caps_no_borrow(self):
        errors = []; barrier = threading.Barrier(2)
        def task(i):
            try:
                barrier.wait(); self.l.decode_once(self.event(str(i), 'proxy'), {}, body, identity(i + 20))
            except ValueError as e:
                errors.append(str(e))
        threads = [threading.Thread(target=task, args=(i,)) for i in range(2)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(errors), 1); self.assertEqual(self.l.quiescent()['phase_charged'], {'qualification': 0, 'proxy': 1})
        with self.assertRaisesRegex(ValueError, 'phase'):
            self.l.decode_once(self.event('dev', 'development'), {}, body, identity())

    def test_failure_is_charged_and_permanent(self):
        def fail(): raise RuntimeError('decoder failed')
        with self.assertRaises(RuntimeError): self.l.decode_once(self.event(), {}, fail, identity())
        self.assertEqual(self.l.snapshot()['charged'], 1)
        with self.assertRaisesRegex(ValueError, 'Failed or unresolved'): self.l.decode_once(self.event(), {}, body, identity())
        with self.assertRaisesRegex(ValueError, 'Failed ledger'): self.l.decode_once(self.event('other'), {}, body, identity())
        with self.assertRaises(ValueError): self.l.quiescent()

    def test_reserved_crash_cannot_retry(self):
        with self.l.connect() as db:
            db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', ('crashed', 'qualification', 'body', 'wire', '{}', digest({}), '{}', 'RESERVED', None, None, None, 0))
        with self.assertRaisesRegex(ValueError, 'Failed or unresolved'): self.l.decode_once(self.event('crashed'), {}, body, identity())
        with self.assertRaises(ValueError): self.l.quiescent()

    def test_old_sqlite_and_changed_registration_never_mutate(self):
        p = self.p.parent / 'old.sqlite'
        with closing(sqlite3.connect(p)) as db, db: db.execute('CREATE TABLE H(x)')
        before = p.read_bytes()
        with self.assertRaises(ValueError): Ledger(p, 'a' * 64, self.caps)
        self.assertEqual(p.read_bytes(), before)
        with self.assertRaises(ValueError): Ledger(self.p, 'b' * 64, self.caps)
        with self.assertRaises(FileExistsError): Ledger.create(self.p, 'b' * 64, self.caps)

    def test_corrupt_complete_result_rejected(self):
        self.l.decode_once(self.event(), {}, body, identity())
        with self.l.connect() as db: db.execute("UPDATE events SET result='{}'")
        with self.assertRaisesRegex(ValueError, 'Stored result'): self.l.decode_once(self.event(), {}, body, identity())

    def test_reserved_future_phases_inaccessible_and_counter_audited(self):
        from main_raw64_ledger import RESERVED_PHASES
        full = dict(self.caps, **{p: 10 for p in RESERVED_PHASES})
        ledger = Ledger.create(self.p.parent / 'full.sqlite', 'b'*64, full)
        for p in RESERVED_PHASES:
            with self.assertRaisesRegex(ValueError, 'Unimplemented'): ledger.decode_once(self.event(p, p), {}, body, identity())
        ledger.decode_once(self.event(), {}, body, identity())
        with ledger.connect() as db: db.execute("UPDATE phase_counts SET charged=0 WHERE phase='qualification'")
        with self.assertRaisesRegex(ValueError, 'counters'): ledger.snapshot()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.cfg, self.plan, self.request = fixture(self.root)

    def test_plan_no_dropped_or_duplicate_cell_and_exact_caps(self):
        self.assertEqual(r.validate_plan(self.plan), {'qualification': 9, 'proxy': 256})
        for change in ['drop', 'duplicate', 'cap', 'extra_snr']:
            p = copy.deepcopy(self.plan)
            if change == 'drop': p['proxy']['reused_cells'].pop()
            elif change == 'duplicate': p['proxy']['cells'] *= 2
            elif change == 'cap': p['phase_caps']['proxy'] += 1
            else: p['snrs_db'].append(25)
            with self.assertRaises(ValueError): r.validate_plan(p)

    def test_schedule_builder_preserves_refined_counts_and_whole_row_hash(self):
        science = self.cfg['science_registration']; group = {'phy_key': 'wire'}
        cat = put(self.root / 'builder_catalogue.json', {'catalogue_digest': 'b'*64, 'profiles': [{'groups': [group]}]})
        originals = [dict(phy_key='wire', snr_db=s, kind='body', n_blocks=20000, n_correct=19900, n_reject=99, n_undetected=1) for s in [1,4,7,10,13]]
        originals += [dict(phy_key='header', snr_db=s, kind='header', n_blocks=8192, n_correct=8100, n_reject=92, n_undetected=0) for s in [1,4,7,10,13,19]]
        old = put(self.root / 'builder_old.json', dict(status='REFINEMENT_COMPLETE', synthetic=False, rows=originals)); newer = put(self.root / 'builder_new.json', [])
        schedule = dict(schema='MAIN_RAW64_EXACT_PHY_SCHEDULE_V1', status='PREPARED_FOR_REGISTRATION', science_registration_sha256=r.sha(science), new_scientific_calls_performed=0, image_quality_read=False, qualification_trials='4noiseless +4highSNR60dB per new body; header8 exact paid IDs including invalid4095 to test rejection; no new quality', blocks_per_missing_cell=256, proxy_randomness='new fixed namespace MAIN_RAW64_PROXY_V1; exact',
            reused_cells=[dict(x, reuse_origin='legacy_bler') for x in originals], reuse_proof=[dict(cell=[x['phy_key'],x['snr_db']], origin='legacy_bler', row_sha256=digest(x)) for x in originals],
            qualification=[dict(kind='body', phy_key='wire', group=group, trials=8),dict(kind='header',phy_key='header',trials=8,profile_ids=list(range(8)))], missing_cells=[dict(kind='body',phy_key='wire',snr_db=19,group=group,n_blocks=256)],phase_caps={'qualification':16,'proxy':256},wanted_cell_count=12,reused_cell_count=11,missing_cell_count=1)
        schedulepath = put(self.root / 'builder_schedule.json',schedule)
        mat = dict(schedule=schedulepath,catalogue=cat,science_registration=science,legacy_bler=old,legacy_MAIN_proxy=newer); mat['bindings']={p:r.sha(p) for p in mat.values()}; path=put(self.root/'builder_materials.json',mat)
        result=r.build_plan(path,self.root/'prepared.json'); plan=r.read(result['path'])
        self.assertEqual(result['reused_header_cells'],6); self.assertEqual(plan['proxy']['reused_cells'][0]['n_blocks'],20000)
        qevents=list(r.events(plan,'qualification',r.points(plan,'qualification')[0]));self.assertEqual([x['noiseless'] for x in qevents],[True]*4+[False]*4)
        schedule['reuse_proof'][0]['row_sha256']='a'*64;put(Path(schedulepath),schedule);mat['bindings'][schedulepath]=r.sha(schedulepath);put(Path(path),mat)
        with self.assertRaisesRegex(ValueError,'whole row'):r.build_plan(path,self.root/'forbidden.json')
        self.assertFalse((self.root/'forbidden.json').exists())

    def test_register_real_load_no_decoder_or_ledger(self):
        registration = r.register(self.request); ctx = r.load(registration)
        self.assertEqual(ctx['plan'], self.plan); self.assertFalse(Path(self.cfg['ledger']).exists())
        with self.assertRaises(ValueError): r.register(self.request)
        Path(self.cfg['adapter_module']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Changed'): r.load(registration)

    def test_request_builder_actual_register_consumer(self):
        self.cfg['project_root'] = str(self.root / 'project')
        self.cfg['adapter_config']['legacy_qualification'] = str(self.root / 'old.json')
        self.plan['input_bindings'] = {str(self.root / 'old.json'): r.sha(self.root / 'old.json')}
        put(Path(self.cfg['plan']), self.plan)
        admission = r.read(self.cfg['reuse_admission']); admission['plan_sha256'] = r.sha(self.cfg['plan']); put(Path(self.cfg['reuse_admission']), admission)
        source = Path(self.cfg['project_root'])/'src/var_comm/header.py'; source.parent.mkdir(parents=True); source.write_text('HEADER=True\n')
        materials = dict(request=self.cfg,source_paths=list(self.cfg['source_bindings']),input_paths=[self.cfg['science_registration']])
        p = put(self.root/'request_materials.json',materials)
        out = self.root/'built_request.json';r.build_request(p,out)
        prepared=r.read(out);self.assertIn(str(source),prepared['source_bindings'])
        result=r.load(r.register(out));self.assertEqual(result['plan'],self.plan);self.assertFalse(Path(self.cfg['ledger']).exists())

    def test_full_phase_merge_links_every_paid_event_and_detects_tampering(self):
        ctx = r.load(r.register(self.request)); ledger = Ledger.create(self.cfg['ledger'], ctx['regsha'], ctx['reg']['phase_caps'])
        phase = 'qualification'; out = Path(self.cfg['out']) / phase / 'worker_0'; out.mkdir(parents=True)
        outputs = {}; count = 0
        for j, point in enumerate(r.points(self.plan, phase)):
            rows = []
            for event in r.events(self.plan, phase, point):
                value = body() if event['kind'] == 'body' else dict(header_crc_ok=True, header_fields_legal=True, header_ok=True, profile_id=event['profile_id'])
                request = {'received_sha': 'actual-test-wave', 'index': event['index']}
                ledger.decode_once(event, request, lambda value=value: value, identity())
                rows.append(dict(event=event, request_sha256=digest(request), result_sha256=digest(value), result=value)); count += 1
            p = out / ('point_' + str(j).zfill(6) + '.json'); put(p, dict(point_index=j, point=point, events=rows)); outputs[str(p)] = r.sha(p)
        done = dict(status='MAIN_RAW64_CPU_WORKER_COMPLETE_V1', phase=phase, worker_index=0, worker_identity=identity(), registration_sha256=ctx['regsha'], config_sha256=r.sha(ctx['cfg']['_path']), packet_calls=count, outputs=outputs)
        cp = out / 'completion.json'; put(cp, done)
        log = self.root / 'closed.log'; log.write_text('normal exit\n')
        exitpath = self.root / 'exit.json'; put(exitpath, dict(process_waited=True, exit_code=0, identity=identity(), log=str(log), log_sha256=r.sha(log)))
        completion = r.merge(ctx, phase, [exitpath]); self.assertEqual(r.read(completion)['packet_calls'], 9)
        # Consistently rehashing a fabricated point still cannot evade the paid ledger.
        p = Path(next(iter(outputs))); point = r.read(p); point['events'][0]['result']['actual_E'] = 9; point['events'][0]['result_sha256'] = digest(point['events'][0]['result']); put(p, point)
        done['outputs'][str(p)] = r.sha(p); put(cp, done)
        with self.assertRaisesRegex(ValueError, 'Point/result'): r.merge(ctx, phase, [exitpath])

    def test_known_unknown_header_and_actual_body_validation(self):
        ev = dict(kind='header', phase='qualification', profile_id=4095)
        r.result_valid(ev, dict(header_crc_ok=True, header_fields_legal=False, header_ok=False, profile_id=None))
        with self.assertRaises(ValueError): r.result_valid(ev, dict(header_crc_ok=True, header_fields_legal=False, header_ok=False, profile_id=4095))
        wrong = body(); wrong['correct'] = False
        with self.assertRaises(ValueError): r.result_valid(dict(kind='body', phase='proxy'), wrong)

    def test_exec_identity_capture_waits_for_correct_argv(self):
        class Child:
            pid = 17
            def poll(self): return None
        wrong = identity(17); right = identity(17); wrong['argv'] = ['parent']; right['argv'] = ['expected']; right['uid'] = 1
        with patch.object(r, 'process', side_effect=[wrong, right]), patch.object(r.os, 'getuid', return_value=1, create=True), patch.object(r.time, 'sleep'):
            self.assertEqual(r.capture(Child(), ['expected']), right)

    def test_stop_write_failure_still_waits_before_return(self):
        ctx = r.load(r.register(self.request)); called = []
        class Child:
            pid = 3
            def poll(self): return None
            def wait(self): called.append('wait'); return 1
        with patch.object(r, 'process', return_value=identity()), patch.object(r.subprocess, 'Popen', return_value=Child()), patch.object(r, 'capture', side_effect=RuntimeError('capture')), patch.object(r, 'request_stop', side_effect=OSError('disk full')):
            with self.assertRaises(RuntimeError): r.run_phase(ctx, 'qualification')
        self.assertEqual(called, ['wait']); failure = r.read(Path(self.cfg['execution_dir']) / 'qualification' / 'failure.json')
        self.assertTrue(failure['all_children_waited']); self.assertIn('STOP_WRITE_FAILED', failure['traceback'])
        self.assertTrue(r.read(Path(self.cfg['execution_dir']) / 'qualification' / 'exit_0.json')['process_waited'])

    def test_separate_runtime_import_only_standard_library(self):
        runtime = self.root / 'runtime'; runtime.mkdir(); cwd = self.root / 'elsewhere'; cwd.mkdir()
        for name in ('main_raw64_cpu_runner.py', 'main_raw64_ledger.py'):
            (runtime / name).write_bytes(Path(r.__file__).with_name(name).read_bytes())
        code = 'import sys;sys.path.insert(0,' + repr(str(runtime)) + ');import main_raw64_cpu_runner;assert "torch" not in sys.modules;assert "numpy" not in sys.modules'
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=cwd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_first_exit_write_failure_still_drains_second_child(self):
        ctx = r.load(r.register(self.request)); waited = []
        class Child:
            def __init__(self, pid): self.pid = pid
            def poll(self): return 1
            def wait(self): waited.append(self.pid); return 1
        children = [Child(30), Child(31)]; save = r.save
        def failing_save(path, value):
            if Path(path).name == 'exit_0.json': raise OSError('first exit write failed')
            save(path, value)
        with patch.object(r, 'process', return_value=identity()), patch.object(r.subprocess, 'Popen', side_effect=children), patch.object(r, 'capture', side_effect=[identity(30), identity(31)]), patch.object(r, 'save', side_effect=failing_save):
            with self.assertRaisesRegex(OSError, 'first exit write failed'):
                r.run_phase(ctx, 'proxy')
        self.assertIn(31, waited)
        failure = r.read(Path(self.cfg['execution_dir']) / 'proxy/failure.json')
        self.assertTrue(failure['all_children_waited'])
        self.assertIn('first exit write failed', failure['drain_evidence_errors']['0_evidence'])
        self.assertTrue(r.read(Path(self.cfg['execution_dir']) / 'proxy/exit_1.json')['process_waited'])


if __name__ == '__main__':
    unittest.main()
