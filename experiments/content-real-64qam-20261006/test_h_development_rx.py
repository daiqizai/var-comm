"""Actual wire/CDF fixtures with a fake RGB renderer; no GPU/PHY or real data."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parent / 'phy_codec', HERE.parent / 'payload', HERE.parent / 'full_calibration'):
    sys.path.insert(0, str(path))
import h_development_rx as dev
import h_development_cpu as core
import h_development_driver as driver_api
import h_payload_rx as rx
import h_payload_render_driver as cache_api
import h64_phy as phy
import h64_source as codec
from test_h_payload_rx import ReceiverTests as ReceiverFixture
from test_h64_core import Provider


class DevelopmentRXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ReceiverFixture.setUpClass()

    def setUp(self):
        self.fixture = ReceiverFixture()
        self.fixture.setUp()
        self.target = np.zeros((3, 256, 256), dtype=np.uint8)
        self.source = dict(status='H_DEVELOPMENT_CPU_SOURCE_TRACES', source_id='dev0', source_index=0,
                           registration_sha256='a' * 64, frames=[])
        self.schedule = []
        wire = self.fixture.trace(self.fixture.profile(q=6), phy.raw_payload(self.fixture.scales, 6))
        for slot in range(18):
            role = 'H_WHOLE_SYSTEM' if slot < 8 else 'H_RAW_PARTIAL_SYSTEM' if slot < 10 else 'H_FIXED_M7_ATTRIBUTION'
            candidate = dict(candidate_id=f'synthetic{slot}', arm='H64-R', target_m=7 if slot >= 10 else 6,
                             K=81 if slot in (8, 9) else 0, q=6, nominal_rate='1/2', snr_db=13 if slot % 2 == 0 else 19)
            self.schedule.append(dict(development_slot=slot, phase='development', status='FROZEN_POLICY_READY',
                                      role=role, candidate=candidate))
            for seed in dev.SEEDS:
                frame = copy.deepcopy(wire)
                frame.update(status='H_DEVELOPMENT_CPU_FRAME_TRACE', stage='development', role=role,
                             development_slot=slot, source_id='dev0', source_index=0, candidate_id=candidate['candidate_id'],
                             arm=candidate['arm'], snr_db=candidate['snr_db'], noise_seed=seed,
                             public_frame_counter=core.frame_counter(slot, 0, seed), scrambling_session=core.SESSION,
                             noise_stage='development', event_id=core.frame_event(slot, 0, seed),
                             execution_registration_sha256='a' * 64, total_symbols=1024, frame_normalized=False,
                             arithmetic_source_decode_run=False, neural_metrics_run=False, policy_selection=False,
                             MAIN_PHY_run=False, evaluation_only=dict(header_correct=False,
                                 parsed_wire_matches_transmission=False, accepted_wire_mismatch=True))
                self.source['frames'].append(frame)

    def render(self, source=None, target=None, schedule=None, boundary=lambda: None):
        target = self.target if target is None else target
        return dev.render_source(self.source if source is None else source, target,
            hashlib.sha256(target.tobytes()).hexdigest(), self.schedule if schedule is None else schedule,
            core, self.fixture.receiver, rx, cache_api, {'synthetic': 'same-frozen-model'}, boundary)

    @staticmethod
    def replace_wire(frame, wire):
        frame['rx'] = copy.deepcopy(wire['rx'])
        frame['rx_profile'] = copy.deepcopy(wire['rx_profile'])

    def test_complete54_actual_rx_cache_is_shared_across_methods_and_noise(self):
        rows, images, cost = self.render()
        self.assertEqual(len(rows), 54)
        self.assertEqual((cost['receiver_cache_misses'], cost['receiver_cache_hits']), (1, 53))
        self.assertEqual(len(images), 1)
        self.assertEqual(len(self.fixture.rendered), 1)
        self.assertTrue(all(r['received_m'] == 6 and r['received_K'] == 0 for r in rows))
        self.assertTrue(all(r['rx_summary']['truth_correction'] is False for r in rows))
        self.assertFalse(cost['MAIN_rendered'])

    def test_tx_truth_and_target_do_not_affect_reconstruction(self):
        first, first_images, _ = self.render()
        source = copy.deepcopy(self.source)
        for frame in source['frames']:
            frame['tx'] = {'profile_id': 4095, 'payload': [1, 0], 'actual_m': 9}
            frame['received_raw_tokens'] = [1] * 680
            frame['gray'] = True
            frame['evaluation_only'] = dict(header_correct=True, parsed_wire_matches_transmission=True, accepted_wire_mismatch=False)
        second, second_images, _ = self.render(source, np.full_like(self.target, 255))
        self.assertEqual([r['image_sha256'] for r in first], [r['image_sha256'] for r in second])
        self.assertNotEqual(first[0]['psnr_db'], second[0]['psnr_db'])
        for key in first_images:
            np.testing.assert_array_equal(first_images[key], second_images[key])

    def test_accepted_wrong_paid_header_controls_source_state(self):
        source = copy.deepcopy(self.source)
        profile = self.fixture.profile(m=7, K=81)
        self.replace_wire(source['frames'][0], self.fixture.trace(profile, phy.raw_payload(self.fixture.scales, 7, 81)))
        rows, _, cost = self.render(source)
        self.assertEqual((rows[0]['target_m'], rows[0]['received_m'], rows[0]['received_K']), (6, 7, 81))
        self.assertTrue(rows[0]['evaluation_only']['reconstructed_after_accepted_wire_mismatch'])
        self.assertEqual(cost['receiver_cache_misses'], 2)
        source['frames'][0]['rx_profile']['K'] = 80
        with self.assertRaisesRegex(rx.TraceIntegrityError, 'differs from actual'):
            self.render(source)

    def test_crc_drop_and_cache_revalidation(self):
        source = copy.deepcopy(self.source)
        profile = self.fixture.profile(q=6)
        bad = phy.pack_body(phy.raw_payload(self.fixture.scales, 6), profile)
        bad[-1] ^= 1
        self.replace_wire(source['frames'][3], self.fixture.trace(profile, decoded=bad))
        rows, images, cost = self.render(source)
        self.assertEqual(cost['gray_frames'], 1)
        self.assertEqual(rows[3]['source_status'], 'WIRE_REJECT_GRAY')
        np.testing.assert_array_equal(images[rows[3]['image_key']], np.full((3, 256, 256), .5, np.float32))
        source = copy.deepcopy(self.source)
        source['frames'][3]['rx']['body']['decoded_bits'][-1] ^= 1
        with self.assertRaisesRegex(rx.TraceIntegrityError, 'parse differs'):
            self.render(source)

    def test_arithmetic_canonical_wrong_accept_and_software_failure(self):
        source = copy.deepcopy(self.source)
        encoded = codec.encode_prefixes(self.fixture.scales, lambda: Provider([]), self.fixture.primitives, (6,))[6]['bits']
        wrong = encoded.copy()
        wrong[0] ^= 1
        wire = self.fixture.trace(self.fixture.profile('arithmetic'), wrong)
        for frame in source['frames']:
            self.replace_wire(frame, wire)
        rows, _, cost = self.render(source)
        self.assertEqual(cost['arithmetic_canonical_calls'], 1)
        self.assertTrue(all(r['source_status'] == 'ARITHMETIC_SOURCE_DECODED' for r in rows))
        self.assertNotEqual(int(self.fixture.rendered[0][0][0]), int(self.fixture.scales[0][0]))
        malformed = self.fixture.trace(self.fixture.profile('arithmetic'), np.append(encoded, 0).astype(np.uint8))
        self.replace_wire(source['frames'][0], malformed)
        rows, _, _ = self.render(source)
        self.assertEqual(rows[0]['source_status'], 'ARITHMETIC_SOURCE_INVALID_GRAY')
        with patch.object(codec, 'decode_prefix', side_effect=RuntimeError('model/resource failure')):
            with self.assertRaisesRegex(RuntimeError, 'model/resource'):
                self.render(source)

    def test_missing_duplicate_MAIN_oldseed_and_infeasible_are_rejected(self):
        bad = copy.deepcopy(self.source)
        bad['frames'].pop()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            self.render(bad)
        bad = copy.deepcopy(self.source)
        bad['frames'].append(copy.deepcopy(bad['frames'][0]))
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            self.render(bad)
        for key, value in (('development_slot', 18), ('noise_seed', 6101)):
            bad = copy.deepcopy(self.source)
            bad['frames'][0][key] = value
            with self.assertRaisesRegex(RuntimeError, 'reserved MAIN'):
                self.render(bad)
        schedule = copy.deepcopy(self.schedule)
        schedule[-1]['status'] = 'NOT_FEASIBLE'
        with self.assertRaisesRegex(RuntimeError, 'not executable'):
            self.render(schedule=schedule)

    def test_wrong_public_counter_and_stop_boundary_rejected(self):
        bad = copy.deepcopy(self.source)
        bad['frames'][0]['public_frame_counter'] += 1
        with self.assertRaisesRegex(RuntimeError, 'schedule identity'):
            self.render(bad)
        with self.assertRaisesRegex(RuntimeError, 'source STOP'):
            self.render(boundary=lambda: (_ for _ in ()).throw(RuntimeError('source STOP')))
        self.assertEqual(self.fixture.rendered, [])

    def test_floatRGB_archive_preserves54rows_and_refuses_overwrite(self):
        rows, images, cost = self.render()
        with tempfile.TemporaryDirectory() as temp:
            outputs, size = dev.write_source(temp, 0, 'dev0', rows, images, cost, {}, 'f' * 64, cache_api, rx)
            self.assertEqual(len(outputs), 3)
            self.assertGreater(size, 0)
            cp = dev.read(Path(temp) / 'source_checkpoints/0000.json')
            self.assertEqual(cp['frame_count'], 54)
            self.assertEqual(cp['scored_metrics'], ['mse', 'psnr_db'])
            self.assertFalse(cp['overall_development_complete'])
            self.assertTrue(all('image_archive' not in r for r in rows))
            with np.load(Path(temp) / 'images/0000.npz') as archive:
                self.assertEqual(archive['image_0000'].dtype, np.float32)
                np.testing.assert_array_equal(archive['image_0000'], images['image_0000'])
            with self.assertRaisesRegex(RuntimeError, 'preserved'):
                dev.write_source(temp, 0, 'dev0', rows, images, cost, {}, 'f' * 64, cache_api, rx)
            dev.verify(outputs)

    def test_5400_aggregate_never_includes_MAIN_or_reselection(self):
        rows, _, _ = self.render()
        ids = [f'dev{i}' for i in range(100)]
        complete = [dict(row, source_index=i, source_id=ids[i]) for i in range(100) for row in rows]
        self.assertEqual(dev.validate_complete_rows(complete, ids), dict(source_count=100, frame_count=5400, H_policy_snr_points=18, MAIN_frames=0))
        complete[-1]['development_slot'] = 18
        with self.assertRaisesRegex(RuntimeError, 'MAIN'):
            dev.validate_complete_rows(complete, ids)

    def test_target_loader_reads_only_pixels_and_exact_original_population(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            trace = base / 'worker_0/traces/0000.json'
            cp = base / 'worker_0/source_checkpoints/0000.json'
            assetcp, archive = base / 'asset.json', base / 'asset.npz'
            for path in (trace, cp):
                path.parent.mkdir(parents=True, exist_ok=True)
            source = dict(self.source, source_bindings={})
            trace.write_text(json.dumps(source), encoding='utf-8')
            cp.write_text(json.dumps(dict(status='H_DEVELOPMENT_CPU_SOURCE_COMPLETE', registration_sha256='a' * 64,
                source_id='dev0', source_index=0, frame_count=54, outputs={str(trace): dev.sha(trace)})), encoding='utf-8')
            # Object-dtype truth deliberately cannot be loaded with allow_pickle=False.
            # The adapter succeeds because it only reads the separate pixels member.
            np.savez(archive, pixels=self.target, tokens=np.array([{'not': 'RX input'}], dtype=object))
            psha = hashlib.sha256(self.target.tobytes()).hexdigest()
            assetcp.write_text(json.dumps(dict(source_index=0, source_id='dev0', preprocessing_id=psha,
                original_development_data_binding={'original': 0}, archive=str(archive), outputs={str(archive): dev.sha(archive)})), encoding='utf-8')
            ctx = dict(source_ids=['dev0'], cfg={'out': str(base)}, done={'registration_sha256': 'a' * 64,
                'outputs': {str(p): dev.sha(p) for p in (trace, cp)}}, population={'preprocessing_ids': [psha], 'data_bindings': [{'original': 0}]},
                context={'manifest': {'records': [{'checkpoint': str(assetcp), 'checkpoint_sha256': dev.sha(assetcp), 'archive': str(archive)}]},
                         'assets': {'outputs': {str(p): dev.sha(p) for p in (assetcp, archive)}}})
            actual, target, targetsha, bindings = dev.load_source(ctx, 0)
            self.assertEqual(actual['source_id'], 'dev0')
            self.assertEqual(targetsha, psha)
            np.testing.assert_array_equal(target, self.target)
            dev.verify(bindings)
            ctx['population']['data_bindings'][0] = {'calibration': 0}
            with self.assertRaisesRegex(RuntimeError, 'original registered'):
                dev.load_source(ctx, 0)


class ClosureGateTests(unittest.TestCase):
    def fixture(self, base):
        paths = {name: str(base / (name + '.json')) for name in ('config', 'owner_config', 'registration', 'launch', 'completion')}
        for path in paths.values():
            Path(path).write_text('{}', encoding='utf-8')
        Path(paths['owner_config']).write_text(json.dumps({'stages': [{'id': 'development', 'resource': 'cpu'}, {'id': 'report', 'resource': 'cpu'}]}), encoding='utf-8')
        cfg = dict(registration=paths['registration'], owner_config=paths['owner_config'], out=str(base), owner_module='frozen-owner',
                   source_completion=paths['launch'], finalized=paths['launch'], budget_registration=paths['launch'], ledger='mock-ledger')
        before = dict(unresolved=0, failed=0, charged=153960, development_remaining=13200, phase_charged={'development': 0, 'whole_calibration': 48000})
        after = dict(before, charged=153962, development_remaining=13198, phase_charged={'development': 2, 'whole_calibration': 48000})
        reg = dict(source_stage_scope='H_DEVELOPMENT_18_POINTS_CPU_ONLY', allowed_stage_ids=['development', 'report'], budget_before=before)
        ids = [f'dev{i}' for i in range(100)]
        schedule = [dict(development_slot=i, phase='development', role='H_WHOLE_SYSTEM' if i < 8 else 'H_RAW_PARTIAL_SYSTEM' if i < 10 else 'H_FIXED_M7_ATTRIBUTION',
                         status='FROZEN_POLICY_READY', candidate={'snr_db': 13}) for i in range(18)]
        outputs = {}
        for i in range(100):
            path = base / f'worker_{i % 2}' / 'traces' / f'{i:04d}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}', encoding='utf-8')
            outputs[str(path)] = dev.sha(path)
            cp = base / f'worker_{i % 2}' / 'source_checkpoints' / f'{i:04d}.json'
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(dict(status='H_DEVELOPMENT_CPU_SOURCE_COMPLETE', source_id=ids[i], source_index=i,
                registration_sha256=dev.sha(paths['registration']), frame_count=54, outputs={str(path): dev.sha(path)})), encoding='utf-8')
            outputs[str(cp)] = dev.sha(cp)
        events = {f'H:development:Hslot00:src0000:seed6201:{k}': {} for k in ('header', 'body')}
        audit = dict(paid_events=2)
        done = dict(status='H_DEVELOPMENT_CPU_RECEIVE_COMPLETE', source_count=100, frame_count=5400, worker_index=None, workers=2,
            declared_policy_snr_points=18, executable_policy_snr_points=18, maximum_H_packet_calls=10800, MAIN_reserved_packet_calls=2400,
            images_scored=False, source_decode_complete=False, arithmetic_source_decode_complete=False, GPU_used=False, development_used=True,
            holdout_used=False, policy_selection=False, MAIN_complete=False, overall_development_complete=False, H_full_delivery_claimed=False,
            visual_stage_started=False, requires_owner_worker_exit_receipt_before_visual_stage=True,
            registration_sha256=dev.sha(paths['registration']), driver_config_sha256=dev.sha(paths['config']),
            source_completion_sha256=dev.sha(cfg['source_completion']), finalized_policies_sha256=dev.sha(cfg['finalized']),
            budget_registration_sha256=dev.sha(cfg['budget_registration']), outputs=outputs, source_bindings={}, input_bindings={},
            source_ids=ids, source_indices=list(range(100)), worker_identities=[{'pid': 1}, {'pid': 2}], ledger_audit=audit,
            budget_before=before, budget_after=after)
        ctx = dict(cfg=cfg, reg=reg, rows=schedule, ids=ids, context={'catalogue': {}}, args={'development_registration': {}}, core=core)
        closure = dict(done=done, children=done['worker_identities'], bindings={paths['completion']: dev.sha(paths['completion'])})
        driver = SimpleNamespace(load_registered=lambda _: ctx, closed_batch=lambda *args: closure,
            trace_inventory=lambda *args: (events, {'source_count': 100, 'frame_count': 5400}), audit_ledger=lambda *args: audit,
            budget_admission=driver_api.budget_admission, PHASES=driver_api.PHASES)
        owner = SimpleNamespace(raw_process_state=None, same_identity=lambda a, b: a == b, budget_snapshot=lambda *a, **kw: after)
        wait = SimpleNamespace(exited=lambda *args: True)
        return paths, driver, owner, wait, ctx, done, after

    def test_normal_devaware_closure_and_precise_ledger_required(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, driver, owner, wait, ctx, done, after = self.fixture(Path(temp))
            result = dev.verify_cpu_closed(spec, driver, owner, wait)
            self.assertEqual(result['before']['development_remaining'], 13198)
            self.assertEqual(len(result['source_ids']), 100)
            done['budget_after'] = dict(after, charged=after['charged'] + 1)
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                dev.verify_cpu_closed(spec, driver, owner, wait)

    def test_unreaped_worker_and_MAIN_scope_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, driver, owner, wait, ctx, done, after = self.fixture(Path(temp))
            wait.exited = lambda *args: False
            with self.assertRaisesRegex(RuntimeError, 'unclosed'):
                dev.verify_cpu_closed(spec, driver, owner, wait)
            wait.exited = lambda *args: True
            ctx['reg']['source_stage_scope'] = 'H_AND_MAIN'
            with self.assertRaisesRegex(RuntimeError, 'scope'):
                dev.verify_cpu_closed(spec, driver, owner, wait)

    def test_missing_checkpoint_binding_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, driver, owner, wait, ctx, done, after = self.fixture(Path(temp))
            cp = str(Path(temp) / 'worker_1/source_checkpoints/0099.json')
            del done['outputs'][cp]
            with self.assertRaisesRegex(RuntimeError, 'before model loading'):
                dev.verify_cpu_closed(spec, driver, owner, wait)


if __name__ == '__main__':
    unittest.main()
