import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import main_raw64_calibration_checkpoints as c


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(c.canonical(value), encoding='utf-8')
    return str(path)


def profiles():
    return [dict(candidate_id=f'{i+1:064x}', wire_key=f'{i+1000:064x}', profile_id=i, K=0 if i < 3 else 1, groups=[dict(symbols=400, phy_key=f'{i+2000:064x}')]) for i in range(433)]


def shortlist(pool):
    rows = []
    for snr in c.SNRS:
        w = [p['candidate_id'] for p in pool[:3]]
        p = [w[0], pool[3]['candidate_id'], pool[4]['candidate_id']]
        rows += [dict(family='WHOLE', snr_db=snr, candidate_ids=w), dict(family='PARTIAL', snr_db=snr, candidate_ids=p, fixed_whole_fallback=w[0])]
    return dict(selection_role='calibration', modulation_coverage_quota=False, cells=rows)


class Catalogue:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows); self.digest = c.digest(rows); self.aliases_digest = c.digest(['aliases', len(rows)])
    def entry(self, pid):
        c.require(type(pid) is int and 0 <= pid < len(self.rows), 'Unknown ID')
        return copy.deepcopy(self.rows[pid])


def present(header, body, catalogue):
    if not header['header_ok']:
        c.require(body is None, 'No body after rejected header')
        state = dict(kind='gray')
    else:
        catalogue.entry(header['profile_id'])
        state = dict(kind='tokens', hard_payload=body['hard_payload'])
    return dict(header_ok=header['header_ok'], receiver_state=state, actual_state_sha256=c.digest(state))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.profiles = profiles(); self.current = Catalogue(self.profiles); self.old = Catalogue(self.profiles[:270])
        self.plan = c.make_plan(self.profiles, shortlist(self.profiles), [f'source{i}' for i in range(1000)], science_sha256='a'*64, shortlist_sha256='b'*64, source_manifest_sha256='c'*64)
        self.point = next(x for x in self.plan['schedule'] if x['snr_db'] == 13 and x['profile_id'] == 0)
        self.record = dict(source_index=7, source_id='source7', tokens_sha256='d'*64, preprocessing_id='e'*64)

    def test_same_stat_rewrite_reads_current_bytes_for_every_file_kind(self):
        for extension in ('', '.json', '.py', '.txt', '.log', '.npz'):
            with self.subTest(extension=extension):
                path = self.root / ('collision' + extension)
                path.write_bytes(b'one')
                unchanged_metadata = path.stat()
                # Reproduce coarse/network filesystem timestamps deterministically.
                # Real open/read/write remain unmocked; only stat repeats its value.
                with patch.object(Path, 'stat', return_value=unchanged_metadata):
                    self.assertEqual(c.file_sha(path), hashlib.sha256(b'one').hexdigest())
                    path.write_bytes(b'two')
                    self.assertEqual(c.file_sha(path), hashlib.sha256(b'two').hexdigest())
                    with self.assertRaisesRegex(ValueError, 'Changed or unbound'):
                        c.verify_files({str(path): hashlib.sha256(b'one').hexdigest()})

    def test_same_stat_repeated_read_does_not_reuse_digest(self):
        path = self.root / 'sealed-looking.npz'
        path.write_bytes(b'one')
        unchanged_metadata = path.stat()
        with patch.object(Path, 'stat', return_value=unchanged_metadata):
            c.file_sha(path)
            with patch.object(Path, 'open', side_effect=OSError('fresh read required')):
                with self.assertRaisesRegex(OSError, 'fresh read required'):
                    c.file_sha(path)

    def test_file_change_during_hash_still_rejected(self):
        from types import SimpleNamespace
        path = self.root / 'changed-during-read.json'
        path.write_bytes(b'one')
        before = path.stat()
        after = SimpleNamespace(st_dev=before.st_dev, st_ino=before.st_ino,
                                st_size=before.st_size, st_mtime_ns=before.st_mtime_ns+1,
                                st_ctime_ns=before.st_ctime_ns)
        with patch.object(Path, 'stat', side_effect=[before, after]):
            with self.assertRaisesRegex(ValueError, 'File changed while hashing'):
                c.file_sha(path)

    def frame(self, seed=4101, *, crc=True, known=True, bodycrc=False, pid=0, current=False):
        wave = np.ones((1024, 2), np.float64); payload = np.zeros(12, np.uint8)
        tx = dict(N=1024, E=2048., header_uses=68, body_uses=400, idle_uses=556, profile_id=0, frame_counter=c.frame_counter(7, 13, seed), waveform_sha256=c.array_sha(wave))
        expected, y = c.prepare_frame(self.record, self.point, seed, payload, wave, tx)
        header = dict(header_crc_ok=crc, header_fields_legal=known, header_ok=crc and known, profile_id=pid if crc and known else None)
        body = dict(profile_id=pid, phy_key=self.profiles[pid]['groups'][0]['phy_key'], hard_payload=[1]*12, decoded_bits=[1]*28, crc_accept=bodycrc) if header['header_ok'] else None
        cat = self.current if current else self.old
        row = dict(expected, status='MAIN_RAW64_KEEP_RECEPTION_COMPLETE' if current else 'MAIN_RAW_KEEP_RECEPTION_COMPLETE', codebook_sha256=cat.digest, aliases_sha256=cat.aliases_digest, source_decode_complete=True, image_reconstruction_complete=False, quality_scored=False, new_metric_calls=0, header=header, body=body, rx_profile_id=header['profile_id'], body_received_float32_sha256=c.array_sha(y[68:468].astype(np.float32)) if body else None, slot=987, family_memberships=['OLD_ROLE'])
        row.update(present(header, body, cat)); row['logical_packet_calls'] = 2 if body else 1
        events = {}; records = []
        for kind, value in [('header', header)]+([('body', body)] if body else []):
            req = dict(schema='MAIN_RAW64_ACTUAL_RX_REQUEST_V1' if current else 'MAIN_ACTUAL_RX_REQUEST_V1',received_sha256=row['received_sha256'], public_frame_counter=row['public_frame_counter'], snr_db=13.0, N=1024, body_session='actual-body', body_group=0, component=kind, codebook_sha256=cat.digest, aliases_sha256=cat.aliases_digest)
            if kind == 'body': req.update(actual_profile_id=pid, actual_wire_key=self.profiles[pid]['wire_key'], header_result_sha256=c.digest(header), body_received_float32_sha256=row['body_received_float32_sha256'])
            key = f'MAIN_CALIBRATION_V1/fake/noise{seed}.{kind}'; phy = 'header' if kind == 'header' else body['phy_key']
            item = dict(event_id=key, kind=kind, phy_key=phy, request_sha256=c.digest(req), result_sha256=c.digest(value))
            records.append(item); events[key] = dict(event_id=key,kind=kind,phy_key=phy,status='COMPLETE', phase='actual_calibration', request=c.canonical(req), result=c.canonical(value),request_sha=c.digest(req),result_sha=c.digest(value))
            if not current: events[key].pop('request')
        row.update(packet_events=records, packet_event_ids=[x['event_id'] for x in records])
        return row, expected, y, events

    def admit(self, row, expected, y, events, *, legacy=True):
        tp = put(self.root/f'trace{row["noise_seed"]}.json', [row]); normal = put(self.root/'normal.json', {'status': 'TEST_NORMAL_CLOSED'})
        evidence = dict(normal_exit_verified=True, paid_results_verified=True, trace_path=tp, normal_receipts=[normal], bindings={p:c.file_sha(p) for p in (tp, normal)})
        if not legacy:
            evidence.pop('normal_exit_verified'); evidence.pop('normal_receipts')
            evidence.update(live_owner_admission_verified=True, admission_receipts=[normal])
        return c.admit_trace(row, expected, self.point, y, catalogue=self.current, present_actual=present, paid_events=events, evidence=evidence, legacy_catalogue=self.old if legacy else None)

    def test_exact_original_counter_and_float64_noise(self):
        self.assertEqual(c.frame_counter(7, 13, 4101), 12021)
        seed = int.from_bytes(hashlib.sha256(b'["MAIN_CALIBRATION_V1",1024,"source7",4101]').digest()[:16], 'little')
        expected = np.random.Generator(np.random.PCG64(seed)).standard_normal((1024, 2))
        self.assertTrue(np.array_equal(c.standard_noise('source7', 4101), expected))
        self.assertEqual(c.standard_noise('source7', 4101).dtype, np.float64)

    def test_full_bounded_schedule_and_no_added_modulation_point(self):
        self.assertEqual((len(self.plan['schedule']), self.plan['frame_count'], self.plan['maximum_new_packet_calls']), (30, 90000, 180000))
        shared = [p for p in self.plan['schedule'] if p['profile_id'] == 0]
        self.assertTrue(all(p['family_memberships'] == ['WHOLE', 'PARTIAL'] for p in shared))
        bad = shortlist(self.profiles); bad['cells'][1]['candidate_ids'].append(self.profiles[5]['candidate_id'])
        with self.assertRaises(ValueError): c.schedule(self.profiles, bad)
        bad = shortlist(self.profiles); bad['modulation_coverage_quota'] = True
        with self.assertRaises(ValueError): c.schedule(self.profiles, bad)

    def test_wrong_known_header_crc_failed_body_keeps_actual_bits(self):
        row, expected, y, events = self.frame(pid=1, bodycrc=False)
        out = self.admit(row, expected, y, events)
        self.assertEqual(out['receiver_state']['hard_payload'], [1]*12)
        self.assertEqual(out['rx_profile_id'], 1)
        self.assertFalse(out['body']['crc_accept'])
        self.assertEqual(out['execution_provenance']['newly_charged_packet_calls'], 0)
        self.assertEqual(out['codebook_sha256'], self.old.digest)
        self.assertEqual(out['effective_catalogue_digest'], self.current.digest)
        self.assertEqual(out['slot'], self.point['slot'])
        self.assertEqual(out['execution_provenance']['original_slot'], 987)

    def test_old_crc_accepted_unknown_rejected_but_crc_failure_allowed(self):
        row, expected, y, events = self.frame(crc=True, known=False)
        with self.assertRaisesRegex(ValueError, 'unknown'): self.admit(row, expected, y, events)
        row, expected, y, events = self.frame(crc=False, known=False)
        out = self.admit(row, expected, y, events)
        self.assertEqual(out['receiver_state'], {'kind': 'gray'})
        self.assertEqual(out['logical_packet_calls'], 1)

    def test_changed_received_id_semantics_or_source_cannot_reuse(self):
        row, expected, y, events = self.frame(pid=1)
        self.current.rows[1]['groups'][0]['symbols'] = 399
        with self.assertRaisesRegex(ValueError, 'semantics'): self.admit(row, expected, y, events)
        self.current = Catalogue(self.profiles); expected['source_tokens_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'identity'): self.admit(row, expected, y, events)

    def test_paid_result_and_actual_waveform_must_match(self):
        row, expected, y, events = self.frame()
        changed = copy.deepcopy(events); e = changed[row['packet_event_ids'][1]]; v = json.loads(e['result']); v['crc_accept'] = True; e['result'] = c.canonical(v)
        with self.assertRaisesRegex(ValueError, 'result'): self.admit(row, expected, y, changed)
        y[0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'waveform'): self.admit(row, expected, y, events)

    def test_new_trace_charges_are_not_historical_reuse(self):
        row, expected, y, events = self.frame(current=True)
        out = self.admit(row, expected, y, events, legacy=False)
        self.assertEqual(out['execution_provenance']['origin'], 'NEW_PAID_RECEIVE')
        self.assertEqual(out['execution_provenance']['newly_charged_packet_calls'], 2)

    def test_real_sqlite_event_writer_to_frame_admission(self):
        from test_main_raw64_calibration_budget import fixture
        from main_raw64_calibration_budget import CalibrationBudget
        _, registration, _ = fixture(self.root/'meter')
        gate = CalibrationBudget.install(registration)
        row, expected, y, events = self.frame(current=True)
        for event_id, value in events.items():
            event = dict(event_id=event_id,phase='actual_calibration',kind=value['kind'],phy_key=value['phy_key'])
            gate.decode_once(event,json.loads(value['request']),lambda v=value:json.loads(v['result']),{'pid':1})
        actual_records = gate.ledger.events('actual_calibration')
        out = self.admit(row, expected, y, actual_records, legacy=False)
        self.assertEqual(out['execution_provenance']['newly_charged_packet_calls'],2)
        self.assertEqual(gate.snapshot()['phase_charged']['actual_calibration'],2)

    def test_point_checkpoint_exact_three_noises_exclusive_and_tamper(self):
        rows = [self.admit(*self.frame(seed)) for seed in c.NOISE]
        pin = c.write_point(self.root/'points', self.plan, 7, self.point, rows, 'a'*64)
        got = c.read_point(pin, self.plan, 7, self.point, 'a'*64)
        self.assertEqual((got['reused_frames'], got['newly_charged_packet_calls']), (3, 0))
        with self.assertRaises(FileExistsError): c.write_point(self.root/'points', self.plan, 7, self.point, rows, 'a'*64)
        with self.assertRaises(ValueError): c.read_point(pin, self.plan, 8, self.point, 'a'*64)
        Path(pin['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash'): c.read_point(pin, self.plan, 7, self.point, 'a'*64)

    def test_visual_reuse_requires_same_float32_identity_and_never_direct(self):
        original, expected, y, events = self.frame()
        cpu = self.admit(original, expected, y, events)
        image = np.full((3, 256, 256), .5, np.float32)
        row = dict(original, image_reconstruction_complete=True, cpu_image_reconstruction_complete=False, synthetic=False, primary_metric=c.PRIMARY, preprocessing_id=self.record['preprocessing_id'], image_sha256=c.array_sha(image), image_archive='/bound/original.npz', image_slot=0, dinov2_vitl14_cosine=.7)
        args = dict(old_visual_identity={'flags':(6,2)}, required_visual_identity={'flags':[6,2]}, old_metric_identity={'model':'L14'}, required_metric_identity={'model':'L14'}, source_record=self.record)
        result = c.visual_reuse(cpu, row, image, **args)
        self.assertEqual(result['render_arm'], 'VAR'); self.assertFalse(result['direct_Dc_qualified'])
        args['required_metric_identity'] = {'model':'changed'}
        with self.assertRaisesRegex(ValueError, 'identity'): c.visual_reuse(cpu, row, image, **args)
        args['required_metric_identity'] = {'model':'L14'}
        with self.assertRaisesRegex(ValueError, 'FLOAT32'): c.visual_reuse(cpu, row, image.astype(np.float64), **args)


if __name__ == '__main__':
    unittest.main()
