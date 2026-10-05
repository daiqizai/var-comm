"""Synthetic complete calibration rows only; no images, PHY or GPU execution."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h_payload_select as s


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def make_candidate(arm, snr, slot, m=7, rate='1/2'):
    wire = dict(arm=arm, q=4 if arm.startswith('H16') else 6, nominal_rate=rate, target_m=m, K=0)
    key = s.digest(wire)
    return dict(**wire, candidate_id='HWHOLE:'+key, policy_key=key, snr_db=snr, slot=slot,
                mean_attempted_tx_probability_scales=float(m if arm.endswith('-A') else 0), selection_rank=1)


def make_fixture(extra=None):
    candidates = [make_candidate(arm, snr, slot) for slot, (snr, arm) in
                  enumerate((snr, arm) for snr in s.SNRS for arm in s.ARMS)]
    if extra is not None:
        candidates.append(make_candidate(extra.get('arm', 'H16-R'), extra.get('snr', 13), 8,
                                         extra.get('m', 8), extra.get('rate', '2/3')))
    ids = [f'fixed_source_{i:04d}' for i in range(200)]
    shortlist = dict(status='H_EXPECTED_PSNR_SHORTLIST_FROZEN', ready_for_real_calibration=True,
                     source_ids=ids, whole_candidates=candidates)
    rows = []
    for c in candidates:
        for i, sid in enumerate(ids):
            for seed in s.SEEDS:
                image_hash = s.digest(['image', c['slot'], i, seed])
                receiver_hash = s.digest(['receiver', c['slot'], i, seed])
                summary = dict(status='H_ACTUAL_RX_RECONSTRUCTION_COMPLETE', source_status='RAW_SOURCE_DECODED',
                    gray=False, source_decode_complete=True, new_packet_decodes=0,
                    target_image_used_for_reconstruction=False, truth_correction=False, cached_clean_image_used=False,
                    image_sha256=image_hash, receiver_view_sha256=receiver_hash, received_mode='raw',
                    arithmetic_canonical_attempted=False, arithmetic_canonical=None,
                    canonical_decode_invalid=False, header_accepted=True, body_parser_accepted=True)
                rows.append(dict(**{k:c[k] for k in ('candidate_id', 'slot', 'arm', 'target_m', 'q', 'nominal_rate', 'snr_db')},
                    source_index=i, source_id=sid, noise_seed=seed, mse=.01, psnr_db=20.,
                    source_status='RAW_SOURCE_DECODED', gray=False, image_sha256=image_hash,
                    receiver_view_sha256=receiver_hash, image_archive=f'/test/sealed/{i:04d}.npz',
                    image_key=f'image_{c["slot"]*3+s.SEEDS.index(seed):04d}', rx_summary=summary))
    protocol = dict(schema='H_CODEC_PROTOCOL_V1', status='FROZEN_BEFORE_DATA', main_snrs_db=[13, 19],
        population={'initial_and_full_cal_noise_seeds':[6101, 6102, 6103]}, calibration={'no_development_selection':True})
    interpretation = dict(schema='H_PRESCREEN_INTERPRETATION_V1', status='FROZEN_BEFORE_COARSE_DATA', original_protocol_changed=False)
    return shortlist, rows, protocol, interpretation


def receipt(shortlist_bytes, metric_bytes, rows):
    outputs = {f'/test/sealed/{i:04d}.npz': s.digest(['synthetic_archive', i]) for i in range(200)}
    outputs['/test/frame_metrics.json'] = hashlib.sha256(metric_bytes).hexdigest()
    return dict(status='H_INITIAL_TRUE200_RX_COMPLETE', images_scored=True, source_decode_complete=True,
        source_count=200, new_packet_decodes=0, development_used=False, registration_sha256='a'*64,
        shortlist_sha256=hashlib.sha256(shortlist_bytes).hexdigest(), frame_count=len(rows), outputs=outputs)


def select(shortlist, rows, protocol, interpretation, override=None):
    shortlist_bytes, metrics = encoded(shortlist), encoded(rows)
    r = receipt(shortlist_bytes, metrics, rows)
    if override:
        override(r)
    return s.select_initial(shortlist_bytes, metrics, frame_metrics_path='/test/frame_metrics.json',
                            render_completion=r, protocol=protocol, interpretation=interpretation)


class SelectionTests(unittest.TestCase):
    def test_complete_grid_selects_eight_and_keeps_per_source_evidence(self):
        result = select(*make_fixture())
        self.assertEqual(result['status'], 'H_INITIAL_TRUE200_SELECTION_COMPLETE')
        self.assertEqual(result['selected_count'], 8)
        self.assertEqual(result['measured_frames'], 4800)
        self.assertEqual(len(result['per_source_evidence']), 1600)
        self.assertTrue(all(r['mean_per_source_psnr_db'] == 20. for r in result['all_candidate_summaries']))
        self.assertFalse(result['full1000_calibration_complete'])
        self.assertFalse(result['execution_registration_created'])

    def test_selection_uses_mean_image_psnr_not_psnr_from_mean_mse(self):
        book, rows, protocol, interpretation = make_fixture(extra={})
        for r in rows:
            if r['slot'] == 0:
                psnr = 5. if r['source_index'] < 100 else 35.
            elif r['slot'] == 8:
                psnr = 19.
            else:
                continue
            r['psnr_db'] = psnr; r['mse'] = 10**(-psnr/10)
        result = select(book, rows, protocol, interpretation)
        chosen = next(c for c in result['selected_candidates'] if c['arm'] == 'H16-R' and c['snr_db'] == 13)
        self.assertEqual(chosen['slot'], 0)  # Mean PSNR20 beats19, despite worse mean MSE.
        summaries = {r['slot']:r for r in result['all_candidate_summaries']}
        self.assertGreater(summaries[0]['mean_overall_mse'], summaries[8]['mean_overall_mse'])
        self.assertAlmostEqual(summaries[0]['mean_per_source_psnr_db'], 20.)
        evidence = next(r for r in result['per_source_evidence'] if r['slot'] == 0 and r['source_index'] == 0)
        self.assertEqual(len(evidence['noise_samples']), 3)
        self.assertAlmostEqual(evidence['mean_noise_psnr_db'], 5.)

    def test_ties_follow_cost_then_exact_rate_target_and_lexical_key(self):
        a = make_candidate('H16-A', 13, 0, m=6, rate='5/6')
        b = make_candidate('H16-A', 13, 1, m=7, rate='1/2')
        by_key = {(r['candidate_id'],13):r for r in (a,b)}
        rows = [dict(candidate_id=r['candidate_id'],snr_db=13,mean_per_source_psnr_db=20.) for r in (b,a)]
        self.assertEqual(s.choose_one(rows, by_key)['candidate_id'], a['candidate_id'])
        c = make_candidate('H16-R', 13, 2, m=9, rate='2/3')
        d = make_candidate('H16-R', 13, 3, m=6, rate='3/4')
        by_key = {(r['candidate_id'],13):r for r in (c,d)}
        rows = [dict(candidate_id=c['candidate_id'],snr_db=13,mean_per_source_psnr_db=20.-.5e-12),
                dict(candidate_id=d['candidate_id'],snr_db=13,mean_per_source_psnr_db=20.)]
        self.assertEqual(s.choose_one(rows, by_key)['candidate_id'], c['candidate_id'])
        rows[0]['mean_per_source_psnr_db'] = 20.-2e-12
        self.assertEqual(s.choose_one(rows, by_key)['candidate_id'], d['candidate_id'])

    def test_missing_duplicate_wrong_source_or_development_seed_rejected(self):
        original = make_fixture()
        mutations = [lambda rows: rows.pop(),
                     lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
                     lambda rows: rows[0].update(source_id='not_registered'),
                     lambda rows: rows[0].update(noise_seed=6201)]
        for mutate in mutations:
            book, rows, protocol, interpretation = copy.deepcopy(original)
            mutate(rows)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                select(book, rows, protocol, interpretation)

    def test_frozen_shortlist_missing_family_extra_candidate_or_wrong_cost_rejected(self):
        book = make_fixture()[0]
        incomplete = copy.deepcopy(book); incomplete['whole_candidates'].pop()
        with self.assertRaisesRegex(ValueError, 'family'):
            s.validate_shortlist(incomplete)
        changed = copy.deepcopy(book); changed['whole_candidates'][1]['mean_attempted_tx_probability_scales'] = 1
        with self.assertRaisesRegex(ValueError, 'tie cost'):
            s.validate_shortlist(changed)
        expanded = copy.deepcopy(book)
        expanded['whole_candidates'] += [make_candidate('H16-R', 13, 8, 8), make_candidate('H16-R', 13, 9, 9)]
        with self.assertRaisesRegex(ValueError, 'one or two'):
            s.validate_shortlist(expanded)

    def test_pending_or_software_error_cannot_be_scored_as_gray(self):
        original = make_fixture()
        for status in ('ARITHMETIC_CANONICAL_RX_REQUIRED', 'SOFTWARE_ERROR', 'UNKNOWN'):
            book, rows, protocol, interpretation = copy.deepcopy(original)
            rows[0]['source_status'] = status; rows[0]['gray'] = True
            rows[0]['rx_summary'].update(source_status=status, gray=True)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'pending/software'):
                select(book, rows, protocol, interpretation)
        book, rows, protocol, interpretation = copy.deepcopy(original)
        rows[0]['rx_summary']['source_decode_complete'] = False
        with self.assertRaisesRegex(ValueError, 'not complete'):
            select(book, rows, protocol, interpretation)

    def test_both_registered_failure_kinds_remain_in_mean(self):
        book, rows, protocol, interpretation = make_fixture()
        for row, status in zip(rows[:2], ('WIRE_REJECT_GRAY', 'ARITHMETIC_SOURCE_INVALID_GRAY')):
            row.update(source_status=status, gray=True, psnr_db=8., mse=10**(-.8))
            row['rx_summary'].update(source_status=status, gray=True)
        rows[0]['rx_summary'].update(header_accepted=False, body_parser_accepted=None)
        rows[1]['rx_summary'].update(arithmetic_canonical_attempted=True, arithmetic_canonical=False,
                                    canonical_decode_invalid=True, received_mode='arithmetic')
        result = select(book, rows, protocol, interpretation)
        summary = next(x for x in result['all_candidate_summaries'] if x['slot'] == 0)
        self.assertAlmostEqual(summary['mean_per_source_psnr_db'], (598*20+2*8)/600)
        self.assertEqual(summary['final_receiver_status_counts']['WIRE_REJECT_GRAY'], 1)
        self.assertEqual(summary['final_receiver_status_counts']['ARITHMETIC_SOURCE_INVALID_GRAY'], 1)

    def test_raw_bytes_image_hash_and_archive_reference_enforced(self):
        book, rows, protocol, interpretation = make_fixture()
        sb, mb = encoded(book), encoded(rows); r = receipt(sb, mb, rows)
        with self.assertRaisesRegex(ValueError, 'raw SHA'):
            s.select_initial(sb+b' ', mb, frame_metrics_path='/test/frame_metrics.json',render_completion=r,
                             protocol=protocol, interpretation=interpretation)
        with self.assertRaisesRegex(ValueError, 'metrics file'):
            s.select_initial(sb, mb+b' ', frame_metrics_path='/test/frame_metrics.json',render_completion=r,
                             protocol=protocol, interpretation=interpretation)
        changed = copy.deepcopy(rows); changed[0]['image_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'image SHA'):
            select(book, changed, protocol, interpretation)
        changed = copy.deepcopy(rows); changed[0]['image_archive'] = '/test/unbound.npz'
        with self.assertRaisesRegex(ValueError, 'archive/reference'):
            select(book, changed, protocol, interpretation)

    def test_incomplete_render_false_truth_flags_and_psnr_mse_mismatch_refused(self):
        book, rows, protocol, interpretation = make_fixture()
        with self.assertRaisesRegex(ValueError, 'completed'):
            select(book, rows, protocol, interpretation, lambda r: r.update(source_decode_complete=False))
        changed = copy.deepcopy(rows); changed[0]['rx_summary']['truth_correction'] = True
        with self.assertRaisesRegex(ValueError, 'source truth'):
            select(book, changed, protocol, interpretation)
        changed = copy.deepcopy(rows); changed[0]['psnr_db'] = 21.
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            select(book, changed, protocol, interpretation)


if __name__ == '__main__':
    unittest.main()
