import copy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h64_catalog import all_buckets, catalogue, RATES, token_count
from h_prescreen import (ARMS, PARTIAL, Evidence, apply_refinement, candidates,
    expected_score, load_population, rank, refinement_requests, resolve_whole,
    seal, sha, shortlist, validate_body, validate_catalog, validate_sources, wilson, csv_seal, run)


def fixture_catalog():
    buckets = all_buckets()
    for i, b in enumerate(buckets):
        b.update(admission='ADMITTED', layout={'layout_id': f'test_layout_{i}'})
    return catalogue(buckets)


def quality(psnr):
    return {'psnr_db': float(psnr), 'mse': 10.**(-psnr/10)}


def fixture_source(cat, i=0):
    states = {(p['m'], p['K']) for p in cat['profiles'] if PARTIAL in p['families']}
    return dict(source_id=f'test_source_{i:04d}', gray=quality(9+i*.0001),
                whole={m: dict(m=m, raw_bits=12*token_count(m),
                    arithmetic_bits=12*token_count(m)-100, **quality(m+12+i*.0001)) for m in (6, 7, 8, 9)},
                clean={state: dict(m=state[0], K=state[1], token_count=token_count(*state),
                    **quality(12+token_count(*state)*.015+i*.0001)) for state in states})


def fixture_cell(b, snr, n=2048, c=1700, u=2):
    counts = dict(correct=c, undetected=u, rejected=n-c-u)
    return dict(layout_id=b['layout']['layout_id'], snr_db=snr, trials=n,
                **{k: b[k] for k in ('q', 'nominal_rate', 'k', 'n', 'resource_id', 'source_capacity')},
                **counts, crc_accepted=c+u, empirical_BLER=1-c/n,
                probabilities={k: v/n for k, v in counts.items()},
                wilson95={k: wilson(v, n) for k, v in counts.items()})


def ranked_row(identifier, *, arm='H16-R', snr=13, score=20., interval=(19., 21.),
               layout='one', q=4, rate='1/2', m=7, K=0, cost=0., bler=.2):
    return dict(candidate_id=identifier, arm=arm, snr_db=snr, target_m=m, K=K, q=q,
                nominal_rate=rate, policy_key=identifier, layout_id=layout,
                expected_psnr_db=score, psnr_screening_envelope=list(interval),
                mean_attempted_tx_probability_scales=cost, body_observed_BLER=bler)


class PrescreenTests(unittest.TestCase):
    def test_all_integer_states_admitted_and_missing_state_fails(self):
        cat = fixture_catalog()
        profiles = validate_catalog(cat)
        self.assertEqual(len(profiles), 1303)
        self.assertEqual([sum(p['nominal_rate'] == r for p in profiles) for r in RATES], [236, 316, 356, 395])
        self.assertEqual(max(token_count(p['m'], p['K']) for p in profiles), 395)
        broken = copy.deepcopy(cat); broken['profiles'].pop()
        with self.assertRaisesRegex(RuntimeError, 'enumeration'):
            validate_catalog(broken)

    def test_mean_psnr_failure_mass_U_not_crc_accept_success_and_signed_envelope(self):
        header = dict(p_header_correct_point=.8, p_header_correct_wilson95=[.7, .9])
        body = dict(correct=40, undetected=50, rejected=10, trials=100)
        value = expected_score([30., 10.], [5., 15.], header, body)
        self.assertAlmostEqual(value['expected_psnr_db'], .32*20+.68*10)
        self.assertAlmostEqual(value['body_crc_accepted_probability'], .9)
        self.assertAlmostEqual(value['expected_correct_probability'], .32)
        self.assertNotAlmostEqual(value['expected_psnr_db'], -10*math.log10(.32*(.001+.1)/2+.68*(10**-.5+10**-1.5)/2))
        # The same probability multiplies the mean delta, not an adversarial per-source probability.
        lo, hi = wilson(40, 100)
        self.assertEqual(value['psnr_screening_envelope'], [10+.7*lo*10, 10+.9*hi*10])
        worse = expected_score([2., 4.], [5., 7.], header, body)
        self.assertAlmostEqual(worse['psnr_screening_envelope'][0], 6-.9*hi*3)
        self.assertAlmostEqual(worse['psnr_screening_envelope'][1], 6-.7*lo*3)

    def test_equal_arithmetic_length_uses_raw_and_overflow_is_source_specific(self):
        cat = fixture_catalog(); src = fixture_source(cat)
        profiles = {(p['q'], p['nominal_rate'], p['m'], p['K'], p['mode']): p for p in cat['profiles']}
        b = cat['buckets'][0]
        # At r1/2 16QAM sourcecapacity1883, m8 exceeds and m7 fits.
        src['whole'][7]['arithmetic_bits'] = src['whole'][7]['raw_bits']
        result = resolve_whole('H16-A', 9, b, src, profiles)
        self.assertEqual((result['actual_m'], result['actual_mode'], result['payload_bits']), (7, 'raw', 1860))
        self.assertEqual([r['m'] for r in result['attempts']], [9, 8, 7])
        self.assertTrue(result['overflow_fallback'])
        second = copy.deepcopy(src); second['whole'][9]['arithmetic_bits'] = 1800
        result2 = resolve_whole('H16-A', 9, b, second, profiles)
        self.assertEqual((result2['actual_m'], result2['actual_mode']), (9, 'arithmetic'))
        self.assertEqual(resolve_whole('H16-R', 9, b, second, profiles)['actual_m'], 7)

    def test_nontransitive_epsilon_ties_are_deterministic_best_extraction(self):
        rows = [ranked_row('A', score=20., rate='5/6'),
                ranked_row('B', score=20.-.75e-12, rate='2/3'),
                ranked_row('C', score=20.-1.5e-12, rate='1/2')]
        self.assertEqual([r['candidate_id'] for r in rank(rows)], ['B', 'A', 'C'])
        self.assertEqual(rank(rows[::-1]), rank(rows))
        # Fewer TX probability scales precedes the lower modulation tie breaker.
        raw = ranked_row('raw', q=6, cost=0.)
        arithmetic = ranked_row('arith', q=4, cost=6.)
        self.assertEqual(rank([arithmetic, raw])[0], raw)

    def test_refinement_uses_five_families_deduplicates_and_never16dB(self):
        rows = []
        for snr in (13, 19):
            for i, arm in enumerate(ARMS+(PARTIAL,)):
                rows.append(ranked_row(f'{arm}_{snr}_best', arm=arm, snr=snr, score=10.+i,
                                      interval=(8.+i, 12.+i), layout=f'layout{i}', q=4 if i<2 else 6))
                rows.append(ranked_row(f'{arm}_{snr}_second', arm=arm, snr=snr, score=9.+i,
                                      interval=(8.+i, 10.+i), layout=f'layout{i}', q=4 if i<2 else 6))
        requests = refinement_requests(rows)
        self.assertEqual(len(requests['family_top2_audit']), 20)
        self.assertEqual(requests['eligible_cells'], 10)
        self.assertEqual(len(requests['requests']), 2)
        self.assertEqual(requests['requested_body_calls'], 12288)
        self.assertEqual([r['snr_db'] for r in requests['requests']], [13, 13])
        self.assertTrue(all(len(r['qualifying_candidates']) == 2 for r in requests['requests']))
        # A low-performing family's best still qualifies; no comparison with global winner.
        self.assertTrue(any(p['arm'] == 'H16-R' and p['eligible'] for p in requests['family_top2_audit']))
        none = [dict(r, body_observed_BLER=0.) for r in rows]
        self.assertEqual(refinement_requests(none)['requests'], [])
        # Statistical overlap alone does not waive the .01..99 rule.
        for r in rows: r['body_observed_BLER'] = .999
        self.assertEqual(refinement_requests(rows)['requests'], [])

    def test_complete_coverage_fallbacks_and_provisional_final_identity(self):
        cat = fixture_catalog(); sources = [fixture_source(cat, i) for i in range(200)]
        cells = [fixture_cell(b, snr) for b in cat['buckets'] for snr in (13, 16, 19)]
        body = validate_body(cells, cat)
        headers = {snr: dict(p_header_correct_point=1., p_header_correct_wilson95=wilson(20000, 20000))
                   for snr in (13, 19)}
        scored, details = candidates(cat, sources, body, headers)
        self.assertEqual(len(scored), 2734)
        self.assertEqual(len(details), 64)
        self.assertEqual(sum(r['arm'] == PARTIAL for r in scored), 2606)
        self.assertEqual(sum(r['arm'] == 'H64-A' and r['snr_db'] == 13 for r in scored), 16)
        selected = shortlist(scored, False, 'coarse')
        self.assertEqual(len(selected['whole_candidates']), 16)
        self.assertEqual(len(selected['partial_candidates']), 6)
        self.assertEqual([r['slot'] for r in selected['whole_candidates']], list(range(16)))
        self.assertEqual(selected['status'], 'H_EXPECTED_PSNR_SHORTLIST_PROVISIONAL')
        self.assertFalse(selected['ready_for_real_calibration'])
        self.assertTrue(shortlist(scored, True, 'final')['ready_for_real_calibration'])

    def test_rejects_incomplete_phy_and_accepted_correct_confusion(self):
        cat = fixture_catalog()
        rows = [fixture_cell(b, s) for b in cat['buckets'] for s in (13, 16, 19)]
        validate_body(rows, cat)
        with self.assertRaisesRegex(RuntimeError, 'Cartesian'):
            validate_body(rows[:-1], cat)
        bad = copy.deepcopy(rows); bad[0]['correct'] += 2
        with self.assertRaisesRegex(RuntimeError, 'partition'):
            validate_body(bad, cat)
        bad = copy.deepcopy(rows); bad[0]['crc_accepted'] = bad[0]['correct']
        with self.assertRaisesRegex(RuntimeError, 'Accepted/correct'):
            validate_body(bad, cat)
        bad = copy.deepcopy(rows); bad[0]['n'] += 6
        with self.assertRaisesRegex(RuntimeError, 'layout/capacity'):
            validate_body(bad, cat)

    def test_receipt_sealing_and_consumed_hash_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.json'
            seal(path, {'data': [1, 2]}); seal(path, {'data': [1, 2]})
            expected = sha(path); evidence = Evidence()
            self.assertEqual(evidence.output({'outputs': {str(path): expected}}, path), {'data': [1, 2]})
            with self.assertRaisesRegex(RuntimeError, 'Existing output'):
                seal(path, {'data': [9]})
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
                Evidence().output({'outputs': {str(path): expected}}, path)

    def test_csv_supports_whole_and_partial_fields_without_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mixed.csv'
            csv_seal(path, [dict(arm='H16-A', target_m=8),
                            dict(arm=PARTIAL, target_m=8, K=140, profile_id=123, token_count=395)])
            self.assertNotIn(b'\r', path.read_bytes())
            with path.open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[1]['token_count'], '395')
            self.assertEqual(rows[1]['K'], '140')
            self.assertEqual(rows[0]['profile_id'], '')

    def test_final_refinement_requires_exact_frozen_cells_and_count_merge(self):
        cat = fixture_catalog()
        coarse = validate_body([fixture_cell(b, s) for b in cat['buckets'] for s in (13, 16, 19)], cat)
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            requests = d/'requests.json'; previous = d/'previous.json'
            complete = d/'refined_completion.json'; refined = d/'refined.json'
            b = cat['buckets'][0]; old = coarse[b['layout']['layout_id'], 13]
            request = dict(status='H_REFINEMENT_REQUEST_FROZEN', requests=[dict(layout_id=old['layout_id'], snr_db=13)])
            seal(requests, request)
            seal(previous, dict(status='H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED', outputs={str(requests): sha(requests)}))
            row = fixture_cell(b, 13, n=8192, c=6800, u=8)
            row['original_counts'] = {k: old[k] for k in ('trials', 'correct', 'rejected', 'undetected')}
            row['additional_counts'] = dict(trials=6144, correct=5100, undetected=6, rejected=1038)
            seal(refined, [row])
            seal(complete, dict(status='H_REFINEMENT_COMPLETE', refinement_requests_sha256=sha(requests),
                 additional_body_decode_events=6144, header_decode_events=0, outputs={str(refined): sha(refined)}))
            cfg = dict(coarse_prescreen_completion=str(previous), refinement_requests=str(requests),
                       refinement_completion=str(complete), refined_cells=str(refined))
            result = apply_refinement(Evidence(), cfg, cat, coarse)
            self.assertEqual(result[old['layout_id'], 13]['trials'], 8192)
            self.assertEqual(result[old['layout_id'], 19]['trials'], 2048)
            # A bad merged count is rejected even when its enclosing file SHA is newly bound.
            row['additional_counts']['correct'] -= 1
            refined.write_text(json.dumps([row]), encoding='utf-8')
            receipt = json.loads(complete.read_text()); receipt['outputs'][str(refined)] = sha(refined)
            complete.write_text(json.dumps(receipt), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'exact original'):
                apply_refinement(Evidence(), cfg, cat, coarse)

    def test_complete_registered_pipeline_with_real_receipt_shapes_and_200_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder); cat = fixture_catalog()
            sources = [fixture_source(cat, i) for i in range(200)]
            original_reg = 'a'*64
            source_dir, clean_dir, coarse_dir = d/'source200', d/'clean-quality200', d/'coarse'
            source_outputs, clean_outputs = {}, {}
            for i, source in enumerate(sources):
                cp = source_dir/'source_checkpoints'/f'{i:04d}.json'
                cq = clean_dir/'source_checkpoints'/f'{i:04d}.json'
                quality_path = clean_dir/'sources'/f'{i:04d}.json'
                seal(cp, dict(source_id=source['source_id'], source_index=i, registration_sha256=original_reg,
                              independent_roundtrip=True, lengths=list(source['whole'].values()), gray=source['gray'],
                              outputs={str(d/'deliberately_unused_image.npz'): '0'*64}))
                seal(quality_path, list(source['clean'].values()))
                seal(cq, dict(source_id=source['source_id'], source_index=i, registration_sha256=original_reg,
                              clean_states=395, all_legal_K_included=True, new_channel_decodes=0,
                              outputs={str(quality_path): sha(quality_path)}))
                source_outputs[str(cp)] = sha(cp)
                clean_outputs.update({str(cq): sha(cq), str(quality_path): sha(quality_path)})
            seal(source_dir/'completion.json', dict(status='H_SOURCE200_COMPLETE', source_count=200,
                  registration_sha256=original_reg, outputs=source_outputs))
            seal(clean_dir/'completion.json', dict(status='H_CLEAN_QUALITY200_COMPLETE', source_count=200,
                  registration_sha256=original_reg, outputs=clean_outputs))
            inputs = dict(protocol=d/'protocol.json', interpretation=d/'interpretation.json',
                          budget_registration=d/'budget.json', source200=d/'source_ids.json',
                          catalogue=d/'catalogue.json', header_reuse=d/'header.json',
                          qualification_completion=d/'qualification.json')
            seal(inputs['protocol'], dict(schema='H_CODEC_PROTOCOL_V1', status='FROZEN_BEFORE_DATA', main_snrs_db=[13, 19]))
            seal(inputs['interpretation'], dict(schema='H_PRESCREEN_INTERPRETATION_V1', original_protocol_changed=False,
                                                status='FROZEN_BEFORE_COARSE_DATA'))
            seal(inputs['budget_registration'], dict(phase_limits={'refine': 12288}, total_cap=200000))
            seal(inputs['source200'], dict(source_ids=[s['source_id'] for s in sources]))
            seal(inputs['catalogue'], cat)
            seal(inputs['header_reuse'], dict(status='H_HEADER_CORRECT_PROBABILITY_REUSE_REGISTERED', points=[
                dict(snr_db=s, p_header_correct_point=1., p_header_correct_wilson95=wilson(20000, 20000)) for s in (13, 19)]))
            seal(inputs['qualification_completion'], dict(status='H_PHY_QUALIFICATION_PASS', registration_sha256=original_reg,
                 outputs={str(inputs['catalogue']): sha(inputs['catalogue'])}))
            table = coarse_dir/'bler_counts.json'
            seal(table, [fixture_cell(b, s, c=2048, u=0) for b in cat['buckets'] for s in (13, 16, 19)])
            seal(coarse_dir/'completion.json', dict(status='H_COARSE_COMPLETE', body_decode_events=49152,
                 header_decode_events=0, budget_registration_sha256=sha(inputs['budget_registration']),
                 qualification_completion_sha256=sha(inputs['qualification_completion']), outputs={str(table): sha(table)}))
            cfg = dict(registration=str(d/'registration.json'), out=str(d/'prescreen'), phase='coarse',
                       source_dir=str(source_dir), clean_dir=str(clean_dir), coarse_dir=str(coarse_dir),
                       **{k: str(v) for k, v in inputs.items()})
            config = d/'config.json'; seal(config, cfg)
            code = Path(__file__).with_name('h_prescreen.py').absolute()
            seal(cfg['registration'], dict(status='H_EXECUTION_REVISION_REGISTERED', branch='H',
                 input_bindings={str(p): sha(p) for p in list(inputs.values())+[config]},
                 source_bindings={str(code): sha(code)}))
            done = run(config)
            self.assertEqual(done['status'], 'H_PRESCREEN_COMPLETE_FINAL')
            self.assertEqual(done['scored_policy_snr_pairs'], 2734)
            self.assertEqual(done['new_packet_decodes'], 0)
            self.assertEqual(done, run(config))
            shortlist_path = Path(cfg['out'])/'shortlist.json'
            selected = json.loads(shortlist_path.read_text())
            self.assertEqual(len(selected['whole_candidates']), 16)
            self.assertEqual(len(selected['partial_candidates']), 6)
            # A consumed clean output mutation is caught even on the completion reuse path.
            q0 = clean_dir/'sources'/'0000.json'
            q0.write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'Changed binding'):
                run(config)


if __name__ == '__main__':
    unittest.main()
