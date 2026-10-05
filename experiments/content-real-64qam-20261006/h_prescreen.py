"""H expected-PSNR prescreen. Reads frozen evidence; never invokes any decoder.

The interval below is a marginal-Wilson propagation envelope, NOT a joint95% CI.
U images are unknown and assigned gray ONLY for this registered screening model.
"""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

from h64_catalog import RATES, catalogue, digest, token_count

ARMS = ('H16-R', 'H16-A', 'H64-R', 'H64-A')
PARTIAL = 'H64-RAW-COMPLETE-STATE-CONTROL'
SNRS = (13, 19)
TOL = 1e-12
INTERPRETATION = 'H_PRESCREEN_INTERPRETATION_V1'
ENVELOPE = 'Wilson95 marginal-propagated screening envelope; not a joint95% confidence interval'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def seal(path, value):
    """No silent overwrite or recovery of existing evidence."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode('utf-8')
    if p.exists():
        require(p.read_bytes() == content, 'Existing output differs: ' + str(p))
        return
    with p.open('xb') as f:
        f.write(content)


def verify(bindings):
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Changed binding: ' + str(path))


def wilson(success, n):
    require(type(success) is int and type(n) is int and 0 <= success <= n and n > 0,
            'Invalid Wilson binomial counts')
    z = 1.959963984540054
    p = success / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    # Mathematical endpoint identities; no modification to original tables.
    return [0. if success == 0 else max(0., center-width),
            1. if success == n else min(1., center+width)]


def expected_score(correct, gray, header, body):
    """Common channel probability multiplies the source-mean signed improvement."""
    require(len(correct) == len(gray) > 0 and all(math.isfinite(x) for x in correct+gray),
            'Bad source scores')
    pc = header['p_header_correct_point'] * body['correct']/body['trials']
    hlow, hhigh = header['p_header_correct_wilson95']
    blow, bhigh = wilson(body['correct'], body['trials'])
    require(0 <= hlow <= header['p_header_correct_point'] <= hhigh <= 1,
            'Invalid header interval')
    glow = math.fsum(gray)/len(gray)
    delta = math.fsum(c-g for c, g in zip(correct, gray))/len(gray)
    endpoints = [glow + p*delta for p in (hlow*blow, hhigh*bhigh)]
    return dict(expected_psnr_db=glow+pc*delta,
                psnr_screening_envelope=sorted(endpoints),
                expected_correct_probability=pc,
                body_correct_probability=body['correct']/body['trials'],
                body_crc_accepted_probability=(body['correct']+body['undetected'])/body['trials'],
                body_undetected_probability=body['undetected']/body['trials'],
                body_rejected_probability=body['rejected']/body['trials'],
                body_observed_BLER=1-body['correct']/body['trials'],
                mean_correct_psnr_db=glow+delta, mean_gray_psnr_db=glow,
                mean_signed_correct_minus_gray_psnr_db=delta,
                screening_surrogate_not_actual_U_image=True)


def tiebreak(row):
    return (row['mean_attempted_tx_probability_scales'], row['q'],
            Fraction(row['nominal_rate']), row['target_m'], row.get('K', 0), row['policy_key'])


def rank(rows):
    remaining = list(rows)
    result = []
    while remaining:
        best = max(x['expected_psnr_db'] for x in remaining)
        chosen = min((x for x in remaining if best-x['expected_psnr_db'] <= TOL), key=tiebreak)
        result.append(chosen)
        remaining.remove(chosen)
    return result


def resolve_whole(arm, target_m, bucket, source, profiles):
    """Apply the source-observable, pre-channel fallback; retain all attempts."""
    require(arm in ARMS and target_m in (6, 7, 8, 9), 'Unregistered whole policy')
    q = 4 if arm.startswith('H16') else 6
    require(bucket['q'] == q, 'Arm/modulation mismatch')
    attempts = []
    for m in range(target_m, 5, -1):
        quality = source['whole'][m]
        raw, arithmetic = quality['raw_bits'], quality['arithmetic_bits']
        mode = 'arithmetic' if arm.endswith('-A') and arithmetic < raw else 'raw'
        length = arithmetic if mode == 'arithmetic' else raw
        fits = length <= bucket['source_capacity']
        attempts.append(dict(m=m, raw_bits=raw, arithmetic_bits=arithmetic,
                             chosen_mode=mode, chosen_bits=length, fits=fits))
        if fits:
            key = (q, bucket['nominal_rate'], m, 0, mode)
            require(key in profiles, 'Chosen fallback has no paid public profile')
            p = profiles[key]
            return dict(source_id=source['source_id'], actual_m=m, actual_K=0,
                        actual_mode=mode, payload_bits=length,
                        actual_profile_id=p['profile_id'], actual_profile_key=p['profile_key'],
                        correct_psnr_db=quality['psnr_db'], correct_mse=quality['mse'],
                        gray_psnr_db=source['gray']['psnr_db'], gray_mse=source['gray']['mse'],
                        overflow_fallback=m != target_m,
                        raw_in_arithmetic_arm=arm.endswith('-A') and mode == 'raw', attempts=attempts)
    raise RuntimeError('No registered m6 fits admitted bucket')


def validate_catalog(cat):
    require(cat == catalogue(cat['buckets']), 'Catalogue differs from complete frozen enumeration')
    require(len(cat['buckets']) == 8 and all(x['admission'] == 'ADMITTED' for x in cat['buckets']),
            'This execution requires the eight qualified layouts')
    rows = [p for p in cat['profiles'] if PARTIAL in p['families']]
    require([sum(p['nominal_rate'] == r for p in rows) for r in RATES] == [236, 316, 356, 395],
            'Raw64 action coverage truncated')
    require({token_count(p['m'], p['K']) for p in rows} == set(range(1, 396)),
            'Missing one of395 canonical clean states')
    return rows


def validate_sources(sources, states, expected_count=200):
    require(len(sources) == expected_count and len({s['source_id'] for s in sources}) == expected_count,
            'Expected exact unique hash200 source population')
    for source in sources:
        require(set(source['whole']) == {6, 7, 8, 9}, 'Missing whole target length/quality')
        require(set(source['clean']) == states, 'Incomplete clean raw64 state coverage')
        for m, row in source['whole'].items():
            require(row['raw_bits'] == 12*token_count(m) and type(row['arithmetic_bits']) is int
                    and row['arithmetic_bits'] > 0, 'Invalid actual payload length')
        for row in list(source['whole'].values())+list(source['clean'].values())+[source['gray']]:
            require(0 < row['mse'] <= 1 and math.isfinite(row['psnr_db']), 'Invalid image quality')
            require(abs(row['psnr_db']+10*math.log10(row['mse'])) < 1e-9,
                    'PSNR and original per-source MSE disagree')


def validate_body(rows, cat, trials=2048, snrs=(13, 16, 19)):
    buckets = {b['layout']['layout_id']: b for b in cat['buckets']}
    result = {}
    for r in rows:
        key = (r['layout_id'], r['snr_db'])
        require(key not in result and key[0] in buckets and key[1] in snrs, 'Unexpected or duplicate PHY cell')
        b = buckets[key[0]]
        require(all(r[k] == b[k] for k in ('q', 'nominal_rate', 'k', 'n', 'resource_id', 'source_capacity')),
                'Body layout/capacity mismatch')
        require(r['trials'] == trials and all(type(r[k]) is int and r[k] >= 0 for k in ('correct', 'rejected', 'undetected')),
                'Wrong observed trial/count identity')
        require(r['correct']+r['rejected']+r['undetected'] == trials, 'C/R/U do not partition observations')
        require(r['crc_accepted'] == r['correct']+r['undetected'], 'Accepted/correct distinction changed')
        for name in ('correct', 'rejected', 'undetected'):
            require(abs(r['probabilities'][name]-r[name]/trials) < 1e-14, 'Stored empirical probability differs')
            require(max(abs(a-bb) for a, bb in zip(r['wilson95'][name], wilson(r[name], trials))) < 1e-12,
                    'Stored Wilson interval differs')
        require(abs(r['empirical_BLER']-(1-r['correct']/trials)) < 1e-14, 'Stored BLER differs')
        result[key] = r
    require(set(result) == {(lid, snr) for lid in buckets for snr in snrs}, 'Incomplete PHY Cartesian table')
    return result


def candidates(cat, sources, body, headers):
    partial = validate_catalog(cat)
    states = {(p['m'], p['K']) for p in partial}
    validate_sources(sources, states)
    profiles = {(p['q'], p['nominal_rate'], p['m'], p['K'], p['mode']): p for p in cat['profiles']}
    by_bucket = {(b['q'], b['nominal_rate']): b for b in cat['buckets']}
    policies = []
    details = {}
    gray = [s['gray']['psnr_db'] for s in sources]
    graymse = math.fsum(s['gray']['mse'] for s in sources)/len(sources)
    for arm in ARMS:
        q = 4 if arm.startswith('H16') else 6
        for rate in RATES:
            b = by_bucket[q, rate]
            for m in (6, 7, 8, 9):
                wire = dict(arm=arm, q=q, nominal_rate=rate, target_m=m, K=0)
                key = digest(wire)
                actual = [resolve_whole(arm, m, b, s, profiles) for s in sources]
                details[key] = actual
                policies.append(dict(**wire, policy_key=key, candidate_id='HWHOLE:'+key,
                                     mean_attempted_tx_probability_scales=float(m if arm.endswith('-A') else 0),
                                     layout_id=b['layout']['layout_id'], source_capacity=b['source_capacity'],
                                     actual_m_histogram={str(x): sum(r['actual_m'] == x for r in actual) for x in (6, 7, 8, 9)},
                                     arithmetic_sources=sum(r['actual_mode'] == 'arithmetic' for r in actual),
                                     raw_sources=sum(r['actual_mode'] == 'raw' for r in actual),
                                     overflow_sources=sum(r['overflow_fallback'] for r in actual)))
    for p in partial:
        key = p['profile_key']
        policies.append(dict(arm=PARTIAL, q=6, nominal_rate=p['nominal_rate'], target_m=p['m'], m=p['m'], K=p['K'],
                             policy_key=key, candidate_id='HPARTIAL:'+key, profile_key=key, profile_id=p['profile_id'],
                             mean_attempted_tx_probability_scales=0., layout_id=p['layout_id'],
                             source_capacity=p['source_capacity'], token_count=token_count(p['m'], p['K'])))
    result = []
    for snr in SNRS:
        for p in policies:
            rows = details[p['policy_key']] if p['arm'] != PARTIAL else [s['clean'][p['m'], p['K']] for s in sources]
            cp = [r['correct_psnr_db'] if p['arm'] != PARTIAL else r['psnr_db'] for r in rows]
            cm = math.fsum(r['correct_mse'] if p['arm'] != PARTIAL else r['mse'] for r in rows)/len(rows)
            cell = body[p['layout_id'], snr]
            scored = expected_score(cp, gray, headers[snr], cell)
            result.append(dict(**p, snr_db=snr, **scored,
                               expected_overall_mse=graymse+scored['expected_correct_probability']*(cm-graymse),
                               mean_correct_mse=cm, mean_gray_mse=graymse))
    require(len(result) == 2*(64+1303), 'Whole or partial policy coverage truncated')
    return result, details


def refinement_requests(rows):
    """Five family comparisons; at most two unique physical cells globally."""
    cells = {}
    audit = []
    for snr in SNRS:
        for arm in ARMS+(PARTIAL,):
            ordered = rank([r for r in rows if r['snr_db'] == snr and r['arm'] == arm])
            require(bool(ordered), 'Missing competition family')
            best = ordered[0]
            for position, row in enumerate(ordered[:2]):
                low, high = row['psnr_screening_envelope']
                bestlow, besthigh = best['psnr_screening_envelope']
                overlap = max(low, bestlow) <= min(high, besthigh)
                eligible = .01 <= row['body_observed_BLER'] <= .99 and overlap
                proof = dict(candidate_id=row['candidate_id'], snr_db=snr, arm=arm,
                             within_family_rank=position+1, family_best_id=best['candidate_id'],
                             interval_overlaps_family_best=overlap, body_observed_BLER=row['body_observed_BLER'],
                             eligible=eligible, envelope_width=high-low)
                audit.append(proof)
                if not eligible:
                    continue
                key = (row['layout_id'], snr)
                if key not in cells:
                    cells[key] = dict(layout_id=row['layout_id'], snr_db=snr, q=row['q'],
                                      nominal_rate=row['nominal_rate'], priority_width=high-low,
                                      additional_body_calls=6144, required_merged_trials=8192,
                                      original_trials=2048, qualifying_candidates=[])
                cells[key]['priority_width'] = max(cells[key]['priority_width'], high-low)
                cells[key]['qualifying_candidates'].append(proof)
    ordered = sorted(cells.values(), key=lambda r: (-r['priority_width'], r['snr_db'], r['q'], Fraction(r['nominal_rate'])))
    return dict(schema='H_REFINEMENT_REQUEST_V1', status='H_REFINEMENT_REQUEST_FROZEN',
                requests=[dict(r, refinement_rank=i+1) for i, r in enumerate(ordered[:2])],
                eligible_cells=len(ordered), not_selected_cells=ordered[2:],
                family_top2_audit=audit, maximum_additional_body_calls=12288,
                requested_body_calls=6144*len(ordered[:2]), new_decodes_in_this_stage=0)


def shortlist(rows, ready, phase):
    whole, partial = [], []
    for snr in SNRS:
        for arm in ARMS+(PARTIAL,):
            take = 3 if arm == PARTIAL else 2
            selected = rank([r for r in rows if r['snr_db'] == snr and r['arm'] == arm])[:take]
            for i, row in enumerate(selected):
                result = dict(row, selection_rank=i+1)
                if arm == PARTIAL:
                    result['slot'] = len(partial)
                    partial.append(result)
                else:
                    result['slot'] = len(whole)
                    whole.append(result)
    return dict(status='H_EXPECTED_PSNR_SHORTLIST_FROZEN' if ready else 'H_EXPECTED_PSNR_SHORTLIST_PROVISIONAL',
                phase=phase, ready_for_real_calibration=ready, whole_candidates=whole,
                partial_candidates=partial, interval_label=ENVELOPE,
                development_used=False, unknown_accepted_scored_as_gray_only_for_screening=True)


class Evidence:
    def __init__(self):
        self.bindings = {}

    def json(self, path, expected=None):
        path = str(Path(path).absolute())
        actual = sha(path)
        require(expected is None or actual == expected, 'Receipt output SHA mismatch: '+path)
        self.bindings[path] = actual
        return read(path)

    def output(self, receipt, path):
        path = str(Path(path).absolute())
        require(path in receipt['outputs'], 'Consumed file missing from completion: '+path)
        return self.json(path, receipt['outputs'][path])


def load_population(evidence, cfg, expected_source_registration=None):
    expected_ids = evidence.json(cfg['source200'])['source_ids']
    require(len(expected_ids) == len(set(expected_ids)) == 200, 'Wrong fixed source200')
    srcdir, cleandir = Path(cfg['source_dir']), Path(cfg['clean_dir'])
    src = evidence.json(srcdir/'completion.json')
    clean = evidence.json(cleandir/'completion.json')
    require(src['status'] == 'H_SOURCE200_COMPLETE' and clean['status'] == 'H_CLEAN_QUALITY200_COMPLETE',
            'Source/clean evidence not complete')
    require(src['registration_sha256'] == clean['registration_sha256'], 'Source/clean registrations differ')
    require(expected_source_registration is None or src['registration_sha256'] == expected_source_registration,
            'Source/clean are not from the qualified initial execution')
    require(src['source_count'] == clean['source_count'] == 200, 'Source population incomplete')
    sources = []
    for i, sid in enumerate(expected_ids):
        cp = evidence.output(src, srcdir/'source_checkpoints'/f'{i:04d}.json')
        cq = evidence.output(clean, cleandir/'source_checkpoints'/f'{i:04d}.json')
        require(cp['source_id'] == cq['source_id'] == sid and cp['source_index'] == cq['source_index'] == i,
                'Population/preprocessing order changed')
        require(cp['registration_sha256'] == cq['registration_sha256'] == src['registration_sha256'],
                'Checkpoint registration changed')
        require(cp['independent_roundtrip'] is True and cq['clean_states'] == 395
                and cq['all_legal_K_included'] is True and cq['new_channel_decodes'] == 0,
                'Clean-state evidence not qualified')
        qp = cleandir/'sources'/f'{i:04d}.json'
        quality = evidence.output(clean, qp)
        require(cq['outputs'].get(str(qp.absolute())) == sha(qp), 'Quality sidecar mismatch')
        require(len(quality) == 395 and len(cp['lengths']) == 4, 'Duplicate/missing source state')
        require(all(r['token_count'] == token_count(r['m'], r['K']) for r in quality), 'Noncanonical quality state')
        sources.append(dict(source_id=sid, gray=cp['gray'], whole={r['m']: r for r in cp['lengths']},
                            clean={(r['m'], r['K']): r for r in quality}))
    return sources


def apply_refinement(evidence, cfg, cat, coarse):
    previous = evidence.json(cfg['coarse_prescreen_completion'])
    require(previous['status'] == 'H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED', 'Wrong prior prescreen state')
    verify(previous.get('input_bindings', {}))
    request = evidence.output(previous, cfg['refinement_requests'])
    require(request['status'] == 'H_REFINEMENT_REQUEST_FROZEN' and 1 <= len(request['requests']) <= 2,
            'Wrong frozen refinement request')
    # A future independent physical stage must satisfy this explicit interface.
    complete = evidence.json(cfg['refinement_completion'])
    require(complete['status'] == 'H_REFINEMENT_COMPLETE' and
            complete['refinement_requests_sha256'] == sha(cfg['refinement_requests']),
            'Refinement is not bound to the frozen request')
    require(complete['additional_body_decode_events'] == 6144*len(request['requests'])
            and complete['header_decode_events'] == 0, 'Wrong refinement paid count')
    rows = evidence.output(complete, cfg['refined_cells'])
    wanted = {(r['layout_id'], r['snr_db']) for r in request['requests']}
    require(len(rows) == len(wanted) and {(r['layout_id'], r['snr_db']) for r in rows} == wanted,
            'Refinement changed or expanded the frozen cell set')
    result = dict(coarse)
    for row in rows:
        key = (row['layout_id'], row['snr_db'])
        old = coarse[key]
        onecat = dict(cat, buckets=[b for b in cat['buckets'] if b['layout']['layout_id'] == key[0]])
        validate_body([row], onecat, trials=8192, snrs=(key[1],))
        require(row['original_counts'] == {k: old[k] for k in ('trials', 'correct', 'rejected', 'undetected')},
                'Refinement lost original coarse observations')
        added = row['additional_counts']
        require(added['trials'] == 6144 and all(type(added[k]) is int and added[k] >= 0
                    and row[k] == old[k]+added[k] for k in ('correct', 'rejected', 'undetected')),
                'Merged counts are not exact original+additional')
        result[key] = row
    return result


def csv_seal(path, rows):
    import io
    require(bool(rows), 'Empty CSV')
    buffer = io.StringIO(newline='')
    columns = list(rows[0])
    columns += sorted({k for row in rows for k in row}-set(columns))
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: json.dumps(v, sort_keys=True, separators=(',', ':')) if isinstance(v, (dict, list)) else v
                     for k, v in row.items()} for row in rows)
    p = Path(path)
    data = buffer.getvalue().encode('utf-8')
    if p.exists():
        require(p.read_bytes() == data, 'Existing CSV differs')
    else:
        with p.open('xb') as f:
            f.write(data)


def run(config_path):
    cfg = read(config_path)
    reg = read(cfg['registration'])
    require(reg['status'] == 'H_EXECUTION_REVISION_REGISTERED' and reg['branch'] == 'H', 'Unregistered execution')
    verify(reg['source_bindings']); verify(reg['input_bindings'])
    require(reg['input_bindings'].get(str(Path(config_path).absolute())) == sha(config_path), 'Config is not bound')
    require(reg['source_bindings'].get(str(Path(__file__).absolute())) == sha(__file__), 'Prescreen source is not bound')
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
    require(not (out/'failure.json').exists(), 'Failure retained; no automatic retry')
    if (out/'completion.json').exists():
        done = read(out/'completion.json')
        require(done['registration_sha256'] == sha(cfg['registration']), 'Completion registration differs')
        verify(done['outputs']); verify(done['input_bindings']); return done
    evidence = Evidence()
    for key in ('protocol', 'interpretation', 'budget_registration', 'source200', 'catalogue',
                'header_reuse', 'qualification_completion'):
        require(reg['input_bindings'].get(str(Path(cfg[key]).absolute())) == sha(cfg[key]),
                'Scientific input not bound: '+key)
    protocol = evidence.json(cfg['protocol'])
    require(protocol['schema'] == 'H_CODEC_PROTOCOL_V1' and protocol['main_snrs_db'] == [13, 19]
            and protocol['status'] == 'FROZEN_BEFORE_DATA', 'Wrong frozen H protocol')
    interpretation = evidence.json(cfg['interpretation'])
    require(interpretation['schema'] == INTERPRETATION and interpretation['original_protocol_changed'] is False
            and interpretation['status'] == 'FROZEN_BEFORE_COARSE_DATA',
            'Missing registered prescreen interpretation')
    budget = evidence.json(cfg['budget_registration'])
    require(budget['phase_limits']['refine'] == 12288 and budget['total_cap'] == 200000, 'Wrong immutable budget')
    cat = evidence.json(cfg['catalogue']); validate_catalog(cat)
    qual = evidence.json(cfg['qualification_completion'])
    require(qual['status'] == 'H_PHY_QUALIFICATION_PASS'
            and qual['outputs'].get(str(Path(cfg['catalogue']).absolute())) == sha(cfg['catalogue']),
            'Catalogue is not qualified evidence')
    header = evidence.json(cfg['header_reuse'])
    require(header['status'] == 'H_HEADER_CORRECT_PROBABILITY_REUSE_REGISTERED', 'Header reuse not registered')
    headers = {p['snr_db']: p for p in header['points']}
    require(set(headers) == set(SNRS), 'Main header probabilities missing')
    coarse_done = evidence.json(Path(cfg['coarse_dir'])/'completion.json')
    require(coarse_done['status'] == 'H_COARSE_COMPLETE' and coarse_done['body_decode_events'] == 49152
            and coarse_done['header_decode_events'] == 0
            and coarse_done['budget_registration_sha256'] == sha(cfg['budget_registration'])
            and coarse_done['qualification_completion_sha256'] == sha(cfg['qualification_completion']),
            'Wrong completed coarse table identity')
    coarse = validate_body(evidence.output(coarse_done, Path(cfg['coarse_dir'])/'bler_counts.json'), cat)
    sources = load_population(evidence, cfg, qual['registration_sha256'])
    require(cfg['phase'] in ('coarse', 'final'), 'Unregistered prescreen phase')
    basis_paths = {k: str(Path(cfg[k]).absolute()) for k in ('protocol', 'interpretation', 'budget_registration',
                   'source200', 'catalogue', 'header_reuse', 'qualification_completion')}
    basis_paths.update({k+'_completion': str((Path(cfg[k])/'completion.json').absolute())
                       for k in ('source_dir', 'clean_dir', 'coarse_dir')})
    basis = {k: dict(path=p, sha256=sha(p)) for k, p in basis_paths.items()}
    if cfg['phase'] == 'final':
        require(evidence.json(cfg['coarse_prescreen_completion'])['prescreen_basis'] == basis,
                'Final prescreen changed the original source/quality/header/coarse inputs')
    table = coarse if cfg['phase'] == 'coarse' else apply_refinement(evidence, cfg, cat, coarse)
    rows, details = candidates(cat, sources, table, headers)
    requests = refinement_requests(rows) if cfg['phase'] == 'coarse' else None
    ready = cfg['phase'] == 'final' or not requests['requests']
    selected = shortlist(rows, ready, cfg['phase'])
    selected.update(source_ids=[s['source_id'] for s in sources], input_bindings=evidence.bindings,
                    registration_sha256=sha(cfg['registration']))
    outputs = {}
    if requests is not None:
        requests.update(input_bindings=evidence.bindings, registration_sha256=sha(cfg['registration']))
        seal(out/'refinement_requests.json', requests)
        outputs[str((out/'refinement_requests.json').absolute())] = sha(out/'refinement_requests.json')
    seal(out/'shortlist.json', selected)
    seal(out/'all_candidates.json', rows)
    seal(out/'whole_per_source_fallback.json', details)
    compact = [{k: r[k] for k in ('candidate_id', 'arm', 'snr_db', 'target_m', 'K', 'q', 'nominal_rate',
                'layout_id', 'expected_psnr_db', 'psnr_screening_envelope', 'expected_overall_mse',
                'body_observed_BLER', 'mean_attempted_tx_probability_scales')} for r in rows]
    csv_seal(out/'candidate_ranking.csv', compact)
    for name in ('shortlist.json', 'all_candidates.json', 'whole_per_source_fallback.json', 'candidate_ranking.csv'):
        outputs[str((out/name).absolute())] = sha(out/name)
    verify(evidence.bindings); verify(reg['source_bindings']); verify(reg['input_bindings'])
    done = dict(status='H_PRESCREEN_COMPLETE_FINAL' if ready else 'H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED',
                schema='H_PRESCREEN_COMPLETION_V1', phase=cfg['phase'],
                registration_sha256=sha(cfg['registration']), input_bindings=evidence.bindings,
                prescreen_basis=basis,
                source_bindings=reg['source_bindings'], outputs=outputs,
                source_count=200, distinct_raw_states=395, raw_profiles_per_snr=1303,
                whole_policies_per_snr=64, scored_policy_snr_pairs=len(rows),
                ready_for_real_calibration=ready, new_packet_decodes=0, new_visual_inference=0,
                development_used=False, holdout_used=False, interval_label=ENVELOPE,
                verification_scope='Hashes of consumed receipt/JSON files verified; unused image NPZ arrays not read')
    seal(out/'completion.json', done)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config), ensure_ascii=False))
    except BaseException as exc:
        cfg = read(args.config)
        failure = Path(cfg['out'])/'failure.json'
        if not failure.exists():
            seal(failure, dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc(),
                               config_sha256=sha(args.config), new_packet_decodes=0))
        raise


if __name__ == '__main__':
    main()
