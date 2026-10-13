"""Frozen finite R0 nomination arithmetic. No PHY, model, or scientific runner.

rank_and_nominate is a pure CPU entry. Its caller must authenticate actual stage
closure and source/model pins before passing the joined evidence. It returns a
proposed shortlist, not an R1 registration or a GO decision.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import contract

require = contract.require
SELECTION_SNRS = (4, 7, 10)
TRIALS_PER_MISSING_POINT = 64
SUPPLEMENT_CAPS = dict(body_trials=88896, missing_layout_snr_points=1389,
    clean_var_reconstructions=5400, clean_dino_scores=5400,
    source_dino_embeddings=200, training_updates=0, header_trials=0, snr13_trials=0)
RULE_ID = 'LAYERED_RAW_R0_DROP_PROXY_NOMINATION_V1'
METRIC = 'dinov2_vitl14_cosine'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def probability(row, expected_n=None):
    n = row['n_blocks']
    counts = [row['n_correct'], row['n_reject'], row['n_undetected']]
    require(type(n) is int and n > 0 and all(type(x) is int and x >= 0 for x in counts)
            and sum(counts) == n, 'Nonpartitioned trial counts')
    require(expected_n is None or n == expected_n, 'Unexpected trial precision')
    for kind, count in zip(('correct', 'reject', 'undetected'), counts):
        p = float(row['p_' + kind]); lo, hi = row['p_' + kind + '_ci']
        require(math.isfinite(p) and abs(p - count / n) < 1e-14
                and 0 <= lo <= p + 1e-14 and p <= hi + 1e-14 and hi <= 1 + 1e-14,
                'Probability/count/interval conflict')
    return counts[0] / n


def exact_keys(catalogue, groups, matched, coverage):
    """Reconcile the already qualified catalogue and old exact-layout join."""
    profiles = catalogue['profiles']
    require(len(profiles) == 689 and len(groups) == 687, 'Frozen complete domain changed')
    require([p['profile_id'] for p in profiles] == list(range(689)), 'Paid IDs changed')
    used = {}
    for p in profiles:
        for group in p['groups']:
            key = group['phy_key']
            require(key in groups and group['layout'] == groups[key]['layout']
                and all(group[k] == groups[key][k] for k in
                    ('source_bits', 'information_bits', 'transmitted_bits', 'modulation', 'symbols')),
                    'Candidate group/qualified layout differs')
            used[key] = groups[key]
    require(set(used) == set(groups), 'Unreferenced or missing body group')
    historic = {}
    for row in matched:
        key = (row['phy_key'], row['snr_db'])
        require(key not in historic and key[0] in groups and key[1] in (4, 7, 10, 13),
                'Duplicate/outside historical exact key')
        probability(row)
        historic[key] = row
    require(len(historic) == 896, 'Historical four-SNR evidence changed')
    wanted = {(k, snr) for k in groups for snr in SELECTION_SNRS}
    old = {k: row for k, row in historic.items() if k[1] in SELECTION_SNRS}
    missing = wanted - set(old)
    coverage_missing = {(x['phy_key'], x['snr_db']) for row in coverage for x in row['missing']
                        if x['snr_db'] in SELECTION_SNRS}
    require(missing == coverage_missing and len(missing) == 1389 and len(old) == 672,
            'Missing exact keys disagree with frozen coverage')
    rows = []
    for index, (key, snr) in enumerate(sorted(missing)):
        group = groups[key]
        q = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[group['modulation']]
        rows.append(dict(point_index=index, phy_key=key, snr_db=snr, trials=64,
            source_bits=group['source_bits'], k=group['information_bits'], n=group['transmitted_bits'],
            q=q, symbols=group['symbols'], layout_id=group['layout']['layout_id'], layout_sha256=contract.digest(group['layout']),
            layout=group['layout']))
    return rows, [old[k] for k in sorted(old)]


def clean_join(profiles, source_ids, historical, new, gray):
    """Exact source/endpoint join; no averaging across different populations."""
    source_ids = list(source_ids)
    require(len(source_ids) == len(set(source_ids)) and bool(source_ids), 'Ambiguous source population')
    ends = {(int(p['m']), int(p['K'])) for p in profiles}
    ends |= {(int(p['j']), 0) for p in profiles if p['G'] == 2}
    wanted = {(s, m, k) for s in source_ids for m, k in ends}
    values = {}
    for origin, rows in (('historical', historical), ('new', new)):
        for row in rows:
            key = (str(row['source_id']), int(row['m']), int(row['K']))
            require(key in wanted and key not in values, 'Duplicate/foreign clean source-state')
            score = float(row[METRIC])
            require(math.isfinite(score) and -1 <= score <= 1, 'Invalid clean DINO score')
            values[key] = score
    require(set(values) == wanted, 'Missing clean source-state; no extrapolation')
    gray_scores = {}
    for row in gray:
        source = str(row['source_id'])
        require(source in source_ids and source not in gray_scores, 'Gray source duplicate/mismatch')
        score = float(row[METRIC]); require(math.isfinite(score) and -1 <= score <= 1, 'Invalid gray DINO')
        gray_scores[source] = score
    require(set(gray_scores) == set(source_ids), 'Missing per-source fixed-gray quality')
    return values, gray_scores


def drop_proxy(profile, source_id, snr, probabilities, header, quality, gray):
    """Correct-reception mixture, conditional independent body-group proxy.

    CRC false acceptance and header false acceptance are treated as non-correct
    outcomes in this approximation. Their actual decoded-image quality is not
    known. This is NOT a guaranteed lower bound and does not model KEEP.
    """
    require(snr in SELECTION_SNRS, 'Pressure SNR cannot enter R0 ranking')
    ph = header[snr]
    p1 = probabilities[(profile['groups'][0]['phy_key'], snr)]
    terminal = quality[(source_id, int(profile['m']), int(profile['K']))]
    if profile['G'] == 1:
        return ph * p1 * terminal + (1 - ph * p1) * gray[source_id]
    p2 = probabilities[(profile['groups'][1]['phy_key'], snr)]
    base = quality[(source_id, int(profile['j']), 0)]
    return ph * p1 * (p2 * terminal + (1 - p2) * base) + (1 - ph * p1) * gray[source_id]


def rank_and_nominate(profiles, source_ids, body_rows, header_rows,
                      historical_clean, new_clean, gray_rows, anchors):
    """Pure-data ranking entry, to be called after actual closure authentication.

    Inputs may contain only selection SNRs; 13 dB data are rejected rather than
    read for a pressure-informed nomination. Every family uses the same point
    estimate, no precision adjustment, adaptive sampling or q6 exclusion.
    """
    source_ids = list(source_ids)
    require(len({p['profile_id'] for p in profiles}) == len(profiles), 'Duplicate profile IDs')
    require(len({p['wire_key'] for p in profiles}) == len(profiles), 'Duplicate public wire')
    by_id = {p['profile_id']: p for p in profiles}
    required_keys = {(g['phy_key'], snr) for p in profiles for g in p['groups'] for snr in SELECTION_SNRS}
    probs, precision = {}, []
    for row in body_rows:
        key = row['phy_key'], row['snr_db']
        require(key in required_keys and key not in probs, 'Unknown/pressure/duplicate BLER key')
        probs[key] = probability(row)
        precision.append({k: row[k] for k in ('phy_key', 'snr_db', 'n_blocks', 'n_correct', 'n_reject',
             'n_undetected', 'p_correct_ci', 'p_reject_ci', 'p_undetected_ci')})
    require(set(probs) == required_keys, 'Missing BLER; no interpolation or zero/one fill')
    head = {}
    for row in header_rows:
        require(row['snr_db'] in SELECTION_SNRS and row['snr_db'] not in head, 'Header SNR mismatch')
        head[row['snr_db']] = probability(row)
    require(set(head) == set(SELECTION_SNRS), 'Missing original header evidence')
    quality, gray = clean_join(profiles, source_ids, historical_clean, new_clean, gray_rows)
    ranks, nominated = [], []
    for hat in contract.HATS:
        profile_scores = []
        for profile in profiles:
            per_source = [contract.design_score(
                drop_proxy(profile, s, hat, probs, head, quality, gray),
                drop_proxy(profile, s, hat - 3, probs, head, quality, gray)) for s in source_ids]
            score = math.fsum(per_source) / len(source_ids)
            memberships = ['S'] if profile['G'] == 1 else profile['family_memberships']
            profile_scores.append((profile, score, memberships))
        for arm, cap in (('U', 3), ('S', 3), ('EEP', 2)):
            pool = [(p, j) for p, j, families in profile_scores if arm in families]
            require(pool, 'Missing candidate family')
            pool.sort(key=lambda item: (-item[1], item[0]['wire_key'], item[0]['candidate_id']))
            for rank, (p, score) in enumerate(pool, 1):
                ranks.append(dict(snr_hat=hat, arm=arm, rank=rank, profile_id=p['profile_id'],
                    candidate_id=p['candidate_id'], physical_key=p['wire_key'], J_DROP_proxy=score,
                    selected=rank <= cap))
                if rank <= cap:
                    nominated.append(dict(snr_hat=hat, arm=arm, profile_id=p['profile_id'],
                        candidate_id=p['candidate_id'], physical_key=p['wire_key'],
                        receiver_variants=['KEEP', 'DROP'] if arm == 'S' else ['DROP']))
        for arm in ('S_nom', 'S_back3'):
            rows = [a for a in anchors if a['hat_snr_db'] == hat and a['method'] == arm]
            require(len(rows) == 1, 'Missing mandatory frozen anchor')
            anchor = rows[0]; p = by_id[anchor['profile_id']]
            require(p['G'] == 1 and p['wire_key'] == anchor['wire_key']
                and (anchor['candidate_id'] == p['candidate_id'] or
                     anchor['candidate_id'] in p.get('alias_candidate_ids', [])), 'Anchor wire/candidate identity changed')
            nominated.append(dict(snr_hat=hat, arm=arm, profile_id=p['profile_id'],
                candidate_id=p['candidate_id'], physical_key=p['wire_key'], receiver_variants=['KEEP']))
    budget = contract.validate_shortlist(nominated)
    return dict(status='R0_NOMINATION_ARITHMETIC_COMPLETE_NOT_REGISTERED_R1', rule_id=RULE_ID,
        rows=nominated, ranking=ranks, precision=precision, budget=budget,
        selection_uses_holdout=False, scope='original calibration200, DROP proxy only',
        global_best_claim=False, KEEP_failed_quality_inferred=False,
        requires_authenticated_actual_stage_closure=True, GO_decision='NOT_RUN', R1='NOT_RUN', R2='NOT_RUN')


def prepare(root, out):
    root, out = Path(root), Path(out)
    O = root / 'results/layered_mismatch_raw_20261013'
    paths = dict(catalogue=O/'candidates_cpu_v1/catalogue.json', groups=O/'candidates_cpu_v1/groups_by_phy_key.json',
        matched=O/'candidates_cpu_v1/coverage_v2/matched_bler_rows.json', coverage=O/'candidates_cpu_v1/coverage_v2/coverage_rows.json',
        gap=O/'R0_GAP_DECISION.md', source=O/'r0_reuse/cal200_sources.json', clean_missing=O/'r0_reuse/clean_quality_missing_pairs.json',
        old_clean=O/'r0_reuse/historical_raw_intact_cal200.csv', gray=O/'r0_reuse/historical_clean_coarse_cal200.csv',
        reuse_audit=O/'r0_reuse/final_audit.json', anchors=O/'candidates_cpu_v1/anchors.json',
        coverage_report=O/'candidates_cpu_v1/coverage_v2/coverage_report.json')
    missing, old = exact_keys(read(paths['catalogue']), read(paths['groups']), read(paths['matched']), read(paths['coverage']))
    require(len(missing) * 64 == SUPPLEMENT_CAPS['body_trials'], 'Supplement cap mismatch')
    write_new(out/'screening_missing_bler_keys.json', missing)
    write_new(out/'screening_historical_bler.json', old)
    headers = [x['row'] for x in read(paths['coverage_report'])['matched_header_rows'] if x['row']['snr_db'] in SELECTION_SNRS]
    require(len(headers) == 3, 'Original selected header rows missing')
    write_new(out/'screening_historical_headers.json', headers)
    frozen = dict(status='FROZEN_RULE_BEFORE_NEW_SCREENING_RESULTS', rule_id=RULE_ID,
        input_bindings={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()},
        source_code_sha256=sha(__file__), contract_sha256=sha(Path(__file__).with_name('contract.py')),
        source_count=200, catalogue_profiles=689, single_profiles=433, U_profiles=201, EEP_profiles=55,
        exact_body_groups=687, selection_snrs=list(SELECTION_SNRS), historical_selection_points=672,
        missing_selection_points=1389, supplement_caps=SUPPLEMENT_CAPS,
        source_embedding_accounting='200 source DINO embeddings are preprocessing for the 5400 newly scored images; counted separately, not 200 extra VAR reconstructions.',
        fixed_probability='n_correct/n_blocks; use archived counts and intervals unchanged, new missing points exactly64, Wilson95% published separately',
        proxy_formula_single='ph*p1*Q(m,K)+(1-ph*p1)*Qgray',
        proxy_formula_grouped='ph*p1*(p2*Q(m,K)+(1-p2)*Q(j,0))+(1-ph*p1)*Qgray',
        proxy_false_accept='Header/body false accepts are approximated as non-correct outcomes, not scored as actual received wrong tokens. This approximation is not a guaranteed lower bound; real R1 retains every false accept and failure.',
        independence='conditional independent group correctness; IID scrambled payload BLER is a screening approximation, not image-payload validation',
        ranking='per source .75*DROP_proxy(hat)+.25*DROP_proxy(hat-3); math.fsum then divide200; all legal candidates; no13dB',
        tie_order=['descending exact float J_DROP_proxy','ascending physical wire_key','ascending canonical candidate_id'],
        per_hat_caps=dict(U=3,S=3,EEP=2,S_nom=1,S_back3=1),
        aliases='Same physical candidate may fill multiple family roles. Keep one canonical ID, evaluate unique physical frame once, preserve all arm metadata.',
        anchors='Original PARTIAL frozen nominal/backoff are mandatory KEEP anchors; actual KEEP outcomes eligible S_robust winner. No proxy KEEP failure quality is inferred.',
        no_adaptive_trials=True, no_after_result_rule_changes=True, no_q6_exclusion=True,
        original_R1_caps=contract.R1_CAPS, original_R2_caps=contract.R2_CAPS,
        training=0, scientific_calls_by_prepare=0, scientific_execution='NOT_RUN', nominations='NOT_RUN')
    write_new(out/'screening_contract.json', frozen)
    approval = dict(status='USER_APPROVED_SEPARATE_FINITE_R0_SUPPLEMENT_NOT_EXECUTED',
        questionItemId='["request_user_input_async","call_qH1MUzhmMMYAsgnzZMIt9wUd",0]',
        question='R0 的实际缺口较大：是否增加一笔独立预筛预算，最多 88,896 次正文包试验和 5,400 次 clean VAR＋DINO 评分？原 R1/R2 上限不变；不训练，已有结果复用。',
        answer='补齐有限预筛（建议，保留完整候选的比较机会）',
        trusted_evidence='Exact human reply forwarded by root coordinator in this session.',
        gap_document=dict(path=str(paths['gap']),sha256=sha(paths['gap'])),
        caps=SUPPLEMENT_CAPS, original_R1_caps=contract.R1_CAPS, original_R2_caps=contract.R2_CAPS,
        source_preprocessing='200 source DINO embeddings counted explicitly alongside 5400 reconstruction scores; no extra VAR or channel calls.',
        frozen_rule_sha256=sha(out/'screening_contract.json'),
        exclusions=['training','new header trials','13dB BLER supplement','adaptive retry/trial count','development or holdout selection','automatic R2'])
    write_new(out/'screening_approval_addendum.json', approval)
    return dict(status='PREPARED_NOT_RUN', missing=len(missing), historical=len(old), extra_body_trials=len(missing)*64)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(prepare(a.root,a.out)))
