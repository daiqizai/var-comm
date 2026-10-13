"""Read-only finite-calibration reduction. No models, PHY, bootstrap or test data."""
from __future__ import annotations
from collections import defaultdict
import math
from contract import (HATS, CAL_SEEDS, SELECTION_METRIC, design_score,
                      calibration_gate, require, validate_shortlist)

R1_METRICS = ('psnr_db', 'lpips_alex', SELECTION_METRIC)
ARM_NAMES = {'U': 'U_two', 'S': 'S_robust', 'EEP': 'E_two',
             'S_nom': 'S_nom', 'S_back3': 'S_back3'}


def validate_and_reduce(rows, sources, nominations):
    """Caller must additionally authenticate actual PHY/visual/scoring exit0."""
    require(len(sources) == 200 and len(set(sources)) == 200, 'Fixed cal200 required')
    validate_shortlist(nominations)
    expected = {}
    for c in nominations:
        key = (int(c['snr_hat']), str(c['candidate_id']))
        # A physical anchor may occur in multiple arms; its variants are merged.
        old = expected.setdefault(key, dict(physical_key=c['physical_key'], rules=set(), arms=set()))
        require(old['physical_key'] == c['physical_key'], 'Ambiguous candidate identity')
        old['rules'].update(c['receiver_variants']); old['arms'].add(c['arm'])
    seen = set(); buckets = defaultdict(list); observations = {}
    for r in rows:
        sid = str(r['source_id']); hat = int(r['snr_hat']); true = int(r['snr_true'])
        seed = int(r['noise_seed']); cid = str(r['candidate_id']); rule = r['receiver_rule']
        require(sid in sources and (hat, cid) in expected, 'Foreign source/candidate')
        e = expected[(hat, cid)]
        require(rule in e['rules'] and r['physical_key'] == e['physical_key'],
                'Candidate or receiver permission changed')
        require(true in (hat-3, hat, hat+3) and seed in CAL_SEEDS, 'Foreign condition/noise')
        key = (sid, hat, true, seed, cid, rule)
        require(key not in seen, 'Duplicate logical scoring row'); seen.add(key)
        require(r['execution_status'] == 'COMPLETE', 'Failure of software is not fixed-gray success')
        require(r['source_truth_supplied_to_RX'] is False, 'Receiver truth access')
        require(r['observation_key'] and r['actual_rx_key'] and r['reconstruction_sha256'],
                'Actual receive and reconstruction identities required')
        physical = (sid, hat, true, seed, r['physical_key'])
        received = (r['observation_key'], r['actual_rx_key'])
        require(physical not in observations or observations[physical] == received,
                'KEEP/DROP or aliases did not share the actual physical observation')
        observations[physical] = received
        for metric in R1_METRICS:
            value = float(r[metric]); require(math.isfinite(value), 'Missing/nonfinite quality')
            buckets[(sid, hat, true, cid, rule, metric)].append(value)
    count = sum(len(x['rules']) for x in expected.values()) * 200 * 3 * 3
    require(len(seen) == count and count <= 46800, 'Incomplete or excessive calibration grid')
    require(all(len(v) == 3 for v in buckets.values()), 'Every source/condition needs three noises')
    means = {key: math.fsum(values)/3 for key, values in buckets.items()}
    source_scores = {}
    for (hat, cid), entry in expected.items():
        for rule in entry['rules']:
            values = [design_score(means[(sid,hat,hat,cid,rule,SELECTION_METRIC)],
                                    means[(sid,hat,hat-3,cid,rule,SELECTION_METRIC)])
                      for sid in sources]
            source_scores[(hat,cid,rule)] = values
    return means, source_scores, expected


def select_from_complete_rows(rows, sources, nominations):
    means, source_scores, expected = validate_and_reduce(rows, sources, nominations)
    winners = {}; ranking = []
    for hat in HATS:
        for arm, display in ARM_NAMES.items():
            choices = []
            eligible = {'S', 'S_nom', 'S_back3'} if arm == 'S' else {arm}
            added = set()
            for nomination in nominations:
                if nomination['snr_hat'] != hat or nomination['arm'] not in eligible: continue
                cid = nomination['candidate_id']
                for rule in nomination['receiver_variants']:
                    if (cid, rule) in added: continue
                    added.add((cid, rule))
                    values = source_scores[(hat,cid,rule)]
                    mean = math.fsum(values)/200
                    # Exact ties retain nominal/backoff actions first, then KEEP.
                    anchor_rank = 0 if 'S_nom' in expected[(hat,cid)]['arms'] else (
                        1 if 'S_back3' in expected[(hat,cid)]['arms'] else 2)
                    tie_key = (anchor_rank, cid, 0 if rule == 'KEEP' else 1)
                    choices.append((mean,tie_key,nomination,rule))
            require(choices, 'Missing arm at a registered estimate')
            choices.sort(key=lambda x:(-x[0],x[1]))
            score, _, n, rule = choices[0]
            winners[(display,hat)] = dict(method=display,snr_hat=hat,
                candidate_id=n['candidate_id'],physical_key=n['physical_key'],
                receiver_rule=rule,cal200_DINO_J=score,
                selection_metric=SELECTION_METRIC,selection_p=.25,
                pressure_e_minus3_used_for_selection=False,
                convnext_used_for_selection=False)
            ranking.extend(dict(method=display,snr_hat=hat,rank=i+1,candidate_id=x[2]['candidate_id'],
                                receiver_rule=x[3],DINO_J=x[0]) for i,x in enumerate(choices))
    j = {method:{hat:winners[(method,hat)]['cal200_DINO_J'] for hat in HATS}
         for method in ('U_two','S_robust','S_back3')}
    return dict(winners=list(winners.values()), ranking=ranking, gate=calibration_gate(j),
                source_count=200, noise_count=3, logical_rows=len(rows),
                bootstrap_calls=0, metrics_used_for_selection=[SELECTION_METRIC],
                requires_authenticated_actual_stage_closure=True)
