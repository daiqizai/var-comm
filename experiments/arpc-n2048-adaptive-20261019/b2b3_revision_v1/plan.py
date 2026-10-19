"""The user's nine-cell amendment. Pure metadata; never imports PHY or models."""
from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_plan import NEST_ENDPOINTS, RATES, choose, layouts, policies

SCHEMA = 'N2048_B2B3_USER_REVISION_20261019_V1'
B2_CELLS = (
    (0, 4, 'nest_t24_q2_r1-2'),
    (0, 4, 'nest_t24_q2_r2-3'),
    (0, 10, 'nest_t24_q2_r2-3'),
    (0, 10, 'nest_t24_q2_r3-4'),
    (0, 10, 'nest_t24_q2_r5-6'),
    (0, 10, 'nest_t24_q4_r1-2'),
    (0, 10, 'nest_t24_q4_r2-3'),
    (1, 4, 'arpc_t7_q2_r2-3'),
    (1, 10, 'arpc_t7_q4_r2-3'),
)
POLICIES = {p['policy_id']: p for codec in (0, 1) for p in policies(codec)}
LAYOUTS = {p['layout_id']: p for p in layouts()}
SEEDS = {'B2': (4101, 4102, 4103), 'B3': (6201, 6202, 6203)}
POPULATIONS = {'B2': 'cal1000', 'B3': 'external_common100'}
SOURCE_COUNTS = {'B2': 1000, 'B3': 100}
CAPS = {
    'B2': dict(logical_objects=27000, subframes=54000, header=54000,
               body=54000, decoder_attempts=108000),
    'B3': dict(logical_objects=1200, subframes=2400, header=2400,
               body=2400, decoder_attempts=4800),
}
TIE_RULE = 'Maximum source-mean DINO; exact ties: lower q, lower nominal rate, higher target, lexical policy ID.'


def cell_rows():
    return [dict(codec=c, snr_db=s, policy_id=p) for c, s, p in B2_CELLS]


def tie_key(row):
    p = POLICIES[row['policy_id']]
    g = LAYOUTS[p['layout_id']]
    v = row['calibration_mean_DINO']
    assert type(v) in (int, float) and math.isfinite(v)
    return (-v, g['q'], RATES.index(g['nominal_rate']), -p['target'], p['policy_id'])


def select_four(candidate_means):
    """Consume completed B2 source means only, without any new bootstrap."""
    assert len(candidate_means) == 9
    keys = [(r['codec'], r['snr_db'], r['policy_id']) for r in candidate_means]
    assert len(set(keys)) == 9 and set(keys) == set(B2_CELLS)
    for row in candidate_means:
        assert row['source_mean_count'] == 1000 and row['noise_per_source'] == 3
        tie_key(row)
    out = []
    for c in (0, 1):
        for s in (4, 10):
            group = [r for r in candidate_means if r['codec'] == c and r['snr_db'] == s]
            winner = min(group, key=tie_key)
            out.append(dict(codec=c, snr_db=s, policy_id=winner['policy_id'],
                            calibration_mean_DINO=winner['calibration_mean_DINO'],
                            tie_break=TIE_RULE))
    return out


def frozen_cells(selection):
    assert selection['schema'] == 'N2048_B2_DINO_FOUR_POLICIES_V1'
    assert selection['status'] == 'FOUR_POLICIES_FROZEN'
    assert selection['metric'] == 'dinov2_vitl14_cosine'
    expected = select_four(selection['candidate_means'])
    assert selection['rows'] == expected
    assert selection['no_additional_candidates'] is True
    assert selection['calibration_bootstrap_calls'] == 0
    return [dict(codec=r['codec'], snr_db=r['snr_db'], policy_id=r['policy_id'])
            for r in expected]


def asset_key(source_id, codec, endpoint):
    import hashlib
    raw = f'{source_id}|{codec}|{endpoint}'.encode()
    return hashlib.sha256(raw).hexdigest()


def make_jobs(phase, cells, sources, catalogue):
    """Length-only fallback from all original endpoints. No quality inputs."""
    assert phase in CAPS
    if phase == 'B2':
        assert cells == cell_rows()
    else:
        assert len(cells) == 4
        assert {(c['codec'], c['snr_db']) for c in cells} == {(c, s) for c in (0, 1) for s in (4, 10)}
        assert all((c['codec'], c['snr_db'], c['policy_id']) in B2_CELLS for c in cells)
    assert len(sources) == SOURCE_COUNTS[phase]
    ids = [s['source_id'] for s in sources]
    assert len(set(ids)) == len(ids)
    index = {}
    for a in catalogue:
        if a['population'] != POPULATIONS[phase] or a['source_id'] not in set(ids):
            continue
        key = (a['source_id'], a['codec'])
        assert a['endpoint'] not in index.setdefault(key, {})
        index[key][a['endpoint']] = a
    for sid in ids:
        assert set(index[sid, 0]) == set(range(25))
        assert set(index[sid, 1]) == set(range(1, 8))
    jobs, assets, fallback_rows = [], {}, []
    for cell in cells:
        pol = POLICIES[cell['policy_id']]
        assert pol['codec'] == cell['codec']
        g = LAYOUTS[pol['layout_id']]
        for source in sources:
            sid = source['source_id']
            streams = index[sid, cell['codec']]
            for a in streams.values():
                assert a['source_index'] == source['source_index']
                assert a['source']['sha256'] == source['source']['sha256']
                assert type(a['bit_length']) is int and a['bit_length'] > 0
            endpoint = choose(streams, cell['codec'], pol['target'], g)
            a = None if endpoint is None else streams[endpoint]
            ak = None if a is None else asset_key(sid, cell['codec'], endpoint)
            if ak is not None:
                if ak in assets:
                    assert assets[ak] == a
                assets[ak] = a
            fallback_rows.append(dict(**cell, source_id=sid, source_index=source['source_index'],
                target=pol['target'], actual_endpoint=endpoint, layout_id=g['layout_id'],
                source_no_fit=a is None, capacity_bits=g['object_bits'],
                object_bits=0 if a is None else a['bit_length'], asset_key=ak))
            for seed in SEEDS[phase]:
                i = len(jobs)
                jobs.append(dict(key=f'{phase.lower()}_{i:05d}', pair_id=(1000000 if phase == 'B2' else 2000000) + i,
                    **cell, source_id=sid, source_index=source['source_index'], noise_seed=seed,
                    layout_id=g['layout_id'], target=pol['target'], actual_endpoint=endpoint,
                    source_no_fit=a is None, asset_key=ak))
    assert len(jobs) == CAPS[phase]['logical_objects']
    assert len({j['key'] for j in jobs}) == len(jobs)
    assert len({j['pair_id'] for j in jobs}) == len(jobs)
    return jobs, assets, fallback_rows


def public_job(key, snr_db, counters, observation):
    import re
    assert isinstance(key, str) and re.fullmatch(r'b[23]_[0-9]{5}', key)
    assert type(snr_db) is int and snr_db in (4, 10)
    assert len(counters) == 2 and counters[1] == counters[0] + 1
    assert all(type(c) is int and 0 <= c < 2**32 for c in counters)
    assert set(observation) == {'path', 'bytes', 'sha256'}
    assert type(observation['bytes']) is int and observation['bytes'] > 0
    assert isinstance(observation['sha256'], str) and re.fullmatch('[0-9a-f]{64}', observation['sha256'])
    return dict(key=key, snr_db=snr_db, counters=counters, observation=observation)


def validate_public_job(j):
    assert set(j) == {'key', 'snr_db', 'counters', 'observation'}
    assert j == public_job(j['key'], j['snr_db'], j['counters'], j['observation'])

