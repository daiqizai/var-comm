"""Pure calibration integrity checks; no model, PHY or random sampling."""
import ast
import copy
import hashlib
import math
import statistics
import numpy as np

METRIC = 'dinov2_vitl14_cosine'
FREEZE_SHA = 'f36101b33d37da0936e138b62cc763e3f28e6c3edd13e296c0dc7a29a5e3ffd6'
CHOICES = {4: ('p256_q2_r1-2', 'p256_q2_r2-3'),
           10: ('p256_q4_r2-3', 'p256_q4_r3-4'),
           19: ('p256_q4_r2-3', 'p256_q6_r5-6')}
RUNTIME_KEYS = ('numeric_flags', 'float_dtype', 'AMP', 'batch_size',
                'device', 'host', 'torch', 'numpy', 'python')
PRO_HOST = 'aibox-r59688baa77c-57c794f5c6-44lm7'


def check_rx_closed(wait, audit, observation):
    # RX PIDs belong to PRO; checking them in MAIN /proc is not evidence.
    assert type(wait['returncode']) is int and wait['returncode'] == 0
    assert wait['actual_wait'] is True
    assert not any(wait.get(k,False) for k in ('timed_out','terminated','killed'))
    assert wait.get('metadata_error') is None
    assert audit['status'] == 'PASS' and not audit['violations']
    assert observation['host'] == PRO_HOST
    expected = dict(worker=wait['process'], owner=wait['parent_process'],
                    auditor=audit['process'], audit_parent=audit['parent_process'])
    assert set(observation['processes']) == set(expected)
    for role, identity in expected.items():
        record = observation['processes'][role]
        assert record['expected'] == identity and record['same_generation_live'] is False
    return True


def image(a):
    assert isinstance(a, np.ndarray) and a.dtype == np.float32
    assert a.shape == (3, 256, 256) and np.isfinite(a).all()
    assert (a >= 0).all() and (a <= 1).all(), 'No hidden clipping/rescaling'
    return a


def check_identity(previous, current_metadata, current_runtime):
    before = copy.deepcopy(previous['DINO_L_metadata'])
    after = copy.deepcopy(current_metadata)
    # Usage attribution changes; models, transform, packages and numerics do not.
    assert 'metric_selection_usage' in before and 'metric_selection_usage' in after
    before.pop('metric_selection_usage'); after.pop('metric_selection_usage')
    assert before == after, 'DINO model/preprocessing/numeric metadata changed'
    assert set(after['metrics']) == {'dinov2_vitl14'}
    for key in RUNTIME_KEYS:
        assert previous[key] == current_runtime[key], ('DINO runtime changed', key)
    return True


def validate_grid(rows, population, expected_sources=1000):
    assert type(expected_sources) is int and 1 <= expected_sources <= 1000
    assert len(population) == expected_sources and len(rows) == expected_sources * 18
    assert len({s['source_id'] for s in population}) == expected_sources
    assert len({s['source_index'] for s in population}) == expected_sources
    ids = {s['source_id']: s['source_index'] for s in population}
    seen = set()
    for r in rows:
        sid = r['source_id']; idx = int(r['source_index'])
        snr = int(r['snr_db']); noise = int(r['noise_seed']); cid = r['candidate_id']
        assert sid in ids and idx == ids[sid]
        assert snr in CHOICES and cid in CHOICES[snr] and noise in (4101,4102,4103)
        key = (sid, snr, cid, noise); assert key not in seen; seen.add(key)
        assert math.isfinite(float(r[METRIC]))
    assert len(seen) == expected_sources * 18
    return ids


def freeze_original(source_bytes, rows, population, expected_sources=1000):
    ids = validate_grid(rows, population, expected_sources)
    assert hashlib.sha256(source_bytes).hexdigest() == FREEZE_SHA
    tree = ast.parse(source_bytes)
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='aggregate']
    assert len(nodes) == 1
    ns = dict(statistics=statistics, METRICS=(METRIC,))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<SHA-pinned original aggregate>','exec'),ns)
    decisions = {str(k):dict(proposed_candidates=list(v)) for k,v in CHOICES.items()}
    means, summaries, policies = ns['aggregate'](rows,decisions,set(ids))
    assert len(means)==expected_sources*6 and len(summaries)==6 and set(policies)=={'4','10','19'}
    return means,summaries,policies
