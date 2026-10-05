"""Complete, calibration-blind H resource catalogue (no PHY execution)."""
from __future__ import annotations
from fractions import Fraction
import hashlib
import json

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
RATES = ('1/2', '2/3', '3/4', '5/6')
HEADER_SYMBOLS = 68
BODY_SYMBOLS = 956
LENGTH_BITS = 13
CRC_BITS = 16
PROTOCOL = 'CONTENT-REAL-64QAM-H-20261006-V1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def normalize_state(m, K=0):
    require(type(m) is int and type(K) is int and 0 <= m <= len(SIZES), 'Invalid state integers')
    require(K >= 0 and (K == 0 if m == len(SIZES) else K <= SIZES[m] ** 2), 'Invalid partial scale count')
    return (m + 1, 0) if m < len(SIZES) and K == SIZES[m] ** 2 else (m, K)


def token_count(m, K=0):
    m, K = normalize_state(m, K)
    return sum(x*x for x in SIZES[:m]) + K


def bucket(q, rate):
    require(q in (4, 6) and rate in RATES, 'Unregistered H modulation/rate')
    n = BODY_SYMBOLS * q
    r = Fraction(rate)
    k = n * r.numerator // r.denominator
    result = dict(modulation=f'{2**q}QAM', q=q, nominal_rate=rate, n=n, k=k,
                  body_symbols=BODY_SYMBOLS, header_symbols=HEADER_SYMBOLS,
                  total_symbols=1024, length_bits=LENGTH_BITS, crc_bits=CRC_BITS,
                  source_capacity=k-LENGTH_BITS-CRC_BITS, effective_rate=k/n)
    result['resource_id'] = digest(result)
    return result


def all_buckets(backend=None):
    rows = []
    for q in (4, 6):
        for rate in RATES:
            row = bucket(q, rate)
            if backend is None:
                row.update(admission='RESOURCE_ONLY', layout=None)
            else:
                # Only explicit UnsupportedConfiguration is an admission rejection.
                from h64_backend import UnsupportedConfiguration
                try:
                    row.update(layout=backend.plan(row['k'], row['n'], q), admission='ADMITTED')
                except UnsupportedConfiguration as exc:
                    row.update(layout=None, admission='UNAVAILABLE', rejection=str(exc))
            rows.append(row)
    return rows


def canonical_states():
    """One state for every nonempty token count; full K becomes the next m."""
    for m in range(len(SIZES)):
        for K in range(SIZES[m] ** 2):
            if token_count(m, K):
                yield m, K
    yield len(SIZES), 0


def make_profile(b, m, K, mode, families):
    m, K = normalize_state(m, K)
    require(mode in ('raw', 'arithmetic'), 'Unknown source mode')
    require(mode == 'raw' or (K == 0 and m in (6, 7, 8, 9)), 'Arithmetic scope is whole m6..m9')
    length = 12 * token_count(m, K)
    require(length > 0, 'Empty source profile')
    require(mode != 'raw' or length <= b['source_capacity'], 'Raw state exceeds capacity')
    wire = dict(m=m, K=K, mode=mode, q=b['q'], k=b['k'], n=b['n'],
                nominal_rate=b['nominal_rate'], length_bits=13, crc_bits=16,
                ordering='raster', receiver='CRC_DROP_CANONICAL_V1',
                layout_id=(b.get('layout') or {}).get('layout_id'))
    return dict(**wire, profile_key=digest(wire), raw_bits=length,
                source_capacity=b['source_capacity'], body_symbols=956,
                header_symbols=68, total_symbols=1024, families=sorted(set(families)),
                resource_id=b['resource_id'], admission=b['admission'])


def catalogue(buckets):
    """Do not shortlist K here. Resource legality precedes any calibration data."""
    require(len(buckets) == 8 and len({(b['q'], b['nominal_rate']) for b in buckets}) == 8,
            'All eight resource buckets, including explicit rejections, are required')
    profiles = {}
    excluded = []
    for b in buckets:
        if b['admission'] == 'UNAVAILABLE':
            continue
        def add(m, K, mode, family):
            p = make_profile(b, m, K, mode, [family])
            old = profiles.get(p['profile_key'])
            if old:
                old['families'] = sorted(set(old['families'] + p['families']))
            else:
                profiles[p['profile_key']] = p
        for m in (6, 7, 8, 9):
            for mode, suffix in (('raw', 'R'), ('arithmetic', 'A')):
                if mode == 'raw' and 12*token_count(m) > b['source_capacity']:
                    excluded.append(dict(resource_id=b['resource_id'], m=m, K=0, mode=mode,
                                         reason='RAW_SOURCE_EXCEEDS_K_MINUS_29'))
                else:
                    add(m, 0, mode, f'H{2**b["q"]}-{suffix}')
        if b['q'] == 6:
            for m, K in canonical_states():
                if 12*token_count(m, K) <= b['source_capacity']:
                    add(m, K, 'raw', 'H64-RAW-COMPLETE-STATE-CONTROL')
    ordered = sorted(profiles.values(), key=lambda p: (p['q'], Fraction(p['nominal_rate']), p['m'], p['K'], p['mode']))
    require(len(ordered) <= 4096, 'Public profile catalogue exceeds the paid 12-bit header')
    for pid, p in enumerate(ordered):
        p['profile_id'] = pid
    return dict(protocol=PROTOCOL, scientific_scope='RESOURCE_ENUMERATION_ONLY',
                buckets=buckets, profiles=ordered, excluded_whole_profiles=excluded,
                profile_count=len(ordered), header_id_bits=12,
                raw64_scope='ALL_NONEMPTY_CANONICAL_INTEGER_M_K_STATES_THAT_FIT_EACH_ADMITTED_BUCKET')
