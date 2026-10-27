"""Finite six-choice calibration plan; no image/model/PHY execution."""
from common import digest
from c_predclass_protocol import class_bits, source_capacity
from c_predclass_frame import FAMILY, NULL, CANDIDATES

NOISES = (4101, 4102, 4103)
NAMESPACE = 'ADA_PREDCLASS_CAL1000_FIXEDFRAME_V1'
COUNTER_BASE = 4000000
COUNTER_STRIDE = 4096


def build_choices(sources, profiles, expected_sources=1000):
    assert type(expected_sources) is int and 1 <= expected_sources <= 1000
    assert len(sources) == expected_sources
    assert len({s['source_id'] for s in sources}) == expected_sources
    indices = [s['source_index'] for s in sources]
    assert len(set(indices)) == expected_sources
    assert all(type(i) is int and 0 <= i < 1000 for i in indices)
    ids = [p['profile_id'] for p in profiles]
    assert len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < 4096 for i in ids)
    choices = []
    for s in sorted(sources, key=lambda v: v['source_index']):
        class_bits(s['predicted_class'])
        lengths = s['lengths']
        assert len(lengths) == 256
        assert {v['prefix'] for v in lengths} == set(range(1, 257))
        assert all(type(v['prefix']) is int and type(v['bits']) is int and v['bits'] >= 2 for v in lengths)
        sizes = {v['prefix']: v['bits'] for v in lengths}
        for snr, q, rate in CANDIDATES:
            pp = [p for p in profiles if p['q'] == q and p['nominal_rate'] == rate and p['family'] in (FAMILY, NULL)]
            positive = {p['prefix_count']: p for p in pp if p['family'] == FAMILY}
            nulls = [p for p in pp if p['family'] == NULL]
            assert len(pp) == 257 and len(nulls) == 1 and set(positive) == set(range(1, 257))
            assert len({p['k'] for p in pp}) == 1
            capacity = source_capacity(pp[0]['k'])
            prefix = max((i for i in range(1, 257) if sizes[i] <= capacity), default=0)
            paper = max((i for i in range(32, 257, 32) if sizes[i] <= capacity), default=0)
            profile = positive[prefix] if prefix else nulls[0]
            assert profile['m'] == prefix and profile['K'] == 0
            counter = COUNTER_BASE + COUNTER_STRIDE * s['source_index'] + profile['profile_id']
            cid = 'p256_q%d_r%s' % (q, rate.replace('/', '-'))
            choice_key = digest([NAMESPACE, s['source_id'], snr, cid])
            wave_key = digest([NAMESPACE, s['source_index'], counter, profile['profile_id'], s['streams']['sha256'], s['predicted_class']])
            choices.append(dict(key=choice_key, source_id=s['source_id'], source_index=s['source_index'],
                snr_db=snr, candidate_id=cid, q=q, rate=rate, target_prefix=256,
                prefix=prefix, profile_id=profile['profile_id'], counter=counter,
                arithmetic_capacity_bits=capacity, arithmetic_bits=sizes[prefix] if prefix else 0,
                paid_class_bits=10 if prefix else 0, container_bits=10 + sizes[prefix] if prefix else 2,
                length_checks=257-prefix if prefix else 256, explicit_null=prefix == 0,
                paper32_prefix=paper, paper32_arithmetic_bits=sizes[paper] if paper else 0,
                waveform_key=wave_key))
    assert len(choices) == expected_sources * 6
    assert len({v['key'] for v in choices}) == len(choices)
    return choices


def events(choices):
    rows = []
    for choice in choices:
        for noise in NOISES:
            key = digest([NAMESPACE, choice['source_id'], choice['snr_db'], choice['candidate_id'], noise])
            rows.append(dict(key=key, choice_key=choice['key'], source_index=choice['source_index'],
                source_id=choice['source_id'], snr_db=choice['snr_db'], candidate_id=choice['candidate_id'],
                noise_seed=noise, counter=choice['counter']))
    assert len(rows) <= 18000 and len({v['key'] for v in rows}) == len(rows)
    return rows
