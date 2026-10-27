"""Frozen formal three-SNR planning; no models, score reading or PHY calls."""
from common import digest
from c_predclass_protocol import class_bits, source_capacity
from c_predclass_frame import FAMILY, NULL, CANDIDATES

POPULATIONS = {
    'common500': dict(sources=500, noises=(6201,6202,6203),
                      namespace='ADA_PREDCLASS_COMMON500_FIXEDFRAME_V1', counter_base=6000000),
    'kodak24': dict(sources=24, noises=(2001,2002,2003),
                    namespace='ADA_PREDCLASS_KODAK24_FIXEDFRAME_V1', counter_base=9000000),
}
COUNTER_STRIDE = 4096


def build_choices(sources, profiles, population, policies, expected_sources=None):
    settings = POPULATIONS[population]
    count = settings['sources'] if expected_sources is None else expected_sources
    assert type(count) is int and 1 <= count <= settings['sources']
    assert len(sources) == count and len({s['source_id'] for s in sources}) == count
    indices = [s['source_index'] for s in sources]
    assert len(set(indices)) == count
    assert all(type(i) is int and 0 <= i < settings['sources'] for i in indices)
    assert set(policies) == {'4','10','19'}
    selected = []
    for snr in (4,10,19):
        matches = [(s,q,rate) for s,q,rate in CANDIDATES
                   if s == snr and 'p256_q%d_r%s'%(q,rate.replace('/','-')) == policies[str(snr)]]
        assert len(matches) == 1, 'One previously calibrated MCS per SNR'
        selected.extend(matches)
    ids = [p['profile_id'] for p in profiles]
    assert len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < 4096 for i in ids)
    choices = []
    for s in sorted(sources, key=lambda v:v['source_index']):
        class_bits(s['predicted_class'])
        lengths = s['lengths']
        assert len(lengths) == 256 and {v['prefix'] for v in lengths} == set(range(1,257))
        assert all(type(v['prefix']) is int and type(v['bits']) is int and v['bits'] >= 2 for v in lengths)
        sizes = {v['prefix']:v['bits'] for v in lengths}
        for snr,q,rate in selected:
            pp = [p for p in profiles if p['q']==q and p['nominal_rate']==rate and p['family'] in (FAMILY,NULL)]
            positive = {p['prefix_count']:p for p in pp if p['family']==FAMILY}
            nulls = [p for p in pp if p['family']==NULL]
            assert len(pp)==257 and len(nulls)==1 and set(positive)==set(range(1,257))
            assert len({p['k'] for p in pp})==1
            capacity = source_capacity(pp[0]['k'])
            prefix = max((i for i in range(1,257) if sizes[i]<=capacity),default=0)
            paper = max((i for i in range(32,257,32) if sizes[i]<=capacity),default=0)
            profile = positive[prefix] if prefix else nulls[0]
            assert profile['m']==prefix and profile['K']==0
            counter = settings['counter_base']+COUNTER_STRIDE*s['source_index']+profile['profile_id']
            cid = policies[str(snr)]
            choice_key = digest([settings['namespace'],s['source_id'],snr,cid])
            wave_key = digest([settings['namespace'],s['source_index'],counter,profile['profile_id'],s['streams']['sha256'],s['predicted_class']])
            choices.append(dict(key=choice_key,source_id=s['source_id'],source_index=s['source_index'],
                population=population,snr_db=snr,candidate_id=cid,q=q,rate=rate,target_prefix=256,
                prefix=prefix,profile_id=profile['profile_id'],counter=counter,
                arithmetic_capacity_bits=capacity,arithmetic_bits=sizes[prefix] if prefix else 0,
                paid_class_bits=10 if prefix else 0,container_bits=10+sizes[prefix] if prefix else 2,
                length_checks=257-prefix if prefix else 256,explicit_null=prefix==0,
                paper32_prefix=paper,paper32_arithmetic_bits=sizes[paper] if paper else 0,waveform_key=wave_key))
    assert len(choices)==count*3 and len({v['key'] for v in choices})==len(choices)
    return choices


def events(choices,population):
    settings=POPULATIONS[population];rows=[]
    for c in choices:
        assert c['population']==population
        for noise in settings['noises']:
            key=digest([settings['namespace'],c['source_id'],c['snr_db'],c['candidate_id'],noise])
            rows.append(dict(key=key,choice_key=c['key'],source_index=c['source_index'],source_id=c['source_id'],
                population=population,snr_db=c['snr_db'],candidate_id=c['candidate_id'],noise_seed=noise,counter=c['counter']))
    assert len(rows)<=settings['sources']*9 and len({v['key'] for v in rows})==len(rows)
    return rows
