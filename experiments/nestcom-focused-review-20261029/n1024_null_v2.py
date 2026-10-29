"""Versioned paid NULL extension; historical N1024 protocols stay immutable.

The unchanged wire mask/session and original 372 profile records are retained.
Only newly appended IDs 372..383 mean an explicit two-zero-bit NULL payload.
This module never silently turns missing source evidence into a NULL event.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys, types
import numpy as np
PROTOCOL = 'NESTCOM_N1024_PAID_NULL_20261029_V2'
NULL = 'EXPLICIT_NULL_FIXEDFRAME'
PINS = {
 'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py': '12cad3e495fc91d8f6819b895674df08e312da3833291019b8966a2211ec8c75',
 'experiments/entropy-partial-mismatch-20261010/scripts/ep_plan.py': '7f354687caa3271ce4375f598f4d9e3ff6086df28d8be478c80835ab4ccd7862',
 'experiments/t1b-m10-calibration-20261016/tb_plan.py': '362419fa2b6718989baf49a7b4dea3b0ab5934b69b7b59b4072c9f4c23b8dc60',
}

def load(root):
    root = Path(root).resolve()
    for rel, expected in PINS.items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest() == expected, rel
    path = root/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py'
    sys.path.insert(0, str(path.parent))
    ns = dict(__file__=str(path), __name__='_n1024_null_v2_isolated')
    exec(compile(path.read_bytes(), str(path), 'exec'), ns)
    # Independent module namespace; never mutate an imported historical module.
    ns['FAMILIES'] = (*ns['FAMILIES'], NULL)
    ns['PROTOCOL'] = PROTOCOL
    tbpath = root/'experiments/t1b-m10-calibration-20261016/tb_plan.py'
    spec = importlib.util.spec_from_file_location('_null_v2_tb_plan', tbpath)
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
    return types.SimpleNamespace(**ns), tb

def extend_catalogue(phy, old):
    assert len(old['profiles']) == 372
    profiles = [dict(p) for p in old['profiles']]
    for q in (2, 4, 6):
        for rate in phy.RATES:
            base = next(p for p in profiles[:372] if
                        (p['family'],p['m'],p['K'],p['q'],p['nominal_rate']) ==
                        ('EC_VAR_WHOLE',4,0,q,rate))
            p = {k:v for k,v in base.items() if k != 'profile_key'}
            p.update(profile_id=len(profiles),family=NULL,m=0,K=0,token_count=0)
            p['profile_key'] = phy.digest(p); profiles.append(p)
    assert profiles[:372] == old['profiles']
    assert [p['profile_id'] for p in profiles] == list(range(384))
    assert all(p['N']==1024 and p['header_symbols']==68 and p['body_symbols']==956 for p in profiles)
    cat = dict(old,protocol=PROTOCOL,profiles=profiles,requested_profiles=384,
               parent_catalogue_sha256=old['catalogue_sha256'],
               status='VERSIONED_NULL_EXTENSION_TARGETED_CHECKS_REQUIRED')
    cat.pop('catalogue_sha256'); cat['catalogue_sha256'] = phy.digest(cat)
    return cat

def choose(streams, candidate, catalogue, policy_version):
    """Preserve the exact legacy/m10 fallback chain; no-fit alone returns NULL."""
    assert policy_version in ('legacy','m10')
    from ep_plan import fallback_endpoints
    q,rate = candidate['q'],candidate['nominal_rate']
    eligible = [p for p in catalogue['profiles'] if (p['q'],p['nominal_rate'])==(q,rate)]
    assert eligible
    capacity = eligible[0]['k']-29
    assert candidate['source_capacity_bits'] == capacity
    if policy_version == 'legacy':
        chain = fallback_endpoints(candidate['target_m'],candidate['target_K'])
    else:
        assert candidate['target_m']==10 and candidate['target_K']==0
        assert candidate['strategy_family'] in ('WHOLE','PARTIAL')
        sizes=(1,2,3,4,5,6,8,10,13,16)
        ends=tuple((m,k) for m in range(4,10) for k in (0,*(sizes[m]**2*j//4 for j in (1,2,3))))
        chain=((10,0),)+(tuple((m,0) for m in range(9,3,-1)) if candidate['strategy_family']=='WHOLE' else tuple(sorted(ends,key=lambda e:sum(s*s for s in sizes[:e[0]])+e[1],reverse=True)))
    attempts=[]
    for m,k in chain:
        bits=np.asarray(streams[f'm{m}_K{k}'])  # Missing evidence raises; never NULL.
        assert bits.dtype==np.uint8 and bits.ndim==1 and len(bits)>=2 and np.isin(bits,(0,1)).all()
        fits=len(bits)<=min(capacity,8191)
        attempts.append(dict(m=m,K=k,bits=len(bits),fits=fits))
        if fits:
            family='EC_VAR_PARTIAL' if k else 'EC_VAR_WHOLE'
            p=next(p for p in eligible if (p['family'],p['m'],p['K'])==(family,m,k))
            return dict(profile=p,payload=bits.copy(),tx_no_fit=False,attempts=attempts)
    p=next(p for p in eligible if p['family']==NULL)
    return dict(profile=p,payload=np.zeros(2,np.uint8),tx_no_fit=True,attempts=attempts)

def received_projection(rx):
    """RX-only decision. A received NULL must pass the same header/body checks."""
    if not rx['header']['header_ok']:return dict(status='FIXED_GRAY',reason='HEADER_REJECT')
    body=rx['body'];p=rx['rx_profile']
    if not body['crc_accepted'] or not body['parser_accepted']:
        return dict(status='FIXED_GRAY',reason=body['status'])
    if p['family']==NULL:
        return dict(status='FIXED_GRAY',reason='EXPLICIT_NULL_PROFILE' if body['payload']==[0,0] else 'INVALID_NULL_PAYLOAD')
    if p['family'] not in ('EC_VAR_WHOLE','EC_VAR_PARTIAL','EC_STATIC_WHOLE'):
        return dict(status='FIXED_GRAY',reason='UNKNOWN_FAMILY')
    return dict(status='SOURCE_INPUT',family=p['family'],m=p['m'],K=p['K'],bits=body['payload'])

def fixed_gray(projection):
    assert projection['status']=='FIXED_GRAY'
    return np.full((3,256,256),0.5,dtype=np.float32)
