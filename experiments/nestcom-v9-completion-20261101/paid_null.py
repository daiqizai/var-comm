"""Additive paid-NULL support for the CURRENT 3378-profile N1024 directory.

Wire coding, old profile records and BPG's reserved IDs are unchanged.
Missing/invalid evidence raises; only exhausted valid candidates yield NULL.
"""
from pathlib import Path
import copy, hashlib, json, sys
import numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'experiments/nestcom-opt-20261030'))
from opt_common import digest
from grid_codec import endpoints, OFFSETS, endpoint
NULL = 'EXPLICIT_NULL_FIXEDFRAME'
PROTOCOL = 'NESTCOM_CURRENT_N1024_PAID_NULL_V9_20261101'

def extend_catalogue(old):
    profiles=copy.deepcopy(old['profiles']); original=copy.deepcopy(profiles)
    used={p['profile_id'] for p in profiles}; assert len(used)==len(profiles)==3378 and max(used)==3389
    reserved=set(range(3000,3012)); assert not used & reserved
    appended=[]; reused=[]; nextid=max(used)+1
    for q in (2,4,6):
        for rate in ('1/2','2/3','3/4','5/6'):
            existing=[p for p in profiles if (p['q'],p['nominal_rate'],p['family'])==(q,rate,NULL)]
            assert len(existing)<=1
            if existing: reused.append(existing[0]['profile_id']); continue
            base=next(p for p in original if (p['q'],p['nominal_rate'],p['family'],p['m'],p['K'])==(q,rate,'EC_VAR_WHOLE',4,0))
            while nextid in used|reserved:nextid+=1
            assert nextid<4096
            p={k:v for k,v in base.items() if k!='profile_key'}
            p.update(profile_id=nextid,family=NULL,m=0,K=0,token_count=0)
            p['profile_key']=digest(p); profiles.append(p);used.add(nextid);appended.append(nextid);nextid+=1
    assert profiles[:len(original)]==old['profiles'] and len(appended)==7 and len(reused)==5
    cat={k:copy.deepcopy(v) for k,v in old.items() if k!='catalogue_sha256'}
    cat.update(protocol=PROTOCOL,profiles=profiles,requested_profiles=len(profiles),parent_catalogue_sha256=old['catalogue_sha256'],null_extension=dict(appended_ids=appended,reused_ids=reused,reserved_ids=sorted(reserved),old_records_unchanged=True))
    cat['catalogue_sha256']=digest(cat)
    return cat

def choose(streams, catalogue, *, q, rate, target, grid='token'):
    m,K=endpoint(*target)
    candidates=tuple((j,0) for j in range(4,11)) if grid=='whole' else endpoints(grid)
    candidates=sorted((e for e in candidates if OFFSETS[e[0]]+e[1]<=OFFSETS[m]+K),key=lambda e:OFFSETS[e[0]]+e[1],reverse=True)
    assert candidates
    eligible=[p for p in catalogue['profiles'] if(p['q'],p['nominal_rate'])==(q,rate)]
    assert eligible and len({p['k'] for p in eligible})==1
    cap=min(eligible[0]['k']-29,8191);attempts=[]
    for a,b in candidates:
        bits=np.asarray(streams[f'm{a}_K{b}'])  # KeyError is an engineering error.
        assert bits.dtype==np.uint8 and bits.ndim==1 and len(bits)>=2 and np.isin(bits,(0,1)).all()
        attempts.append(dict(m=a,K=b,bits=len(bits),fits=len(bits)<=cap))
        if len(bits)<=cap:
            family='EC_VAR_PARTIAL' if b else 'EC_VAR_WHOLE'
            p=next(p for p in eligible if(p['family'],p['m'],p['K'])==(family,a,b))
            return p,bits.copy(),False,attempts
    p=next(p for p in eligible if p['family']==NULL)
    return p,np.zeros(2,np.uint8),True,attempts

def runtime(qualification,catalogue):
    sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
    import c_predclass_frame as cp
    phy=cp.isolated_engine(); ns=phy.pack_body.__globals__
    ns['FAMILIES']=(*ns['FAMILIES'][:-2],cp.FAMILY,cp.NULL)
    rt=phy.create_runtime(R,str(qualification));rt.catalogue=catalogue
    rt.profiles={str(p['profile_id']):p for p in catalogue['profiles']}
    return phy,rt

def received_projection(rx):
    sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'))
    from n512_phy import received_projection as project
    return project(rx)
