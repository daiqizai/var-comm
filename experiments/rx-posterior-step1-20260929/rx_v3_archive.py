"""Lossless, inspectable plain-JSON storage for sparse reliability bins; no binary archives."""
import json
MARK='__rx_sparse_bins_v1__'
KEYS=['count','confidence_sum','correct_sum']
def canonical_empty(b):
    return list(b)==KEYS and type(b['count']) is int and b['count']==0 and type(b['correct_sum']) is int and b['correct_sum']==0 and type(b['confidence_sum']) is float and repr(b['confidence_sum'])=='0.0'
def pack(value):
    if isinstance(value,list):return [pack(x) for x in value]
    if not isinstance(value,dict):return value
    assert MARK not in value,'reserved encoding marker'
    out={}
    for k,v in value.items():
        if k=='bins':
            assert isinstance(v,list) and len(v)==15 and all(isinstance(x,dict) and list(x)==KEYS for x in v)
            out[k]={MARK:[[j,b['count'],b['confidence_sum'],b['correct_sum']] for j,b in enumerate(v) if not canonical_empty(b)]}
        else:out[k]=pack(v)
    return out
def unpack(value):
    if isinstance(value,list):return [unpack(x) for x in value]
    if not isinstance(value,dict):return value
    if MARK in value:
        assert list(value)==[MARK];bins=[dict(count=0,confidence_sum=0.0,correct_sum=0) for _ in range(15)];seen=set()
        for j,n,c,y in value[MARK]:
            assert type(j) is int and 0<=j<15 and j not in seen;seen.add(j)
            bins[j]=dict(count=n,confidence_sum=c,correct_sum=y)
        return bins
    return {k:unpack(v) for k,v in value.items()}
def original_bytes(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
