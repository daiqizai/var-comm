"""Received-only deduplication and finite deterministic shards. No inference."""
import numpy as np
from common import asha,digest
from c_predclass_protocol import class_bits,received_class,generation_seed
from c_predclass_frame import source_projection


def received_input(projection):
    if projection['status']=='FIXED_GRAY':return None
    assert projection['status']=='SOURCE_INPUT'
    a=np.asarray(projection['bits'])
    assert a.ndim==1 and 2<=len(a)<=8181 and np.isin(a,(0,1)).all()
    bits=a.astype(np.uint8);prefix=projection['prefix']
    assert type(prefix)is int and 1<=prefix<=256
    assert asha(bits)==projection['payload_sha256']
    cb=class_bits(projection['received_class']);cs=asha(np.concatenate([cb,bits]))
    return dict(key=digest(dict(prefix=prefix,container_sha256=cs)),prefix=prefix,
        bits=bits.tolist(),payload_sha256=asha(bits),received_class_bits=cb.tolist(),container_sha256=cs)


def build_plan(frames,cache,contract_sha256,expected_logical=18000,shard_size=1000):
    assert len(contract_sha256)==64 and all(v in '0123456789abcdef'for v in contract_sha256)
    assert type(expected_logical)is int and 0<expected_logical<=18000
    assert len(frames)==expected_logical
    assert type(shard_size)is int and 1<=shard_size<=1000
    cache_by_key={}
    for c in cache:
        assert c['contract_sha256']==contract_sha256
        inp=c['input'];key=inp['key'];out=c['output']
        assert out['key']==key and out['prefix']==inp['prefix']
        assert out['payload_sha256']==inp['payload_sha256']
        cb=np.array(inp['received_class_bits'],np.uint8);bits=np.array(inp['bits'],np.uint8)
        cl=received_class(cb)
        assert out['received_class']==cl and out['generation_seed']==generation_seed(bits,inp['prefix'],cl)
        assert asha(bits)==inp['payload_sha256']and asha(np.concatenate([cb,bits]))==inp['container_sha256']
        assert key==digest(dict(prefix=inp['prefix'],container_sha256=inp['container_sha256']))
        assert out['source_truth_used']is False and out['TX_CDF_used']is False
        if key in cache_by_key:assert cache_by_key[key]['input']==inp and cache_by_key[key]['output']==out
        else:cache_by_key[key]=c
    events=set();new={};used_reuse={};logical=[]
    ordered=sorted(frames,key=lambda x:(x['value']['event']['source_index'],x['value']['event']['snr_db'],x['value']['event']['candidate_id'],x['value']['event']['noise_seed']))
    for row in ordered:
        f=row['value'];ev=f['event'];key=ev['key'];assert key not in events;events.add(key)
        p=source_projection(f['actual_RX']);assert p==f['source_projection']
        inp=received_input(p)
        item=dict(event_key=key,source_index=ev['source_index'],source_id=ev['source_id'],
            snr_db=ev['snr_db'],candidate_id=ev['candidate_id'],noise_seed=ev['noise_seed'],frame=row['descriptor'],
            wire_status=f['actual_RX']['status'],projection_status=p['status'],RX_key=None if inp is None else inp['key'])
        if inp is None:
            item.update(origin='FIXED_GRAY',reason=p['reason'])
        elif inp['key']in cache_by_key:
            c=cache_by_key[inp['key']];assert c['input']==inp
            used_reuse[inp['key']]=c;item.update(origin='EXACT_CLOSED_RX_REUSE',reason=None)
        else:
            if inp['key']in new:assert new[inp['key']]==inp
            else:new[inp['key']]=inp
            item.update(origin='NEW_UNIQUE_RX',reason=None)
        logical.append(item)
    inputs=list(new.values())
    shards=[inputs[i:i+shard_size]for i in range(0,len(inputs),shard_size)]
    assert len(logical)==expected_logical and sum(len(v)for v in shards)==len(new)
    return dict(logical=logical,shards=shards,reused=list(used_reuse.values()),
        counts=dict(logical_events=len(logical),gray_events=sum(v['origin']=='FIXED_GRAY'for v in logical),
            reused_logical=sum(v['origin']=='EXACT_CLOSED_RX_REUSE'for v in logical),
            new_logical=sum(v['origin']=='NEW_UNIQUE_RX'for v in logical),
            reused_unique=len(used_reuse),new_unique=len(inputs),shards=len(shards)),
        contract_sha256=contract_sha256)
