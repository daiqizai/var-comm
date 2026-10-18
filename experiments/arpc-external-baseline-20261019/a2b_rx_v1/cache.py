"""Exact received-input cache lookup, independent of truth/source identity."""
def match(jobs,states):
    cache={}
    for s in states:
        assert s['generation_seed']==0
        key=(s['wire']['sha256'],s['wire']['bytes'],s['endpoint'],s['generation_seed'])
        if key in cache:
            assert all(cache[key]['image'][k]==s['image'][k] for k in ['sha256','bytes']),'Inconsistent independent RX cache'
        cache.setdefault(key,s)
    result=[];new=[];keys=set()
    for j in jobs:
        assert j['key'] not in keys;keys.add(j['key'])
        assert j['generation_seed']==0 and j['paid_profile'] in range(1,8)
        key=(j['input']['sha256'],j['input']['bytes'],j['paid_profile'],j['generation_seed'])
        if key in cache:
            s=cache[key];result.append({'key':j['key'],'endpoint':j['paid_profile'],'input':j['input'],'image':s['image'],'origin':'EXACT_INDEPENDENT_RX_CACHE'})
        else:new.append(j)
    assert len(new)<=6300,'RESOURCE_REESTIMATE_REQUIRED: actual source states exceed remaining cap'
    return result,new
