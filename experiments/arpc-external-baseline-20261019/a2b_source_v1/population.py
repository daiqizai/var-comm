"""Source-ID selection only, with no metric or pixel content input."""
def missing(full,previous):
    assert len(full)==1000 and len(previous)==100
    ids={x['source_id']:x for x in full};old={x['source_id']:x for x in previous}
    assert len(ids)==1000 and len(old)==100 and set(old)<=set(ids)
    assert sorted(x['source_index'] for x in full)==list(range(1000))
    for sid,a in old.items():
        b=ids[sid];assert a['source_index']==b['source_index'] and a['archive']==b['archive'] and a['preprocessing_id']==b['preprocessing_id']
    out=[]
    for s in sorted(full,key=lambda x:x['source_index']):
        if s['source_id'] in old:continue
        out.append({k:s[k] for k in ['source_id','source_index','archive','preprocessing_id']})
        out[-1]['rank']=s['source_index']
    assert len(out)==900;return out
