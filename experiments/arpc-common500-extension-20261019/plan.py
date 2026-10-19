"""Enumerate the frozen full-calibration policy grid without accessing pixels."""
import hashlib,json,re
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def events(sources,decisions,profiles):
    assert len({s['source_id'] for s in sources})==len(sources)
    out=[]
    for s in sorted(sources,key=lambda s:s['source_index']):
        streams={x['endpoint']:x for x in s['streams']};assert set(streams)==set(range(1,8))
        for snr in [4,10,19]:
            candidates=decisions[str(snr)]['proposed_candidates'];assert len(candidates)==1 and len(set(candidates))==len(candidates)
            for cid in candidates:
                m=re.fullmatch(r'e([1-7])_q([246])_r(\d)-(\d)',cid);assert m
                target,q,num,den=map(int,m.groups());rate=f'{num}/{den}'
                template=next(p for p in profiles if (p['endpoint'],p['q'],p['nominal_rate'])==(target,q,rate))
                end=max([e for e in range(1,target+1) if streams[e]['payload_bits']<=template['source_capacity_bits']],default=0)
                profile=next(p for p in profiles if (p['endpoint'],p['q'],p['nominal_rate'])==(end,q,rate))
                wire=None if end==0 else streams[end]['wire'];counter=3000000+96*s['source_index']+profile['profile_id']
                wavekey=digest([s['source_index'],counter,profile['profile_id'],wire['sha256'] if wire else 'NULL'])
                for noise in [6201,6202,6203]:
                    key=digest([s['source_id'],snr,cid,noise])
                    out.append({'key':key,'source_id':s['source_id'],'source_index':s['source_index'],'snr_db':snr,'candidate_id':cid,
                      'noise_seed':noise,'target_endpoint':target,'endpoint':end,'source_no_fit':end==0,'profile_id':profile['profile_id'],
                      'q':q,'nominal_rate':rate,'counter':counter,'waveform_key':wavekey,'wire':wire})
    assert len(out)<=3600 and len({x['key'] for x in out})==len(out);return out
