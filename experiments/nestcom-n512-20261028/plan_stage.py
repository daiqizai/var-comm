"""Deterministic candidates and exact physical-state dedup, no model calls."""
from n512_phy import *
def main(stage,expected):
 rp=O/'protocol.json';assert sha(rp)==expected;r=read(rp)
 for d in r['planning_code']:checked(d)
 assert stage in ('pilot','full','formal');out=L/(stage+'_plan');out.mkdir(exist_ok=False)
 cat=cr(r['catalogue']);sources=cr(r['sources']['common500' if stage=='formal' else 'cal1000'])
 if stage=='pilot':
  ids=set(r['pilot_source_ids']);sources=[s for s in sources if s['source_id']in ids];assert len(sources)==100
  configs={str(snr):r['candidates'] for snr in (4,10,19)};seeds=(4101,)
 elif stage=='full':
  finalists=read(O/'pilot_finalists.json');assert finalists['protocol_sha256']==expected
  configs=finalists['candidates_by_snr'];seeds=(4101,4102,4103)
 else:
  policies=read(O/'frozen_policies.json');assert policies['protocol_sha256']==expected
  configs={snr:list(v.values()) for snr,v in policies['policies'].items()};seeds=(6201,6202,6203)
 physical={};logical=[];choices={};total=0
 for s in sources:
  with np.load(checked(s['nest_streams']),allow_pickle=False)as z:n={k:z[k].copy() for k in z.files}
  with np.load(checked(s['ada_streams']),allow_pickle=False)as z:a={k:z[k].copy() for k in z.files}
  for snr in (4,10,19):
   for c in configs[str(snr)]:
    choice_id=digest([s['source_id'],c]);p,b,no_fit=choose(a if c['method']=='ADA' else n,cat['profiles'],c,s['predicted_class'])
    choices[choice_id]=dict(profile_id=p['profile_id'],payload=b.tolist(),payload_sha256=asha(b),arithmetic_bits=0 if no_fit else len(b)-(10 if c['method']=='ADA' else 0),container_bits=len(b),class_bits=0 if no_fit or c['method']!='ADA' else 10,tx_no_fit=no_fit,m=p['m'],K=p['K'],q=p['q'],rate=p['nominal_rate'],k=p['k'],capacity=p['arithmetic_capacity_bits'],padding=p['k']-29-len(b))
    for seed in seeds:
     counter=(90000000 if stage=='formal' else 85000000)+s['source_index']*3+(seed-seeds[0])
     ident=dict(source_id=s['source_id'],source_index=s['source_index'],snr_db=snr,noise_seed=seed,counter=counter,profile_id=p['profile_id'],payload_sha256=asha(b),session=WIRE,protocol_sha256=expected)
     key=digest(ident)
     physical.setdefault(key,dict(key=key,**ident,choice_id=choice_id))
     logical.append(dict(source_id=s['source_id'],source_index=s['source_index'],method=c['method'],snr_db=snr,noise_seed=seed,candidate_id=c['candidate_id'],target=c['target'],physical_key=key,choice_id=choice_id))
  if len(logical)%1000==0:print('PLANNED',stage,len(logical),len(physical),flush=True)
 expected_count=sum(len(configs[str(snr)]) for snr in (4,10,19))*len(sources)*len(seeds)
 assert len(logical)==expected_count<=r['caps'][stage+'_logical']
 if stage=='formal':assert len(logical)==13500 and len(sources)==500
 save(out/'logical.json',logical);save(out/'choices.json',choices);save(out/'physical.json',list(physical.values()))
 # Source-based disjoint shards retain all repeated states within one worker.
 shards=[]
 for shard in range(8):
  jobs=[v for v in physical.values() if v['source_index']%8==shard];path=out/f'shard_{shard:02d}.json';save(path,jobs);shards.append(pin(path))
 save(out/'completion.json',dict(status='PLANNED',stage=stage,protocol=pin(rp),logical=pin(out/'logical.json'),choices=pin(out/'choices.json'),physical=pin(out/'physical.json'),shards=shards,logical_count=len(logical),physical_count=len(physical),source_count=len(sources),old_N1024_frames_reused=0))
if __name__=='__main__':main(*sys.argv[1:])
