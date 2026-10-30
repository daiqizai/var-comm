"""Reuse a donor only after exact actual received projection/seed equality."""
from followup_common import *
from collections import Counter

def main():
 closed('E6_phy');out=L/'E6_metric_plan';out.mkdir(exist_ok=False)
 sources=read(L/'inventory_core/sources.json')[:100];sm={s['source_id']:s for s in sources}
 donors={d['key']:d for d in read(L/'E6_plan_metadata_v2/donors.json')};frames=[];missing={};unknown={};counts=Counter()
 old={}
 for v in read(L/'E1_join/frames.json'):old.setdefault(v['dists_pair_key'],v)
 for exp in ['e3','e4']:
  for v in rows(D/exp/'frame_metrics.csv'):old.setdefault(v['pair_key'],dict(v,**{m:float(v[m])for m in METRICS}))
 gray=asha(np.full((3,256,256),.5,np.float32))
 for row in read(L/'E6_phy/frames.json'):
  v=dict(row);f=cr(v['frame']);d=donors[v['donor_key']];rx=f['actual_RX'];p=f['projection'];s=sm[v['source_id']]
  assert f['counter']==d['counter']and f['session']==d['session']and f['waveform_sha256']==d['transmission']['transmitted_frame_sha256']and not f['source_truth_supplied_to_RX']
  if p['status']=='SOURCE_INPUT':
   if p!=d['projection']or v['generation_seed']!=d['generation_seed']:
    unknown.setdefault(v['received_key'],dict(key=v['received_key'],projection=p,generation_seed=v['generation_seed'],example_actual_frame=v['frame']));continue
   image=d['image'];pixels=d['pixel_sha256'];category='SUCCESS';origin='EXACT_ACTUAL_RECEIVED_PREFIX_AND_SEED_DONOR'
  else:
   image=None;pixels=gray;origin='ACTUAL_REJECTION_FIXED_GRAY'
   category='HEADER_REJECT'if not rx['header']['header_ok']else'NULL'if 'NULL'in str(p.get('reason','')).upper()else'BODY_OR_PROFILE_PARSE_REJECT'
  counts[category]+=1;pk=digest([s['reference_sha256'],pixels]);v.update(image=image,pixel_sha256=pixels,pair_key=pk,outcome=category,receiver_origin=origin)
  if pk in old:
   q=old[pk];assert(q['reference_sha256'],q['pixel_sha256'])==(v['reference_sha256'],pixels);v.update({m:q[m]for m in ['dreamsim','dists']});v['metric_origin']='EXACT_COMPLETED_NATIVE_PAIR'
  else:
   missing.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=pixels,image=image));v['metric_origin']='NEW_TWO_METRIC_GAP'
  frames.append(v)
 save(out/'unexpected_received_inputs.json',list(unknown.values()))
 if unknown:
  save(out/'blocked.json',dict(reason='Actual accepted projections differ from donor; must independently decode, never force rejection or use TX truth',inputs=len(unknown)));raise RuntimeError('E6 legitimate new RX inputs require isolated decoder stage')
 assert len(frames)==30000
 save(out/'logical.json',frames);save(out/'plan.json',dict(sources=sources,pairs=list(missing.values())));save(out/'pairs.json',list(missing.values()))
 save(out/'completion.json',dict(status='PREPARED',logical=30000,unique_pairs=len({v['pair_key']for v in frames}),new_pairs=len(missing),outcomes=dict(counts),new_models=0,new_PHY=0));print(read(out/'completion.json'))
if __name__=='__main__':main()
