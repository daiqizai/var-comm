"""After all actual E2 receivers close, join exact outputs and isolate metric gaps."""
from followup_common import *

def main():
 roles=['E2_nest']+[f'E2_ada_{i:02d}'for i in range(4)]
 for role in roles:closed(role)
 out=L/'E2_metric_plan';out.mkdir(exist_ok=False)
 sources=read(L/'inventory_core/sources.json');sm={s['source_id']:s for s in sources};rx={}
 for role in roles:
  for v in read(L/role/'outputs.json'):
   if v['key']in rx:assert v['pixel_sha256']==rx[v['key']]['pixel_sha256']
   else:rx[v['key']]=v
 old={}
 for v in read(L/'E1_join/frames.json'):old.setdefault(v['dists_pair_key'],v)
 # Subsequent completed controlled/intervention pairs use the same evaluators.
 for exp in ['e3','e4']:
  for v in rows(D/exp/'frame_metrics.csv'):
   pk=v['pair_key'];old.setdefault(pk,dict(v,**{m:float(v[m])for m in METRICS}))
 frames=[];missing={};used=set()
 for v in read(L/'E2_prepare_metadata_v2/logical.json'):
  v=dict(v);q=rx[v['key']];s=sm[v['source_id']];used.add(v['key'])
  assert v['reference_sha256']==s['reference_sha256']and v['source_pixels_sha256']==s['preprocessing_id']
  if v['method']=='Ada'and q['origin']=='NEW_ERROR_FREE_PREFIX_DECODE_AND_SAMPLING':assert v['generation_seed']==q['generation_seed']
  pk=digest([s['reference_sha256'],q['pixel_sha256']]);v.update(image=q['image'],pixel_sha256=q['pixel_sha256'],pair_key=pk,receiver_status=q['status'],receiver_origin=q['origin'])
  if pk in old:
   p=old[pk];assert(p['reference_sha256'],p['pixel_sha256'])==(v['reference_sha256'],v['pixel_sha256']);v.update({m:p[m]for m in METRICS});v['metric_origin']='EXACT_COMPLETED_NATIVE_PAIR'
  else:
   missing.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=q['pixel_sha256'],image=q['image']));v['metric_origin']='NEW_FIVE_METRIC_GAP'
  frames.append(v)
 assert len(frames)==15000 and used==set(rx)
 save(out/'logical.json',frames);save(out/'plan.json',dict(sources=sources,pairs=list(missing.values())));save(out/'pairs.json',list(missing.values()))
 save(out/'completion.json',dict(status='PREPARED',logical=15000,unique_receiver_inputs=len(rx),unique_pairs=len({v['pair_key']for v in frames}),new_pairs=len(missing),new_models=0,new_PHY=0));print(read(out/'completion.json'))
if __name__=='__main__':main()
