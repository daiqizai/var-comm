"""Join E3 actual receive outputs and isolate native pairs absent from E1."""
from followup_common import *

def main():
 closed('E3_rx');closed('E3_phy');out=L/'E3_metric_plan';out.mkdir(exist_ok=False)
 source=read(L/'inventory_core/sources.json')[:100];sm={s['source_id']:s for s in source}
 old={}
 for v in read(L/'E1_join/frames.json'):
  if v['method']in ['final_NeST','original_NeST','WHOLE_final_receiver']:old.setdefault(v['dists_pair_key'],v)
 physical={v['physical_key']:v for v in read(L/'E3_phy/frames.json')};rx={v['key']:v for v in read(L/'E3_rx/outputs.json')};frames=[];missing={}
 for v in read(L/'E3_plan/logical.json'):
  v=dict(v);f=physical[v['physical_key']];q=rx[f['received_key']];s=sm[v['source_id']]
  assert v['reference_sha256']==s['reference_sha256'];pk=digest([s['reference_sha256'],q['pixel_sha256']])
  v.update(received_key=f['received_key'],frame=f['frame'],image=q['image'],pixel_sha256=q['pixel_sha256'],pair_key=pk,receiver_status=q['status'])
  if pk in old:
   p=old[pk];assert(p['reference_sha256'],p['pixel_sha256'])==(v['reference_sha256'],v['pixel_sha256']);v.update({m:p[m]for m in METRICS});v['metric_origin']='EXACT_E1_NATIVE_PAIR'
  else:
   missing.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=q['pixel_sha256'],image=q['image']));v['metric_origin']='NEW_FIVE_METRIC_GAP'
  frames.append(v)
 save(out/'logical.json',frames);save(out/'plan.json',dict(sources=source,pairs=list(missing.values())));save(out/'pairs.json',list(missing.values()))
 save(out/'completion.json',dict(status='PREPARED',logical=len(frames),unique_pairs=len({v['pair_key']for v in frames}),new_pairs=len(missing),new_models=0,new_PHY=0));print(read(out/'completion.json'))
if __name__=='__main__':main()
