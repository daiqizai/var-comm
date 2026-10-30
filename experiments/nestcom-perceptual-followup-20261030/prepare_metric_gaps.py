"""Plan only missing metrics; no scientific model calls."""
from followup_common import *
def main():
 sources=read(L/'inventory_core/sources.json')
 closed('E4_interventions');out=L/'metric_gaps';out.mkdir(exist_ok=False)
 seven=[v for v in read(L/'inventory_external/logical.json')if v['snr_db']==7]
 assert len(seven)==6000
 pairs={v['dists_pair_key']:dict(pair_key=v['dists_pair_key'],source_id=v['source_id'],source_index=v['source_index'],reference_sha256=v['reference_sha256'],pixel_sha256=v['pixel_sha256'],image=v['image'])for v in seven}
 save(out/'E1_external_clip.json',dict(sources=sources,pairs=list(pairs.values())))
 outputs={(v['key'],v['arm']):v for v in read(L/'E4_interventions/outputs.json')};logical=read(L/'E4_plan/logical.json');ep={}
 for v in logical:
  if v['reuse_reason']=='NEW_INTERVENTION':
   r=outputs[(v['key'],v['arm'])];v.update(image=r['image'],pixel_sha256=r['pixel_sha256'],receiver_status='ACCEPTED_SOURCE')
   key=digest([v['reference_sha256'],v['pixel_sha256']]);ep.setdefault(key,dict(pair_key=key,source_id=v['source_id'],source_index=v['source_index'],reference_sha256=v['reference_sha256'],pixel_sha256=v['pixel_sha256'],image=v['image']))
  v['pair_key']=digest([v['reference_sha256'],v['pixel_sha256']]);assert v['pixel_sha256']is not None
 save(out/'E4_new.json',dict(sources=sources[:100],pairs=list(ep.values())))
 save(out/'E4_new_pairs.json',list(ep.values()));save(out/'E4_logical.json',logical)
 save(out/'completion.json',dict(status='PREPARED',E1_missing_clip_logical=6000,E1_missing_clip_unique=len(pairs),E4_new_metric_pairs=len(ep),E4_logical=len(logical),new_models=0,new_PHY=0))
 print(read(out/'completion.json'))
if __name__=='__main__':main()
