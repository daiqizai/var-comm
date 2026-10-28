"""After genuine receiver closure, register native-pixel metrics exactly once."""
from evidence_common import *
import subprocess,copy
NL=B/'nestcom_n512_20261028_v1'
def main():
 prep=L/'metric_preparation';prep.mkdir(exist_ok=False);rows=[];outputs={};closure=[]
 for s in read(O/'receiver_stage_manifest.json'):
  w=read(L/(s['role']+'_control')/'actual_wait.json');assert w['actual_wait']and w['returncode']==0 and gone(w['process'])and gone(w['owner'])
  out=L/s['role'];assert not(out/'failure.json').exists();c=read(out/'completion.json');assert c['count']==s['count']and c['ledger']['unresolved']==0
  for j in cr(c['outputs']):
   with np.load(checked(j['image']),allow_pickle=False)as z:assert asha(z['image'])==j['pixel_sha256']
   assert j['key']not in outputs;outputs[j['key']]=j
  closure.append(dict(stage=s,actual_wait=pin(L/(s['role']+'_control')/'actual_wait.json'),completion=pin(out/'completion.json'),qualifications=cr(c['qualification'])))
 assert len(outputs)==485;save(O/'receiver_closure.json',closure)
 sources=read(L/'input_inventory_v2/sources.json')[:100];logical=read(L/'input_inventory_v2/logical.json');pairs={}
 for row in logical:
  if 'new_key'not in row:continue
  j=outputs[row['new_key']];key=digest([row['reference_sha256'],j['pixel_sha256']]);pairs.setdefault(key,dict(pair_key=key,source_id=row['source_id'],source_index=row['source_index'],reference_sha256=row['reference_sha256'],image=j['image'],pixel_sha256=j['pixel_sha256']))
  row['image']=j['image'];row['pixel_sha256']=j['pixel_sha256'];row['pair_key']=key;row.pop('metrics')
 save(prep/'resolved_logical.json',logical);save(prep/'pairs.json',dict(sources=sources,pairs=list(pairs.values())));assert len(pairs)<=485
 stages=[]
 for mode,gpu,pci in [('three',6,'0000:a3:00.0'),('clip',1,'0000:62:00.0')]:
  original=R/f'results/nestcom_n512_20261028_v1/formal_{mode}_metrics_registration.json';r=read(original);role=mode+'_metrics';r['output']=str(L/role);r['plan']=pin(prep/'pairs.json');r['max_seconds']=3600
  # Reuse all identity-matched source and previous reconstruction embeddings.
  features=cr(r['feature_cache'])if r.get('feature_cache')else{};features.update(read(NL/f'formal_{mode}_metrics/features.json'))
  save(prep/(mode+'_feature_cache.json'),features);r['feature_cache']=pin(prep/(mode+'_feature_cache.json'))
  paircache=cr(r['pair_cache'])if r.get('pair_cache')else{};paircache.update(read(NL/f'formal_{mode}_metrics/scores.json'))
  save(prep/(mode+'_pair_cache.json'),paircache);r['pair_cache']=pin(prep/(mode+'_pair_cache.json'))
  r['caps']=dict(embedding=len(pairs),pair=len(pairs),**({'load_lpips':1,'load_dreamsim':1}if mode=='three'else{'load_clip':1}))
  r['original_metric_registration']=pin(original);r['checkpoint_relocation']=pin(R/'tools/checkpoint_relocation_20261028.py');r['protocol']=pin(O/'protocol.json')
  files={d['path']:d for d in r['code']}
  for p in[E/'metric_worker.py',E/'evidence_common.py',R/'tools/checkpoint_relocation_20261028.py']:files[str(p)]=pin(p)
  r['code']=list(files.values());path=O/(role+'_registration.json');save(path,r)
  py=str(RT/'envs/frozen_visual_py310/bin/python')if mode=='three'else'/usr/bin/python'
  ctrl=dict(control=str(L/(role+'_control')),code=[pin(E/'control.py'),pin(E/'evidence_common.py')],argv=[py,'-B',str(E/'metric_worker.py'),str(path),sha(path)],environment=dict(CUDA_VISIBLE_DEVICES=str(gpu)),seconds=3660)
  cp=O/(role+'_control_registration.json');save(cp,ctrl);stages.append(dict(role=role,request=pin(path),control=pin(cp),gpu=gpu,pci=pci,count=len(pairs)))
 save(O/'metric_stage_manifest.json',stages);print(json.dumps(dict(pairs=len(pairs),stages=stages)))
if __name__=='__main__':main()
