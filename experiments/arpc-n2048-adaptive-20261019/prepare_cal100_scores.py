"""Closed native output audit and exact pixel-pair DINO gap plan."""
from shared import *
from adaptive_plan import NEST_ENDPOINTS,SIZES
import numpy as np,csv
def main():
 w=read(L/'controls/cal100_clean_native/actual_wait.json');launch=read(L/'controls/cal100_clean_native/launch.json');assert w['returncode']==0 and gone(w['process']) and gone(launch['parent_process'])
 root=L/'cal100_scores_v1';root.mkdir();nc=read(L/'cal100_clean_v1/rx/completion.json');assert nc['ledger']['unresolved']==0
 images={x['source_input_key']:x for x in nc['outputs']};images.update({x['source_input_key']:x for x in read(L/'qualification/native/rx/completion.json')['outputs']})
 cached=read(L/'cal100_verified_cache_v1/cache.json');paircache={(x['source_id'],x['image_float_sha256']):x for x in cached};states=[dict(source_id=x['source_id'],source_index=x['source_index'],codec=x['codec'],endpoint=x['endpoint'],source=x['source'],image=x['image'],array_key=x['array_key'],image_float_sha256=x['image_float_sha256'],dino=x['dino'],origin='VERIFIED_EXISTING_PIXEL_PAIR')for x in cached]
 groups={};newpairs={};logicalreuse=0
 for x in read(L/'cal100_clean_v1/private_mapping.json'):
  s=x['source'];im=images[x['key']];assert im['status']=='SOURCE_DECODED' and not im['source_truth_used'] and not im['TX_CDF_used']
  with np.load(verify(s['source']),allow_pickle=False)as z:truth=z['tokens'].copy()
  m,K=NEST_ENDPOINTS[s['endpoint']];assert np.array_equal(truth[:sum(v*v for v in SIZES[:m])+K],im['received_tokens']),('BLOCKED_SOURCE_CDF_MODEL_MISMATCH',s['source_id'],s['endpoint'])
  with np.load(verify(im['image']),allow_pickle=False)as z:a=z['image'].copy()
  assert hashlib.sha256(a.tobytes()).hexdigest()==im['float_sha256']
  key=(s['source_id'],im['float_sha256']);old=paircache.get(key)
  row=dict(source_id=s['source_id'],source_index=s['source_index'],codec=0,endpoint=s['endpoint'],source=s['source'],image=im['image'],array_key='image',image_float_sha256=im['float_sha256'],origin='EXACT_EXISTING_PIXEL_PAIR' if old else 'NEW_DINO_REQUIRED',dino=old['dino'] if old else None)
  states.append(row)
  if old:logicalreuse+=1
  elif key not in newpairs:newpairs[key]=row
 assert len(states)==3200
 prior=read(OLD_O/'a2b_metrics_v1/registration.json');refs={x['source_id']:x for x in prior['prior_references']};assert len(refs)==100
 oldmetric=read(OLD_O/'a2_metrics_v1/registration.json');metadata=read(OLD_O/'a2_metrics_v1/metric_identity.json')
 for row in newpairs.values():
  sid=row['source_id']
  if sid not in groups:
   ref=refs[sid];assert ref['source']['sha256']==row['source']['sha256'];verify(ref['file'])
   groups[sid]=dict(source_id=sid,source_index=row['source_index'],archive=row['source'],reference=ref['file'],pairs=[])
  groups[sid]['pairs'].append(row)
 plan=list(groups.values());put(root/'plan.json',plan);put(root/'all_states.json',states)
 gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=pci.bus_id,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(gpu[2])>=8192
 put(O/'cal100_score_registration.json',dict(status='DINO_ONLY_CAL100_FROZEN_ASSETS',plan=pin(root/'plan.json'),all_states=pin(root/'all_states.json'),metric_config=oldmetric['metric_config'],prior_metadata=pin(OLD_O/'a2_metrics_v1/metric_identity.json'),prior_reference_registration=pin(OLD_O/'a2b_metrics_v1/registration.json'),source=pin(E/'cal100_dino.py'),expected_runtime=read(verify(read(REG)['model_routes']))['native_qualification']['numeric_settings']['runtime'],source_reference_reuse=len(groups),reference_forward_new=0,new_DINO_pairs=len(newpairs),logical_pixel_pair_reuse=logicalreuse,source_images=100,clean_logical_states=3200,gpu_index=0,gpu_pci=gpu[0].strip(),gpu_uuid=gpu[1].strip(),native_wait=pin(L/'controls/cal100_clean_native/actual_wait.json'),max_seconds=1800,bootstrap_calls=0))
 print(json.dumps(dict(new_DINO_pairs=len(newpairs),reference_reuse=len(groups),logical_additional_reuse=logicalreuse,registration=pin(O/'cal100_score_registration.json'))))
if __name__=='__main__':main()
