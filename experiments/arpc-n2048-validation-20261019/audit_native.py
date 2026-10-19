"""Post-exit comparison only; no image/model/source decoder calls."""
from shared import *
import numpy as np
def main(expected):
 r=registration(expected);phase='qualification';run=L/phase
 # Remote native worker closure is independently verified on its actual host.
 arpc_auth=read(O/'qualification_arpc_resource_authorization.json')
 assert arpc_auth['status']=='AUTHORIZED_ZERO_MODEL_REPLAY'
 for k in ['registration','source','original_public','public','old_jobs','jobs']:verify(arpc_auth[k])
 for d in arpc_auth['failure_evidence']:verify(d)
 for stage in ['qualification_nest','qualification_arpc_resource_v1']:
  wait=read(L/'controls'/stage/'actual_wait.json');assert type(wait['returncode'])is int and wait['returncode']==0 and not any(wait[k] for k in ['timed_out','terminated','killed','metadata_error'])
 nest=read(run/'nest/rx/completion.json');arpc=read(run/'arpc_resource_v1/rx/completion.json');assert nest['status']==arpc['status']=='COMPLETE'
 assert nest['ledger']['unresolved']==0 and not arpc['blocked_reads'] and arpc['counts']==dict(model_load=1,rx_prefix=6,rx_render=6)
 n={x['source_input_key']:x for x in nest['outputs']};a={x['key']:x for x in arpc['outputs']}
 frames=read(O/phase/'logical_frames.json');sourcejobs={x['key']:x for x in r['qualification_jobs']};rows=[]
 for x in frames:
  if x['case']!='clean':continue
  job=sourcejobs[x['key']];asset=job['asset'];key=x['source_input_key'];assert x['exact_post_exit'];verify(asset['source_archive'])
  if x['codec']==0:
   actual=n[key];assert actual['status']=='SOURCE_DECODED' and not actual['source_truth_used'] and not actual['TX_CDF_used']
   with np.load(asset['source_archive']['path'],allow_pickle=False) as z:truth=z['tokens'].copy()
   assert np.array_equal(truth,np.asarray(actual['received_tokens']))
   ip=actual['image'];verify(ip)
   with np.load(ip['path'],allow_pickle=False) as z:image=z['image'].copy()
   row=dict(codec=0,source_id=x['source_id'],tokens_exact=True,image=ip,image_std=float(image.std()),kind='ACTUAL_INDEPENDENT_FULL_M10')
  else:
   actual=a[key];assert actual['endpoint']==7;ip=pin(actual['path']);reference=asset['full_image'];verify(reference)
   with np.load(ip['path'],allow_pickle=False) as z:image=z['float_output'].copy();pixels=z['pixels'].copy()
   with np.load(reference['path'],allow_pickle=False) as z:ref=z['float_output'].copy();rp=z['pixels'].copy()
   err=float(np.max(np.abs(image-ref)));assert err<=1e-6 and np.array_equal(pixels,rp),(x['source_id'],err)
   row=dict(codec=1,source_id=x['source_id'],exact_uint8=True,max_float_error=err,image=ip,old_reference=reference,image_std=float(image.std()),kind='ACTUAL_INDEPENDENT_FULL_E7')
  assert image.shape==(3,256,256) and np.isfinite(image).all() and image.min()>=0 and image.max()<=1 and image.std()>0
  rows.append(row)
 assert len(rows)==12
 put(O/'qualification/native_audit.json',dict(status='PASS_ACTUAL_12_SOURCE_ROUNDTRIPS',outputs=rows,TX_truth_only_after_outputs_fixed=True,metrics=0,source_scientific_calls=12,root_visual_review_still_required=True,arpc_resource_authorization=pin(O/'qualification_arpc_resource_authorization.json'),waits=[pin(L/'controls'/s/'actual_wait.json')for s in ['qualification_nest','qualification_arpc_resource_v1']]))
 print(json.dumps(dict(status='PASS_ACTUAL_12_SOURCE_ROUNDTRIPS',outputs=len(rows))))
if __name__=='__main__':main(sys.argv[1])
