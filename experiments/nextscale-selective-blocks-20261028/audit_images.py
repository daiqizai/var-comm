"""Audit saved native outputs and create a finite metric plan, without inference."""
from block_common import *
from audit_prepare_p1 import closed
import traceback

def image_checked(r):
 with np.load(checked(r['image']),allow_pickle=False)as z:a=z['image'].copy()
 assert a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
 assert asha(a)==r['pixel_sha256']
 assert digest(r['state'])==r['key']
 return a

def main():
 out=L/'p1_image_audit';out.mkdir(exist_ok=False)
 try:
  sources=read(O/'source_manifest.json');source={s['source_id']:s for s in sources}
  reuse=cr(read(O/'reuse_inventory.json')['single_reuse']);single={};bywire={};receipts=[]
  for r in reuse:
   image_checked(r);assert r['key']not in single and r['wire_key']not in bywire
   single[r['key']]=r;bywire[r['wire_key']]=r
  selective=[]
  for role in ['p1_images_single_00','p1_images_single_01','p1_images_selective_00','p1_images_selective_01']:
   c=closed(L,role,L/(role+'_control'));assert not(L/role/'failure.json').exists()
   rows=cr(c['outputs']);assert c['count']==len(rows)
   receipts.append(dict(completion=pin(L/role/'completion.json'),wait=pin(L/(role+'_control')/'actual_wait.json')))
   for r in rows:
    image_checked(r);assert not r['source_truth_used']and not r['TX_CDF_used']
    if c['mode']=='SINGLE':
     assert r['key']not in single and r['wire_key']not in bywire
     single[r['key']]=r;bywire[r['wire_key']]=r
    else:selective.append(r)
  assert len(single)==800 and len(selective)==256
  alignment=[];normal=dict(single)
  for r in selective:
   if r['role']=='ALIGNMENT':
    b=single[r['key']];a=image_checked(r);v=image_checked(b)
    same=np.array_equal(a,v)
    alignment.append(dict(key=r['key'],selective_image=r['image'],continuous_image=b['image'],selective_pixel_sha256=r['pixel_sha256'],continuous_pixel_sha256=b['pixel_sha256'],exact=same,max_abs_difference=float(np.max(np.abs(a-v)))))
   else:
    assert r['role']=='MECHANISM'and r['key']not in normal;normal[r['key']]=r
  save(out/'alignment.json',alignment)
  assert len(alignment)==128 and all(a['exact']for a in alignment),'Preserve outputs: full-CAB alignment failed; no automatic retry'
  assert len(normal)==928
  pairs={};events=[]
  def pair(sid,r):
   s=source[sid];px=asha(np.full((3,256,256),.5,np.float32))if r is None else r['pixel_sha256']
   identity=dict(source_id=sid,reference_sha256=s['reference_sha256'],pixel_sha256=px)
   key=digest(identity)
   if key not in pairs:pairs[key]=dict(pair_key=key,source_index=s['source_index'],**identity,image=None if r is None else r['image'])
   return key
  for e in read(O/'single_state_events.json'):
   r=bywire[e['wire_key']];assert r['state']['preprocessing']==source[e['source_id']]['preprocessing_id']
   events.append(dict(**e,state='SINGLE',state_key=r['key'],pair_key=pair(e['source_id'],r)))
  for e in read(O/'clean_state_events.json'):
   r=normal[e['key']];assert r['state']['preprocessing']==source[e['source_id']]['preprocessing_id']
   events.append(dict(**e,state_key=r['key'],pair_key=pair(e['source_id'],r)))
  for s in sources:events.append(dict(source_id=s['source_id'],source_index=s['source_index'],state='GRAY',pair_key=pair(s['source_id'],None)))
  assert len(events)==1344 and len(pairs)<=960
  save(out/'normal_native_states.json',list(normal.values()))
  save(out/'metric_plan.json',dict(sources=sources,pairs=list(pairs.values())))
  save(O/'clean_metric_events.json',events)
  save(O/'render_alignment_proof.json',dict(status='PASS',complete_CAB_exact=128,normal_states=928,new_continuous=553,reused_continuous=247,new_CB=128,extra_alignment_calls=128,wireless_frames=0,alignment=pin(out/'alignment.json'),receipts=receipts,metric_plan=pin(out/'metric_plan.json'),metric_pair_count=len(pairs)))
  save(out/'completion.json',dict(status='COMPLETE_IMAGE_AUDIT',normal_states=928,metric_pairs=len(pairs),metric_plan=pin(out/'metric_plan.json'),proof=pin(O/'render_alignment_proof.json')))
  print(json.dumps(dict(alignment_exact=128,normal_states=928,metric_pairs=len(pairs))))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
