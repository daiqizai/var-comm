"""Freeze exact missing source/pixel metric pairs after closed packet evidence."""
from p1_common import *
def main():
 c=read(L/'link_closed_v1/completion.json');assert c['logical']==1800 and c['unique_new_accepted_inputs']==0 and c['new_scientific_calls']==0
 w=read(L/'link_metadata_closure_control/actual_wait.json');assert w['returncode']==0
 oldw=read(L/'link_control/actual_wait.json');assert oldw['returncode']==1 and gone(oldw['process'])and gone(oldw['parent_process'])
 r=read(O/'p1_registration.json');sources=cr(r['sources']);frames=cr(c['frames']);out=L/'metric_preparation';out.mkdir()
 gray=np.full((3,256,256),.5,np.float32)
 with(out/'gray.npz').open('xb')as f:np.savez(f,image=gray)
 clip={};pins=[]
 for rel in ('results/clip_i_common500_20261020_v1/delivery/frame_scores.csv','results/adatoken_response_20261026_v1/a_delivery/frame_metrics.csv','results/adatoken_response_20261026_v1/b_delivery/frame_metrics.csv'):
  p=R/rel;pins.append(pin(p))
  with p.open()as f:
   for x in csv.DictReader(f):
    if x.get('pixel_sha256')and x.get('clip_image_cosine'):clip.setdefault((x['source_id'],x['pixel_sha256']),dict(value=float(x['clip_image_cosine']),origin=str(p)))
 pairs={};logical=[]
 for i,f in enumerate(frames):
  j=f['event'];s=sources[j['rank']];a=f['reconstruction'];accepted=a is not None
  if accepted:image=a['image_archive'];ik=a['image_key'];slot=a.get('image_slot');ps=a['image_sha256'];values={m:f['old_metrics'][m]for m in ('psnr_db','lpips_alex','dreamsim')}
  else:assert f['RX_resolution']=='FIXED_GRAY_REJECT';image=pin(out/'gray.npz');ik='image';slot=None;ps=ash(gray);values={}
  key=s['reference_sha256']+'_'+ps
  if(s['source_id'],ps)in clip:values['clip_image_cosine']=clip[s['source_id'],ps]['value']
  item=dict(key=key,source_id=s['source_id'],source_index=s['source_index'],rank=j['rank'],reference_sha256=s['reference_sha256'],image=image,image_key=ik,image_slot=slot,pixel_sha256=ps,values=values)
  if key in pairs:assert pairs[key]['values']==values
  else:pairs[key]=item
  logical.append(dict(logical_index=i,source_id=s['source_id'],source_index=s['source_index'],rank=j['rank'],family=j['family'],TX_policy_SNR=10,snr_db=j['snr_db'],noise_seed=j['noise_seed'],m=j['m'],K=j['K'],arithmetic_bits=j['arithmetic_bits'],source_capacity_bits=2520,accepted=accepted,final_status='ARITHMETIC_SOURCE_DECODED'if accepted else f['wire_status'],pair_key=key,pixel_sha256=ps,physical_origin=f['physical_origin'],physical_frame=f['physical_frame'],source_input_key=f['actual_RX_input_key']))
 for j in pairs.values():
  with np.load(checked(j['image']),allow_pickle=False)as z:a=z[j['image_key']].copy()
  if j['image_slot']is not None:a=a[j['image_slot']]
  assert ash(a)==j['pixel_sha256']and a.dtype==np.float32
 save(out/'pairs.json',list(pairs.values()));save(out/'logical.json',logical)
 donorpath=R/'results/adatoken_response_20261026_v1/a_metrics_registration.json';donor=read(donorpath)
 limits={role:sum(any(m not in j['values']for m in metrics)for j in pairs.values())for role,metrics in {'three':('psnr_db','lpips_alex','dreamsim'),'clip':('clip_image_cosine',)}.items()}
 reg=dict(schema='P1_MISSING_METRICS_ONLY_V1',parent_registration=pin(O/'p1_registration.json'),link_closure=pin(L/'link_closed_v1/completion.json'),metadata_wait=pin(L/'link_metadata_closure_control/actual_wait.json'),sources=r['sources'],pairs=pin(out/'pairs.json'),logical=pin(out/'logical.json'),cache_scores=pins,metric_donor=pin(donorpath),max_seconds=3600,limits=limits,resources={'three':dict(index=6,pci='0000:a3:00.0',torch='2.11.0+cu128',python=str(RT/'envs/frozen_visual_py310/bin/python')),'clip':dict(index=1,pci='0000:62:00.0',torch='2.10.0+cu130',python='/usr/bin/python')},code=[pin(E/n)for n in ('p1_common.py','p1_metrics_prepare.py','p1_metrics.py','p1_metrics_control.py')],automatic_retry=False,new_VAR_calls=0,new_source_decode=0)
 save(O/'p1_metrics_registration.json',reg);print(json.dumps(dict(registration=pin(O/'p1_metrics_registration.json'),pairs=len(pairs),missing=limits)))
if __name__=='__main__':main()
