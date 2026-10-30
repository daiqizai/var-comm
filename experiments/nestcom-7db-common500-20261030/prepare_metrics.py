"""Map all9000 outcomes to native arrays, preserving gray failures and repeats."""
from g import *
from dispatch import launch,gpu,dependency
def main():
 closed('chain');sources=read(L/'sources.json');byid={s['source_id']:s for s in sources};frames=[];pairs={};statuses={}
 gray=np.full((3,256,256),.5,np.float32);grayhash=asha(gray)
 def add(row,image,pixel,status):
  s=byid[row['source_id']];assert s['source_index']==row['source_index']
  if image:
   with np.load(checked(image),allow_pickle=False)as z:assert asha(z['image'])==pixel
  else:assert pixel==grayhash
  key=digest([s['reference_sha256'],pixel]);pairs.setdefault(key,dict(pair_key=key,source_id=s['source_id'],source_index=s['source_index'],image=image,pixel_sha256=pixel))
  frames.append(dict(row,pair_key=key,pixel_sha256=pixel,image=image,receive_status=status));statuses[status]=statuses.get(status,0)+1
 for method in ('NEST','ADA','ARPC'):
  pr='phy_'+method.lower();rr='rx_'+method.lower();phy=closed(pr);rx=closed(rr);outs={v['key']:v for v in cr(rx['outputs'])}
  assert phy['physical_frames']==1500 and phy['ledger']['unresolved']==rx['ledger']['unresolved']==0
  for f in cr(phy['frames']):
   if f['rx_key']is None:image=None;pixel=grayhash;status='PHY_REJECT_GRAY'
   else:v=outs[f['rx_key']];image=v['image'];pixel=v['pixel_sha256'];status=v['status']
   row={k:f[k]for k in ['method','source_id','source_index','N','snr_db','repeat','noise_seed','endpoint','source_bits','capacity_bits','accepted','no_fit']};row['frame_record']=pin(L/pr/(f['key']+'.json'));add(row,image,pixel,status)
 for method in ('BPG','P100000','Swin200000'):
  role=method.lower()+'_rx';done=closed(role);assert done['count']==1500 and done['ledger']['unresolved']==0
  for v in cr(done['outputs']):
   row={k:v[k]for k in ['method','source_id','source_index','N','snr_db','repeat','noise_seed']};row['frame_record']=pin(L/role/(v['key']+'.json'));add(row,v['image'],v['pixel_sha256'],v['status'])
 assert len(frames)==len({(r['method'],r['source_id'],r['repeat'])for r in frames})==9000
 for m in METHODS:assert len([r for r in frames if r['method']==m])==1500
 save(L/'logical_pair_map.json',frames);plan=dict(sources=sources,pairs=list(pairs.values()));save(L/'metric_plan.json',plan)
 save(D/'reception_audit.json',dict(status='PASS',logical_results=9000,sources=500,methods=6,repetitions=3,unique_metric_pairs=len(pairs),receive_status_counts=statuses,plan=pin(L/'metric_plan.json'),logical_mapping=pin(L/'logical_pair_map.json'),failure_gray_kept=True,source_truth_used=False))
 donor=cr(read(OLD/'dev_scores_registration.json')['metric_donor'])
 r=dict(role='metrics',**gpu(6),cpu_affinity=list(range(6)),dependencies=[dependency('chain')],max_seconds=3600,mode='three',plan=pin(L/'metric_plan.json'),config=donor['config'],frozen_metric_registration=donor['frozen_metric_registration'],original_four=donor['original_four'],torch='2.11.0+cu128',caps=dict(load_lpips=1,load_dreamsim=1,embedding=len(pairs)+500,pair=len(pairs)))
 print(launch(r,'metrics.py'))
if __name__=='__main__':main()
