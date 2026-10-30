"""Audit immutable historical regression rows; never infer missing metric values."""
from opt_common import *
import csv,collections

METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
def rows(path):
 with Path(path).open()as f:return list(csv.DictReader(f))

def main(request):
 r=read(request);sources=cr(r['sources']);sm={s['source_id']:s for s in sources};assert len(sm)==500
 out=L/r['output'];assert not out.exists();evidence=[];result=[];verified={};physical={}
 def load(p):evidence.append(pin(p));return read(p)
 def table(p):evidence.append(pin(p));return rows(p)
 def image_check(desc,h):
  if desc is None:assert h==asha(np.full((3,256,256),.5,np.float32));return
  key=(desc['path'],desc['sha256'])
  if key not in verified:
   with np.load(checked(desc),allow_pickle=False)as z:a=z['image'].copy()
   assert a.shape==(3,256,256)and a.dtype==np.float32 and np.isfinite(a).all()and 0<=a.min()<=a.max()<=1
   verified[key]=asha(a)
  assert verified[key]==h
 def add(v,N,method,image,h,origin,reference=None,extra=None):
  s=sm[v['source_id']];assert int(v['source_index'])==s['source_index']
  if reference is not None:assert reference==s['reference_sha256']
  image_check(image,h);values={m:float(v[m])for m in METRICS if m in v and v[m]not in('',None)}
  assert all(np.isfinite(x)for x in values.values())
  result.append(dict(source_id=s['source_id'],source_index=s['source_index'],source_pixels_sha256=s['preprocessing_id'],reference_sha256=s['reference_sha256'],N=N,snr_db=int(v['snr_db']),noise_seed=int(v['noise_seed']),channel_repeat=int(v['noise_seed'])-6201,method=method,variant='original'if method!='ADA'else'ADA_frozen',pair_key=digest([s['reference_sha256'],h]),image=image,pixel_sha256=h,metric_origin=origin,**values,**(extra or{})))
 def old_frame(N,v,desc):
  f=cr(desc);t=f['transmission'];source_id=v['source_id'];assert source_id in sm
  payload_hash=f.get('payload_sha256',f.get('job',{}).get('payload_sha256'))
  if payload_hash is None:payload_hash=asha(np.asarray(f['actual_RX']['body']['payload'],np.uint8))
  identity=dict(N=N,source_id=source_id,source_index=int(v['source_index']),snr_db=int(v['snr_db']),noise_seed=int(v['noise_seed']),counter=t['public_frame_counter'],profile_id=t['profile_id'],payload_sha256=payload_hash)
  key=digest(identity)
  if key in physical:assert physical[key]['identity']==identity
  else:physical[key]=dict(physical_key=key,frame=desc,identity=identity)
 # Complete original N512 formal records and their native arrays.
 n=B/'nestcom_n512_20261028_v1';rx={}
 for v in load(n/'formal_rx_plan/reuse.json').values():rx[v['record']['key']]=v['record']
 for kind in ('nest','ada'):
  for i in range(4):
   p=n/f'formal_rx_{kind}_{i:02d}'/'outputs.json'
   if p.exists():
    for v in load(p):rx[v['key']]=v
 frames=load(n/'formal_rx_plan/frames.json')
 for v in table(R/'results/nestcom_n512_20261028_v1/delivery/frame_scores.csv'):
  rec=rx.get(v['received_key'])
  if rec is None:
   f=cr(frames[v['physical_key']]);assert v['fixed_gray']=='True'and f['projection']['status']!='SOURCE_INPUT'
   rec=dict(image=None,pixel_sha256=asha(np.full((3,256,256),.5,np.float32)))
  assert rec['pixel_sha256']==v['pixel_sha256']
  add(v,512,v['method'],rec['image'],v['pixel_sha256'],'EXACT_N512_FORMAL',v['reference_sha256'],dict(receiver_status=v['receiver_status'],endpoint=[int(v['actual_m']),int(v['actual_K'])],source_bits=int(v['arithmetic_bits']),capacity_bits=int(v['arithmetic_capacity_bits']),utilization=float(v['utilization'])))
  if v['method']!='ADA':old_frame(512,v,frames[v['physical_key']])
 # N1024 source ID and raw/native pixel hash proof; indices alone never join.
 identities=load(R/'results/adatoken_response_20261026_v1/c_predclass_formal_delivery_v1/common500/source_identity_checks.json')
 for v in identities:
  s=sm[v['source_id']];assert s['source_index']==v['source_index']and s['preprocessing_id']==v['uint8_pixels_sha256']and s['reference_sha256']==v['reference_float_sha256']
 clip=table(B/'clip_i_common500_20261020_v1/score/frame_scores.csv')
 cliprows={(v['source_id'],int(v['snr_db']),int(v['noise_seed'])):v for v in clip if v['method']=='EC_VAR_PARTIAL'}
 clip_pairs={}
 for v in clip:
  if v['method']=='EC_VAR_PARTIAL':
   s=sm[v['source_id']];assert s['reference_sha256']==v['reference_sha256'];clip_pairs[(s['source_id'],v['pixel_sha256'])]=float(v['clip_image_cosine'])
 original=load(B/'n1024_consistency_20261019_v1/repair_v1/delivery/metric_rows_9000.json')
 for v in original:
  rec=v['reconstruction'];assert rec['image_key']=='image'and rec['image_slot']is None
  method='NEST'if v['strategy_family']=='PARTIAL'else'WHOLE';metric=dict(v,**v['metrics']);h=rec['image_sha256']
  if method=='NEST':
   c=cliprows[(v['source_id'],v['snr_db'],v['noise_seed'])];assert c['pixel_sha256']==h;metric['clip_image_cosine']=float(c['clip_image_cosine'])
  elif(v['source_id'],h)in clip_pairs:metric['clip_image_cosine']=clip_pairs[(v['source_id'],h)]
  f=cr(v['physical_frame']);cap=f['transmission']['k']-29
  add(metric,1024,method,rec['image_archive'],h,'EXACT_N1024_HISTORICAL_H800',extra=dict(receiver_status=f['actual_RX']['status'],endpoint=[v['actual_m'],v['actual_K']],source_bits=v['arithmetic_bits'],capacity_bits=cap,utilization=v['arithmetic_bits']/cap))
  old_frame(1024,v,v['physical_frame'])
 # Frozen predicted-class Ada, exact native image + four historical metrics.
 a=B/'adatoken_response_20261026_v1/c_predclass_common500_metric_preparation';plan=load(a/'three_plan.json');af={(v['source_id'],v['snr_db'],v['noise_seed']):v for v in plan['frames']}
 for s in plan['sources']:assert s['reference_sha256']==sm[s['source_id']]['reference_sha256']and s['preprocessing_id']==sm[s['source_id']]['preprocessing_id']
 for v in table(R/'results/adatoken_response_20261026_v1/c_predclass_formal_delivery_v1/common500/frame_metrics.csv'):
  f=af[(v['source_id'],int(v['snr_db']),int(v['noise_seed']))];assert f['pixel_sha256']==v['pixel_sha256']
  add(v,1024,'ADA',f['image'],f['pixel_sha256'],'EXACT_PREDCLASS_FORMAL',f['reference_sha256'],dict(receiver_status=v['source_status'],received_prefix=int(v['received_prefix'])))
 # The registered 7 dB delivery has three metrics; do not invent CLIP-I.
 seven=B/'nestcom_7db_common500_20261030_v1';seven_frames={(v['source_id'],v['repeat']):v for v in load(seven/'phy_nest/frames.json')}
 for v in table(R/'results/nestcom_7db_common500_20261030_v1/delivery/frame_metrics.csv'):
  if v['method']not in ('NEST','ADA_PREDCLASS_FIXEDFRAME'):continue
  p=Path(v['image_path']);desc=dict(path=str(p),bytes=p.stat().st_size,sha256=v['image_file_sha256'])if v['image_path']not in ('','FIXED_GRAY')else None
  method='NEST'if v['method']=='NEST'else'ADA';extra=dict(receiver_status=v['receive_status'])
  if method=='NEST':
   f=seven_frames[(v['source_id'],int(v['repeat']))];extra.update(endpoint=f['endpoint'],source_bits=f['source_bits'],capacity_bits=f['capacity_bits'],utilization=f['source_bits']/f['capacity_bits']);old_frame(1024,v,pin(seven/'phy_nest'/(f['key']+'.json')))
  add(v,1024,method,desc,v['pixel_sha256'],'EXACT_7DB_THREE_METRICS_CLIP_MISSING',extra=extra)
 counts=collections.Counter((v['method'],v['N'],v['snr_db'])for v in result);assert len(result)==30000 and all(n==1500 for n in counts.values()),counts
 missing={}
 for v in result:
  if'clip_image_cosine'not in v:missing.setdefault(v['pair_key'],{k:v[k]for k in ['pair_key','source_id','source_index','image','pixel_sha256']})
 out.mkdir();save(out/'frames.json',result);save(out/'physical_index.json',list(physical.values()))
 save(out/'missing_clip_plan.json',dict(sources=[{k:s[k]for k in ['source_id','source_index','archive','reference_sha256']}for s in sources],pairs=list(missing.values())))
 save(out/'completion.json',dict(status='HISTORICAL_INPUTS_AUDITED',registration=pin(request),frames=len(result),unique_arrays_verified=len(verified),missing_clip_pairs=len(missing),missing_logical=sum('clip_image_cosine'not in v for v in result),counts=[dict(method=k[0],N=k[1],snr_db=k[2],logical=n)for k,n in counts.items()],inputs=evidence,new_models=0,new_PHY=0,new_metric_calls=0,old_numeric_routes_preserved=True,historical_H800_not_claimed_bitwise_equal_to_new_PRO=True))
 print(dict(frames=len(result),arrays=len(verified),missing_clip_pairs=len(missing),physical_reuse_index=len(physical)))

if __name__=='__main__':main(*sys.argv[1:])
