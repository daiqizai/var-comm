"""Authenticate all 42,000 historical core events; no models or new PHY."""
from followup_common import *
import traceback,collections
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']];out=Path(r['output']);out.mkdir(exist_ok=False)
 os.sched_setaffinity(0,set(r['cpu_affinity']));start=time.monotonic()
 try:
  freeze=cr(r['final_freeze']);assert freeze['configuration']=='original_receiver_plus_D_token'
  ss=cr(freeze['common500_sources']);assert len(ss)==500 and[s['source_index']for s in ss]==list(range(500))
  sm={s['source_id']:source_minimal(s)for s in ss};assert len(sm)==500
  for s in sm.values():
   with np.load(checked(s['archive']),allow_pickle=False)as z:a=z['pixels']
   assert a.dtype==np.uint8 and a.shape==(3,256,256)and hashlib.sha256(a.tobytes()).hexdigest()==s['preprocessing_id']
   assert asha(a.astype(np.float32)/np.float32(255))==s['reference_sha256']
  ids=list(sm);bankinfo=cr(r['statistics_identity']);assert bankinfo['source_order']==ids
  bank=bankinfo['bank'];ba=np.load(checked(bank),allow_pickle=False);assert ba.shape==(10000,500)and ba.min()>=0 and ba.max()<500
  control=np.random.default_rng(2026103004).integers(0,100,size=(10000,100),dtype=np.int64)
  bp=out/'control100_bootstrap.npy'
  with bp.open('xb')as f:np.save(f,control,allow_pickle=False)
  save(D/'source_sets.json',dict(common500=ids,control100=ids[:100],timing32=ids[:32],rule='original common500 full source manifest order',original_manifest=freeze['common500_sources'],bootstrap500=bank,bootstrap100=pin(bp),bootstrap100_seed=2026103004,created_before_new_metric_results=True))
  frames=rows(checked(r['frames']));assert len(frames)==42000;image_cache={};pairs={};logical=[];counts=collections.Counter();events=set()
  gray=np.full((3,256,256),.5,np.float32);gh=asha(gray)
  for i,v in enumerate(frames):
   if time.monotonic()-start>r['seconds']:raise BudgetStop('core audit time cap')
   s=sm[v['source_id']];assert int(v['source_index'])==s['source_index']and v['source_pixels_sha256']==s['preprocessing_id']and v['reference_sha256']==s['reference_sha256']
   key=(v['method'],int(v['N']),int(v['snr_db']),s['source_id'],int(v['channel_repeat']));assert key not in events;events.add(key);counts[key[:3]]+=1
   im=image_spec(v['image']);h=v['pixel_sha256']
   if im is None:assert h==gh
   elif im['path']not in image_cache:
    with np.load(checked(im),allow_pickle=False)as z:a=z['image']
    assert a.dtype==np.float32 and a.shape==(3,256,256)and np.isfinite(a).all()and 0<=a.min()<=a.max()<=1 and asha(a)==h
    image_cache[im['path']]=(im['sha256'],h)
   else:assert image_cache[im['path']]==(im['sha256'],h)
   pk=digest([s['reference_sha256'],h]);pairs.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=h,image=im))
   v=dict(v,dists_pair_key=pk,native_status='VERIFIED_GRAY'if im is None else'VERIFIED_NATIVE_FLOAT32');logical.append(v)
   if(i+1)%3000==0:print('AUDIT_CORE',i+1,len(image_cache),flush=True)
  assert len(counts)==28 and set(counts.values())=={1500}
  save(out/'sources.json',list(sm.values()));save(out/'pairs.json',list(pairs.values()));save(out/'logical.json',logical)
  csvout(D/'input_inventory_core.csv',logical)
  save(out/'completion.json',dict(status='COMPLETE',registration=pin(request),sources=pin(out/'sources.json'),pairs=pin(out/'pairs.json'),logical=pin(out/'logical.json'),logical_count=len(logical),unique_pairs=len(pairs),verified_native_files=len(image_cache),new_model_calls=0,new_PHY_calls=0,seconds=time.monotonic()-start))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
