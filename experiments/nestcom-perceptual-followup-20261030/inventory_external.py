"""Authenticate the 24,000 external-baseline native events, preserving locators."""
from followup_common import *
import traceback,collections
METHODS={'ARPC_ADAPTED':'ARPC','ARPC':'ARPC','P100000':'Latent_JSCC','Swin200000':'SwinJSCC','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG_LDPC','BPG':'BPG_LDPC'}
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']];out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));beg=time.monotonic()
 try:
  sm={s['source_id']:s for s in cr(r['sources'])};old=cr(r['old_inventory']);seven=cr(r['seven_frames']);frames=[]
  for v in cr(old['frames_file']):
   if v['method']not in METHODS:continue
   s=sm[v['source_id']];assert s['source_index']==v['source_index']and s['reference_sha256']==v['reference_sha256']
   seeds=[6201,6202,6203]if v['method']=='ARPC_ADAPTED'else[2001,2002,2003]
   frames.append(dict(v,method=METHODS[v['method']],historical_method=v['method'],source_pixels_sha256=s['preprocessing_id'],image=v['archive'],channel_repeat=seeds.index(v['noise_seed']),native_origin='HISTORICAL_4_10_19'))
  for v in seven:
   if v['method']not in METHODS:continue
   s=sm[v['source_id']];assert s['source_index']==v['source_index']and v['repeat']in[1,2,3]
   frames.append(dict(v,method=METHODS[v['method']],historical_method=v['method'],source_pixels_sha256=s['preprocessing_id'],reference_sha256=s['reference_sha256'],image_key='image',image_slot=None,channel_repeat=v['repeat']-1,receiver_status=v['receive_status'],native_origin='FORMAL_7_DB_ORIGINAL_REPEAT_ONE_BASED'))
  assert len(frames)==24000;events=set();counts=collections.Counter();files={};arrays={};pairs={};gray=asha(np.full((3,256,256),.5,np.float32))
  for i,v in enumerate(frames):
   if time.monotonic()-beg>r['seconds']:raise BudgetStop('external native audit')
   key=(v['method'],v['snr_db'],v['source_id'],v['channel_repeat']);assert key not in events;events.add(key);counts[key[:2]]+=1
   s=sm[v['source_id']];im=v['image'];h=v['pixel_sha256'];slot=v.get('image_slot');ik=v.get('image_key','image')
   if im is None:assert h==gray
   else:
    p=im['path']
    if p not in files:checked(im);files[p]=im['sha256']
    else:assert files[p]==im['sha256']
    k=(p,ik,slot)
    if k not in arrays:
     with np.load(p,allow_pickle=False)as z:a=z[ik]
     if slot is not None:a=a[int(slot)]
     assert a.dtype==np.float32 and a.shape==(3,256,256)and np.isfinite(a).all()and 0<=a.min()<=a.max()<=1
     arrays[k]=asha(a)
    assert arrays[k]==h
   pk=digest([s['reference_sha256'],h]);v['dists_pair_key']=pk;v['native_status']='VERIFIED_GRAY'if im is None else'VERIFIED_NATIVE_FLOAT32'
   pairs.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=h,image=im,image_key=ik,image_slot=slot))
   if(i+1)%3000==0:print('AUDIT_EXTERNAL',i+1,len(files),flush=True)
  assert len(counts)==16 and set(counts.values())=={1500}
  save(out/'pairs.json',list(pairs.values()));save(out/'logical.json',frames);csvout(D/'input_inventory_external.csv',frames)
  save(out/'completion.json',dict(status='COMPLETE',registration=pin(request),sources=r['sources'],pairs=pin(out/'pairs.json'),logical=pin(out/'logical.json'),logical_count=len(frames),unique_pairs=len(pairs),native_files=len(files),seconds=time.monotonic()-beg,new_models=0,new_PHY=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
