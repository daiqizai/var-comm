"""Display-only export of the two already chosen sources at the first event."""
from study_common import *
import ast,csv,copy,traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']+r['inputs']]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));beg=time.monotonic()
 ledger=Ledger(out/'calls.sqlite',dict(display_export=6))
 try:
  from PIL import Image
  assert read(L/'calibration_chain/completion.json')['status']=='COMPLETE_SCIENCE'
  base=cr(r['original_mapping']);new=copy.deepcopy(base);sources=read(L/'cal_formal_plan/sources.json');sm={s['source_id']:s for s in sources}
  with checked(r['formal_frames']).open(newline='')as f:frames=list(csv.DictReader(f))
  name={'WHOLE_final_receiver':'WHOLE','final_NeST':'NeST','Ada_frozen':'Ada'}
  assert [v['source_id']for v in base if v['method']=='Source']==r['fixed_source_ids']
  display=out/'png';display.mkdir()
  for v in new:
   assert v['source_index']in [135,201]and v['source_id']in r['fixed_source_ids']
   s=sm[v['source_id']];assert v['reference_sha256']==s['reference_sha256']and v['source_pixels_sha256']==s['preprocessing_id']
   if v['method']not in name:continue
   rr=[f for f in frames if f['source_id']==v['source_id']and f['method']==name[v['method']]and int(f['snr_db'])==10 and int(f['channel_repeat'])==0]
   assert len(rr)==1;f=rr[0];assert int(f['noise_seed'])==6201 and int(f['source_index'])==v['source_index']
   desc=ast.literal_eval(f['image'])if f['image']not in ['','None']else None
   def export():
    if desc is None:y=np.full((3,256,256),.5,np.float32)
    else:
     with np.load(checked(desc),allow_pickle=False)as z:y=z['image'].copy()
    assert y.shape==(3,256,256)and y.dtype==np.float32 and asha(y)==f['pixel_sha256']
    assert np.isfinite(y).all()and 0<=y.min()<=y.max()<=1
    rgb=np.rint(y.transpose(1,2,0)*255).astype(np.uint8);path=display/f"s10_i{v['source_index']}_{name[v['method']]}.png";Image.fromarray(rgb).save(path)
    return dict(native_pixel_sha256=f['pixel_sha256'],png=pin(path),display_RGB_sha256=asha(rgb),quantization='nearest uint8 from native float; no crop or enhancement')
   proof=ledger.call('display_export',v['source_id']+'/'+name[v['method']],export)
   v.update(**{m:float(f[m])for m in METRICS},pixel_sha256=f['pixel_sha256'],image=desc,image_key=None,image_slot=None,receiver_status=f['receiver_status'],noise_seed=6201,channel_repeat=0,
            unified_method=name[v['method']],endpoint=[int(f['m']),int(f['K'])],q=int(f['q']),rate=f['rate'],display_origin=proof)
   v['display']=dict(path='figures/individual/fig3_unified_20261103/'+Path(proof['png']['path']).name,sha256=proof['png']['sha256'],bytes=proof['png']['bytes'])
  selection=dict(base_selection=r['original_selection'],fixed_source_ids=r['fixed_source_ids'],selection='Original user choices retained before new calibration; no quality-based reselection',N=1024,snr_db=10,noise_seeds=dict(WHOLE=6201,NeST=6201,Ada=6201,SwinJSCC=2001),all_first_registered_repeats=True,native_metric_source=r['formal_frames'],frozen_policy=pin(D/'frozen_policies.json'),display_only=True,new_scientific_calls=0)
  save(out/'native_mapping.json',new);save(out/'selection.json',selection)
  dest=D/'figure3';dest.mkdir(exist_ok=False);save(dest/'native_mapping.json',new);save(dest/'selection.json',selection)
  csvout(dest/'display_metrics.csv',[{k:v[k]for k in ['source_id','source_index','method','row','column','display_method','channel_repeat','noise_seed',*METRICS]}for v in new if v['method']!='Source'])
  save(out/'completion.json',dict(status='COMPLETE',ledger=ledger.summary(),seconds=time.monotonic()-beg,new_render=0,new_metrics=0,new_PHY=0,display_pngs=6,source_ids=r['fixed_source_ids'],mapping=pin(out/'native_mapping.json')))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
