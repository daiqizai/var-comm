"""CPU byte/pixel inventory only. Does not import Torch or execute model calls."""
from common import *
import collections,traceback,subprocess

def main():
 out=T/'inventory';out.mkdir(parents=True);O.mkdir()
 save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),started_unix=time.time()))
 inputs=[];sources=[];rows=[];files={};locators={};missing=[]
 def bind(p):
  d=desc(p);inputs.append(d);return read(checked(d))
 try:
  pop=bind(R/'results/arpc_common500_extension_20261019_v1/population.json')
  sr=bind(R/'results/quality_metrics_supplement_20261014_v1/common500/registration.json')
  source_old={s['source_index']:s for s in sr['sources']}
  for s in sorted(pop['common500'],key=lambda x:x['source_index']):
   old=source_old[s['source_index']]
   assert s['source_id']==old['source_id']and s['preprocessing_id']==old['preprocessing_id']
   assert s['archive']['sha256']==old['archive']['sha256']
   v=dict(source_index=s['source_index'],source_id=s['source_id'],archive=s['archive'],image_key='pixels',image_slot=None,
          source_uint8=True,preprocessing_id=s['preprocessing_id'],pixel_sha256=old['reference_rgb_sha256'])
   sources.append(v)
  frames=read(checked(sr['frames']));inputs.append(sr['frames'])
  for v in frames:
   if v['method']not in METHODS[1:4]:continue
   rows.append(dict(method=v['method'],source_index=v['source_index'],source_id=v['source_id'],N=1024,snr_db=v['snr_db'],
       noise_seed=v['noise_seed'],reference_sha256=v['reference_rgb_sha256'],archive=v['archive'],
       image_key=v['image_key'],image_slot=v['image_slot'],expected_pixel_sha256=v['reconstruction_sha256'],
       hash_rule='canonical',receiver_status=v['failure_status'],fixed_gray=bool(v['fixed_gray']),
       lineage='FROZEN_FINAL_100K_200K_BPG',endpoint=None))
  nc=bind(R/'results/n1024_consistency_20261019_v1/repair_v1/delivery/closure_audit.json')
  current=read(checked(nc['metric_rows']));inputs.append(nc['metric_rows'])
  flatpath=R/'results/n1024_consistency_20261019_v1/repair_v1/delivery/frame_scores_9000.csv'
  inputs.append(desc(flatpath))
  with flatpath.open()as f:flat={(int(v['source_index']),int(v['snr_db']),int(v['noise_seed'])):v for v in csv.DictReader(f)if v['family']=='PARTIAL'}
  for v in current:
   if v['strategy_family']!='PARTIAL':continue
   a=v['reconstruction'];f=flat[v['source_index'],v['snr_db'],v['noise_seed']]
   assert a['image_sha256']==f['image_sha256']
   rows.append(dict(method=METHODS[0],source_index=v['source_index'],source_id=v['source_id'],N=1024,snr_db=v['snr_db'],
       noise_seed=v['noise_seed'],reference_sha256=sources[v['source_index']]['pixel_sha256'],archive=a['image_archive'],
       image_key=a['image_key'],image_slot=a.get('image_slot'),expected_pixel_sha256=a['image_sha256'],hash_rule='canonical',
       receiver_status=f['RX_status'],fixed_gray=None,lineage='T1C_M10_CURRENT'if v['snr_db']==19 else 'FROZEN_PARTIAL',
       endpoint=dict(m=v['actual_m'],K=v['actual_K'])))
  for rel in ['results/arpc_external_baseline_20261019_v1/a3_rx_v1/logical_receivers.json','results/arpc_common500_extension_20261019_v1/rx_v1/logical_receivers.json']:
   for v in bind(R/rel):
    assert v['image']is not None,'An unsaved gray requires explicit audit, never manufacture a replacement'
    rows.append(dict(method=METHODS[4],source_index=v['source_index'],source_id=v['source_id'],N=1024,snr_db=v['snr_db'],
       noise_seed=v['noise_seed'],reference_sha256=sources[v['source_index']]['pixel_sha256'],archive=v['image'],
       image_key='float_output',image_slot=None,expected_pixel_sha256=v['image_float_sha256'],hash_rule='rawbytes',
       receiver_status=v['actual_status']+'/'+v['receiver_status'],fixed_gray=False,
       lineage='ORIGINAL_A3_100'if '/a3_rx_'in rel else 'COMMON500_EXTENSION_400',endpoint=v['actual_rx_endpoint']))
  grid(rows,sources)
  grouped={}
  for v in sources+rows:
   p=v['archive']['path'];grouped.setdefault(p,[]).append(v)
  for count,(p,vs)in enumerate(sorted(grouped.items()),1):
   d=vs[0]['archive'];assert all(v['archive']['sha256']==d['sha256']for v in vs)
   if not Path(p).is_file():
    missing.append(dict(path=p,logical_positions=len([v for v in vs if 'method'in v])));continue
   a=Path(p).stat();b=Path(p).read_bytes();z=Path(p).stat()
   assert(a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)and hashlib.sha256(b).hexdigest()==d['sha256'],p
   assert 'bytes'not in d or len(b)==d['bytes']
   files[p]=dict(path=p,bytes=len(b),sha256=d['sha256'],mtime_ns=z.st_mtime_ns)
   with np.load(io.BytesIO(b),allow_pickle=False)as npz:
    for v in vs:
     ar=npz[v['image_key']]
     if v['image_slot']is not None:ar=ar[v['image_slot']]
     if v.get('source_uint8'):
      assert ar.dtype==np.uint8 and rawsha(ar)==v['preprocessing_id']
      ar=ar.astype(np.float32)/np.float32(255)
     ar=pixels(ar);ps=ash(ar);raw=rawsha(ar)
     expect=v.get('expected_pixel_sha256',v.get('pixel_sha256'))
     assert (raw if v.get('hash_rule')=='rawbytes'else ps)==expect,(p,v.get('method'))
     v['pixel_sha256']=ps
     gray=bool(np.all(ar==np.float32(.5)))
     if 'method'in v:
      if v['fixed_gray']is not None:assert v['fixed_gray']==gray
      v['fixed_gray']=gray
     locator={k:v[k]for k in ['archive','image_key','image_slot','pixel_sha256']}
     if v.get('source_uint8'):locator.update(source_uint8=True,preprocessing_id=v['preprocessing_id'])
     locators.setdefault(ps,locator)
   if count%500==0:print('INVENTORIED_FILES',count,'/',len(grouped),flush=True)
  save(out/'missing.json',missing)
  assert not missing,('MISSING_NATIVE_ARRAYS',len(missing))
  bank=bind(R/'results/n1024_consistency_20261019_v1/completion_report_v1/actual_compute_and_reuse.json')['bootstrap']
  checked(bank['indices'])
  ix=np.load(bank['indices']['path'],allow_pickle=False)
  assert ix.shape==(10000,500)and ix.dtype==np.int64 and ash(ix)==bank['array_sha256']and bank['seed']==2026100701
  rows.sort(key=key);grid(rows,sources)
  save(out/'sources.json',sources);save(out/'frames.json',rows);save(out/'feature_inputs.json',locators);save(out/'verified_files.json',list(files.values()))
  counts=[]
  for m in METHODS:
   for snr in SNRS:
    group=[v for v in rows if v['method']==m and v['snr_db']==snr]
    counts.append(dict(method=m,snr_db=snr,logical_pairs=len(group),sources=len({v['source_id']for v in group}),
       fixed_gray=sum(v['fixed_gray']for v in group),unique_pixels=len({v['pixel_sha256']for v in group})))
  summary=dict(status='NATIVE_ARRAY_INVENTORY_COMPLETE',logical_pairs=22500,sources=500,unique_feature_inputs=len(locators),
       unique_reconstruction_pixels=len({v['pixel_sha256']for v in rows}),
       unique_source_reconstruction_pairs=len({(v['reference_sha256'],v['pixel_sha256'])for v in rows}),
       counts=counts,missing_files=0,verified_files=len(files),verified_bytes=sum(v['bytes']for v in files.values()),
       bootstrap=bank,inputs=inputs,sources_file=desc(out/'sources.json'),frames_file=desc(out/'frames.json'),
       feature_inputs=desc(out/'feature_inputs.json'),files=desc(out/'verified_files.json'),
       new_model_calls=0,new_channel=0,new_reconstruction=0,original_scores_changed=False)
  save(out/'completion.json',summary);save(O/'inventory.json',summary)
  print(json.dumps({k:v for k,v in summary.items()if k in ('status','logical_pairs','unique_feature_inputs','counts','verified_bytes')}),flush=True)
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),new_model_calls=0,new_channel=0,new_reconstruction=0));raise
if __name__=='__main__':main()
