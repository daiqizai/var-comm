"""New DINO-L values only, identical frozen preprocessing and source features."""
from shared import *
import numpy as np,csv,platform
def main(expected):
 regpath=O/'cal100_score_registration.json';assert sha(regpath)==expected;r=read(regpath)
 for k in ('plan','all_states','prior_metadata','prior_reference_registration','source','native_wait'):verify(r[k])
 root=L/'cal100_scores_v1';out=root/'worker';out.mkdir();cache(out);ledger=None;start=time.time()
 try:
  sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'));import h800_ep_pilot_metrics_v1 as old
  config=r['metric_config'];assets=old.bound_assets(config)
  import torch,ctypes
  torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
  assert old.flags_function(config['sources']['step0_reference_metrics.py']['path'])(torch)==old.FLAGS
  actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads());assert actual==r['expected_runtime']
  sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'));from tb_native import definitions
  ns=dict(ctypes=ctypes);definitions(R/'experiments/entropy-partial-mismatch-20261010/scripts/leo_whole_math_v1.py',{'cuda_driver_pci_bus_id'},ns)
  assert ns['cuda_driver_pci_bus_id']().lower().lstrip('0')==r['gpu_pci'].lower().lstrip('0') and torch.cuda.device_count()==1
  props=torch.cuda.get_device_properties(0);assert torch.cuda.mem_get_info()[0]>=8<<30;torch.cuda.set_per_process_memory_fraction((6<<30)/props.total_memory)
  ledger=Ledger(out/'calls.sqlite',dict(model_load=1,reconstruction_dino=r['new_DINO_pairs']))
  metric=old.load_module(config['sources']['metric_models.py'],'_N2048_adaptive_frozen_DINO')
  with metric.no_network():ev=ledger.call('model_load','DINO_L',lambda:metric.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
  previous=read(r['prior_metadata']['path'])['DINO_L_metadata'];current=dict(ev.metadata)
  previous=dict(previous);previous.pop('metric_selection_usage',None);current.pop('metric_selection_usage',None);assert current==previous,(current,previous)
  put(out/'metric_identity.json',dict(original_DINO_metadata=previous,current_DINO_metadata=current,assets=assets,runtime=actual,flags=old.FLAGS,preprocessing='Original MetricEvaluator; source uint8/255 float32, native CHW float32 unchanged',reference_features='Exact old cal100 DINO features, source archive SHA checked',actual_gpu_name=props.name,actual_pci=ns['cuda_driver_pci_bus_id'](),allocator_bytes=6<<30,new_LPIPS_or_ConvNeXt=0))
  (out/'pairs').mkdir();rows=[]
  for s in read(r['plan']['path']):
   with np.load(verify(s['archive']),allow_pickle=False)as z:pixels=z['pixels'].copy()
   assert pixels.dtype==np.uint8 and pixels.shape==(3,256,256);x=torch.as_tensor((pixels.astype(np.float32)/255)[None],device='cuda:0')
   with np.load(verify(s['reference']),allow_pickle=False)as z:features=z['features'].copy()
   assert features.shape==(1,1024) and features.dtype==np.float32 and np.isfinite(features).all()
   prepared=dict(digest=metric.image_digest(x),batch=1,evaluator_identity=ev.identity(),dinov2_vitl14=torch.as_tensor(features))
   for row in s['pairs']:
    assert time.time()-start<r['max_seconds'] and torch.cuda.mem_get_info()[0]>=2<<30
    with np.load(verify(row['image']),allow_pickle=False)as z:a=z[row['array_key']].copy()
    assert hashlib.sha256(a.tobytes()).hexdigest()==row['image_float_sha256'];y=torch.as_tensor(a[None],device='cuda:0');key=hashlib.sha256((s['source_id']+'|'+row['image_float_sha256']).encode()).hexdigest()
    def score():
     with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):v=float(ev.score(x,y,[0],label_conditioned=False,prepared=prepared)[0]['dinov2_vitl14_cosine'])
     assert np.isfinite(v);put(out/'pairs'/(key+'.json'),dict(source=s['archive'],image=row['image'],source_id=s['source_id'],image_float_sha256=row['image_float_sha256'],dinov2_vitl14_cosine=v));return v
    v=ledger.call('reconstruction_dino',key,score);rows.append(dict(source_id=s['source_id'],image_float_sha256=row['image_float_sha256'],dino=v,pair=pin(out/'pairs'/(key+'.json'))))
   print(json.dumps(dict(sources_scored=s['source_id'],pairs=len(rows),total=r['new_DINO_pairs'])),flush=True)
  assert len(rows)==r['new_DINO_pairs'] and ledger.summary()['unresolved']==0
  put(out/'scores.json',rows);put(out/'completion.json',dict(status='NEW_DINO_ONLY_COMPLETE',scores=pin(out/'scores.json'),ledger=ledger.summary(),new_source_features=0,new_VAR=0,new_channel=0,new_bootstrap=0,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),metric_identity=pin(out/'metric_identity.json')))
 except BaseException:put(out/'failure.json',dict(error=traceback.format_exc(),ledger=None if ledger is None else ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(sys.argv[1])
