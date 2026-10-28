"""DINO calibration only; source and pair caching with fixed device/numerics."""
from block_common import *
import tempfile,traceback,ctypes
def main(request,expected):
 assert sha(request)==expected;r=read(request)
 for d in r['code']:checked(d)
 plan=cr(r['plan']);out=Path(r['output']);out.mkdir(exist_ok=False);(out/'references').mkdir();(out/'pairs').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');os.sched_setaffinity(0,set(r['cpu_affinity']))
 ledger=Ledger(out/'calls.sqlite',dict(model_load=1,reference=len(plan['sources']),pair=len(plan['pairs'])));beg=time.monotonic()
 try:
  sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'));import h800_ep_pilot_metrics_v1 as old
  config=r['metric_config'];sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation;assets=relocation.metric_assets(old,config)
  import torch
  torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
  assert torch.__version__=='2.11.0+cu128' and old.flags_function(config['sources']['step0_reference_metrics.py']['path'])(torch)==old.FLAGS
  lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0;buf=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(buf,32,dev)==0 and buf.value.decode().lower()==r['pci'].lower()
  assert torch.cuda.mem_get_info()[0]>=8<<30;torch.cuda.set_per_process_memory_fraction((6<<30)/torch.cuda.get_device_properties(0).total_memory)
  module=old.load_module(config['sources']['metric_models.py'],'_CAB_FROZEN_DINO')
  with module.no_network():evaluator=ledger.call('model_load','once',lambda:module.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
  identity=dict(evaluator_identity=evaluator.identity(),metadata=evaluator.metadata,pci=r['pci'],torch=torch.__version__,numpy=np.__version__,FP32=True,TF32=False,AMP=False,batch=1,threads=6,interop=2)
  save(out/'identity.json',identity);prior_refs={};prior_pairs={}
  for olddir in r.get('prior_score_dirs',[]):
   d=Path(olddir);assert read(d/'identity.json')==identity;done=read(d/'completion.json');assert done['status']=='COMPLETE_DINO'
   prior_refs.update(cr(done['references']));prior_pairs.update(cr(done['scores']))
  refs={};scores={};sources={s['source_id']:s for s in plan['sources']};pixels={};features={}
  with torch.inference_mode(),torch.autocast('cuda',enabled=False):
   for j in plan['pairs']:
    assert time.monotonic()-beg<r['max_seconds'] and torch.cuda.mem_get_info()[0]>=2<<30 and not (L/'STOP').exists()
    sid=j['source_id'];s=sources[sid];refkey=s['reference_sha256'];key=j['pair_key']
    if key in prior_pairs:scores[key]=prior_pairs[key];continue
    if sid not in pixels:
     with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels'].astype(np.float32)/np.float32(255)
     assert asha(x)==refkey;pixels[sid]=x
     if refkey in prior_refs:
      desc=prior_refs[refkey];a=np.load(checked(desc),allow_pickle=False);prepared={'dinov2_vitl14':torch.from_numpy(a)};refs[refkey]=desc
     else:
      def reference():
       v=evaluator.prepare_reference(torch.from_numpy(x[None]).cuda());a=v['dinov2_vitl14'].numpy();p=out/'references'/(refkey+'.npy')
       with p.open('xb')as f:np.save(f,a,allow_pickle=False)
       refs[refkey]=pin(p);return v
      prepared=ledger.call('reference',refkey,reference)
     features[sid]=prepared
    x=pixels[sid]
    if j['image'] is None:y=np.full((3,256,256),.5,np.float32)
    else:
     with np.load(checked(j['image']),allow_pickle=False)as z:y=z['image'].copy()
    assert asha(y)==j['pixel_sha256'] and y.dtype==np.float32 and y.shape==(3,256,256)
    def compute():
     val=evaluator.score(torch.from_numpy(x[None]).cuda(),torch.from_numpy(y[None]).cuda(),[0],label_conditioned=False,prepared=features[sid])[0]['dinov2_vitl14_cosine'];assert np.isfinite(val)
     record=dict(source_id=sid,pair_key=key,pixel_sha256=j['pixel_sha256'],reference_sha256=refkey,value=float(val));save(out/'pairs'/(key+'.json'),record);return record
    scores[key]=ledger.call('pair',key,compute)
    if len(scores)%100==0:print('DINO_PAIRS',len(scores),len(plan['pairs']),flush=True)
  save(out/'references.json',refs);save(out/'scores.json',scores);assert ledger.summary()['unresolved']==0
  save(out/'completion.json',dict(status='COMPLETE_DINO',request=pin(request),scores=pin(out/'scores.json'),references=pin(out/'references.json'),identity=pin(out/'identity.json'),count=len(scores),ledger=ledger.summary(),seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
