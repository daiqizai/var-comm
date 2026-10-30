"""Official native-float DISTS only, exact pair de-duplication and durable ledger."""
from followup_common import *
import importlib.util,traceback,ctypes,tempfile
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(x)for x in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'pairs').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(out/'tmp'));os.sched_setaffinity(0,set(r['cpu_affinity']))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps'])
 try:
  import torch
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_device(0)
  assert torch.__version__==r['torch'];lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0
  bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,dev)==0 and bus.value.decode().lower().lstrip('0')==r['pci'].lower().lstrip('0')
  props=torch.cuda.get_device_properties(0);torch.cuda.set_per_process_memory_fraction((8<<30)/props.total_memory)
  spec=importlib.util.spec_from_file_location('_registered_followup_metrics',checked(r['loader']));met=importlib.util.module_from_spec(spec);spec.loader.exec_module(met)
  identity=dict(loader=r['loader'],assets=r['assets'],torch=torch.__version__,torchvision=met.package_versions()['torchvision'],GPU_UUID=str(props.uuid),PCI=bus.value.decode().lower(),FP32=True,TF32=False,AMP=False,batch=1,resize=False,crop=False,requantize=False,require_grad=False,batch_average=False)
  save(out/'identity.json',identity)
  reuse={}
  for p in r.get('caches',[]):
   c=cr(p);assert c['status']=='COMPLETE'and cr(c['identity'])==identity;reuse.update(cr(c['scores']))
  with met.no_network():model=ledger.call('load_dists','once',lambda:met.load_dists(r['assets'],torch.device('cuda:0')))
  ss={s['source_id']:s for s in cr(r['sources'])};pairs=cr(r['pairs']);scores={};last=None;new=0;times=[]
  priority=set(r.get('priority_pairs',[]))
  with torch.inference_mode(),torch.autocast('cuda',enabled=False):
   for j in sorted(pairs,key=lambda j:(j['pair_key']not in priority,j['source_index'],j['pair_key'])):
    if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('DISTS bounded stage')
    key=j['pair_key'];s=ss[j['source_id']];assert j['reference_sha256']==s['reference_sha256']
    if key in reuse:
     v=reuse[key];assert v['reference_sha256']==s['reference_sha256']and v['pixel_sha256']==j['pixel_sha256'];scores[key]=dict(v,origin='EXACT_SAME_EVALUATOR_PAIR_REUSE');continue
    if last!=s['source_id']:
     with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels'].astype(np.float32)/np.float32(255)
     assert asha(x)==s['reference_sha256'];xt=torch.from_numpy(x[None]).cuda();last=s['source_id']
    if j['image']is None:y=np.full((3,256,256),.5,np.float32)
    else:
     with np.load(checked(j['image']),allow_pickle=False)as z:y=z[j.get('image_key','image')].copy()
     if j.get('image_slot')is not None:y=y[int(j['image_slot'])]
    assert y.dtype==np.float32 and y.shape==(3,256,256)and np.isfinite(y).all()and 0<=y.min()<=y.max()<=1 and asha(y)==j['pixel_sha256']
    yt=torch.from_numpy(y[None]).cuda()
    def score():
     torch.cuda.synchronize();beg=time.monotonic();v=float(model(xt,yt,require_grad=False,batch_average=False).reshape(-1)[0].item());torch.cuda.synchronize();elapsed=time.monotonic()-beg
     assert np.isfinite(v);record=dict(pair_key=key,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=j['pixel_sha256'],dists=v,origin='NEW_OFFICIAL_DISTS_FP32',seconds=elapsed)
     save(out/'pairs'/(key+'.json'),record);times.append(elapsed);return record
    scores[key]=ledger.call('pair',key,score);new+=1
    if new==8:save(out/'initial_cost_observation.json',dict(actual_first_pairs=8,mean_GPU_forward_seconds=float(np.mean(times)),estimated_forward_seconds_for_all=float(np.mean(times)*len(pairs)),not_timing_benchmark=True,actual_pairs_reused_in_final=True))
    if len(scores)%200==0:print('DISTS',len(scores),len(pairs),flush=True)
  assert ledger.summary()['unresolved']==0 and len(scores)==len(pairs)
  save(out/'scores.json',scores);save(out/'completion.json',dict(status='COMPLETE',registration=pin(request),identity=pin(out/'identity.json'),scores=pin(out/'scores.json'),pairs=len(scores),new_pairs=new,reused_pairs=len(scores)-new,ledger=ledger.summary(),seconds=time.monotonic()-start,new_PHY=0,new_reconstruction=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
