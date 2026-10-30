"""E selection sees source RGB and deterministic successful candidates only."""
from opt_common import *
from metrics import module
import traceback,ctypes

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 choices=cr(r['choices']);outputs=cr(r['receiver_outputs']);bykey={v['key']:v for v in outputs};old=cr(r['metric_registration']);cfg=old['config']
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',dict(load_lpips=1,pair=600));save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 try:
  import torch
  torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  assert torch.__version__==old['torch'];torch.cuda.set_device(0);lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0;bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,dev)==0 and bus.value.decode().lower().lstrip('0')==r['pci'].lower().lstrip('0')
  torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
  met=module(checked(cfg['clip']['loader']),'_OPT_TX_METRIC_LOADER');sys.path.insert(0,str(Path(old['original_four']['path']).parent));four=module(checked(old['original_four']),'_OPT_TX_FOUR');c=cfg['config']['metric_config'];sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation
  assets=relocation.metric_assets(four,c);import lpips
  assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
  load,pair,proof=four.lpips_projection(c['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
  with met.no_network():network=ledger.call('load_lpips','once',lambda:load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
  save(out/'identity.json',dict(torch=str(torch.__version__),GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),PCI=bus.value.decode(),FP32=True,TF32=False,lpips_AST=proof,objective='mean_RGB_squared_error + 0.1 * LPIPS-Alex',tie_tolerance=1e-12,no_DreamSim_CLIP_or_actual_channel_inputs=True))
  cache={};rows=[];all_scores=[]
  with torch.inference_mode():
   for ch in choices:
    if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('E TX-score cap')
    with np.load(checked(ch['reference']),allow_pickle=False)as z:x=z['pixels'].astype(np.float32)/np.float32(255)
    assert asha(x)==ch['source_float_sha256'];xt=torch.from_numpy(x[None]).cuda();scores=[]
    for candidate in ch['candidates']:
     v=bykey[candidate['received_key']];assert v['status']=='ACCEPTED_SOURCE'and v['image']is not None
     key=digest([ch['source_float_sha256'],v['pixel_sha256']])
     if key not in cache:
      with np.load(checked(v['image']),allow_pickle=False)as z:y=z['image'].copy()
      assert asha(y)==v['pixel_sha256']
      def score():
       mse=float(np.square(x.astype(np.float64)-y.astype(np.float64)).mean());lp=float(pair(network,torch.from_numpy(y[None]).cuda(),xt)[0].item());return dict(mse_rgb=mse,lpips_alex=lp,objective=mse+.1*lp)
      cache[key]=ledger.call('pair',key,score)
     scores.append(dict(candidate,**cache[key],pair_key=key));all_scores.append(dict(source_id=ch['source_id'],source_index=ch['source_index'],N=ch['N'],snr_db=ch['snr_db'],**scores[-1]))
    best=scores[0]
    for v in scores[1:]:
     if v['objective']<best['objective']-1e-12:best=v
    rows.append(dict(source_id=ch['source_id'],source_index=ch['source_index'],N=ch['N'],snr_db=ch['snr_db'],source_float_sha256=ch['source_float_sha256'],selected_rank=best['rank'],endpoint=best['endpoint'],received_key=best['received_key'],objective=best['objective']))
  save(out/'scores.json',all_scores);save(out/'selection.json',rows);save(out/'completion.json',dict(status='COMPLETE',source_workpoints=len(rows),unique_pairs=len(cache),selected_shorter=sum(v['selected_rank']>0 for v in rows),selection=pin(out/'selection.json'),ledger=ledger.summary(),seconds=time.monotonic()-start,new_PHY=0,selection_used_actual_channel=False))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
