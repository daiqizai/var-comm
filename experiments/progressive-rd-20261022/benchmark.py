"""32 training sources x two real entropy/RX endpoints; 50 decoder updates.

Reuses exact historical RGB/tokens, current independent arithmetic decoder and
unchanged predict-overwrite-advance receiver. No PHY, labels, held-out data,
learned selector, encoder update, conditional decoder or production training.
"""
from common import *
import argparse, copy, csv, signal, sys, traceback, types

def main(regsha):
 regpath=O/'stage0_registration.json';assert sha(regpath)==regsha
 reg=read(regpath)
 for d in reg['new_sources']+reg['frozen_sources']:checked(d)
 for d in reg['assets']:checked(d)
 out=L/'benchmark_v1';out.mkdir(exist_ok=False)
 for k in ('tmp','inductor','triton'): (out/'runtime_cache'/k).mkdir(parents=True)
 os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'))
 import tempfile;tempfile.tempdir=os.environ['TMPDIR']
 ledger=Ledger(out/'ledger.jsonl');started=time.perf_counter();stop=[False]
 def stopfn(*_):stop[0]=True
 signal.signal(signal.SIGTERM,stopfn);signal.signal(signal.SIGINT,stopfn)
 def boundary():
  if stop[0]:raise InterruptedError('Owned benchmark stop requested')
  if time.perf_counter()-started>reg['walltime_seconds']:raise TimeoutError('Registered benchmark wall time')
  import subprocess
  if torch.cuda.mem_get_info()[0] < 2<<30:raise RuntimeError('Device free memory below 2 GiB')
 import numpy as np,torch
 from pathlib import Path
 torch.set_num_threads(6);torch.set_num_interop_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
 assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and torch.cuda.device_count()==1
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((48<<30)/torch.cuda.get_device_properties(0).total_memory)
 torch.manual_seed(reg['seed']);np.random.seed(reg['seed'])
 sys.path[:0]=[str(R/'experiments/entropy-partial-mismatch-20261010/scripts'),str(R/'experiments/wcl-evidence-closure-20261009/scripts'),str(R/'src')]
 from ep_source_codec import PartialSourceCodec,endpoint,quarter_counts
 from t1_codec_runtime import MODEL_STATES
 for m,k in reg['candidate_endpoints']:
  if m==10:assert k==0
  else:assert endpoint(m,k)==(m,k)
 ns=dict(torch=torch,Path=Path,sys=sys,hashlib=hashlib,PATCH_NUMS=SIZES,VOCAB_SIZE=4096)
 definitions(R/'src/var_comm/next_scale_prior.py',{'load_models','state_sha256'},ns)
 definitions(R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',{'ContinuousDecoder'},dict_ns:=dict(torch=torch,nn=torch.nn,copy=copy))
 routes=read(checked(reg['model_routes']));paths={'var_source':routes['author_root']}
 for d in routes['files']:
  p=checked(d)
  paths['vae_checkpoint' if p.name=='vae_ch160v4096z32.pth' else 'var_checkpoint' if p.name=='var_d16.pth' else 'decoder_checkpoint']=str(p)
 def load():
  vae,var=ns['load_models'](paths,torch.device('cuda:0'))
  decoder=dict_ns['ContinuousDecoder'](vae).cuda();cp=torch.load(paths['decoder_checkpoint'],map_location='cpu',weights_only=True)
  decoder.load_state_dict(cp['decoder'],strict=True);decoder.eval().requires_grad_(False)
  state={k:ns['state_sha256'](m) for k,m in [('vae',vae),('var',var),('decoder',decoder)]}
  assert state==MODEL_STATES,(state,MODEL_STATES)
  return vae,var,decoder,state
 vae,var,decoder,states=ledger.call('model_load','frozen_models',load)
 pn=dict(torch=torch,np=np,SIZES=SIZES);definitions(R/'experiments/scale-causal-partial-residual-20261002/partial_receiver.py',{'_Prior'},pn)
 captured={}
 class CaptureDecoder:
  def __call__(self,z):
   captured['latent']=z.detach().float().cpu().clone()
   return decoder(z)
 native=types.SimpleNamespace(torch=torch,receiver=types.SimpleNamespace(_Prior=pn['_Prior']),loaded=dict(vae=vae,var=var,decoder=CaptureDecoder(),device=torch.device('cuda:0'),identity=dict(models=states)),flags=dict(deterministic=True,matmul_tf32=False,cudnn_tf32=False,precision='highest'))
 hp=dict(np=np,time=time,SIZES=SIZES,token_count=lambda m,K=0:OFFSETS[m]+K,require=lambda ok,msg:None if ok else (_ for _ in ()).throw(RuntimeError(msg)))
 definitions(R/'experiments/content-real-64qam-20261006/h_source_driver.py',{'render_received'},hp)
 codec=PartialSourceCodec(R,native=native)
 def load_lpips():
  import lpips
  lp=lpips.LPIPS(net='alex',pnet_rand=True,model_path=str(checked(reg['lpips_linear'])),verbose=False)
  weights=torch.load(checked(reg['alexnet']),map_location='cpu',weights_only=True)
  lp.net.load_state_dict({n:weights['features.'+n.split('.',1)[1]] for n in lp.net.state_dict()},strict=True)
  lp=lp.cuda().eval().requires_grad_(False)
  assert ns['state_sha256'](lp)==reg['lpips_state_sha256']
  return lp
 lp=ledger.call('metric_load','lpips',load_lpips)
 from pytorch_msssim import ms_ssim
 save(out/'native_identity.json',dict(models=states,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),gpu=torch.cuda.get_device_name(0),FP32=True,TF32=False,deterministic=True,threads=6,interop=2,affinity=sorted(os.sched_getaffinity(0)),generation='class1000_greedy_no_CFG',cache_latent_precision='FP32',source_truth_used_by_RX=False))
 rows=[];latents=[];targets=[];source_io=0.;selected=reg['selected_sources'];endpoints=[tuple(x) for x in reg['benchmark_endpoints']]
 for row in selected:
  boundary();t=time.perf_counter();rgb=torch.load(checked(row['rgb_shard']),map_location='cpu',weights_only=True);old=torch.load(checked(row['latent_shard']),map_location='cpu',weights_only=True)
  j=row['within_shard'];sid=row['source_id'];assert rgb['image_ids'][j]==old['image_ids'][j]==sid
  assert int(old['source_indices'][j])==row['source_index'];pixels=rgb['targets_u8'][j].contiguous();tokens=old['full_tokens'][j].numpy().astype(np.int64);target=pixels.float().div(255);del rgb,old
  assert pixels.dtype==torch.uint8 and list(pixels.shape)==[3,256,256] and tokens.shape==(680,)
  source_io+=time.perf_counter()-t;key=f"i{row['source_index']:05d}";source_out=out/key;source_out.mkdir()
  pixelsha=hashlib.sha256(pixels.numpy().tobytes()).hexdigest();toksha=hashlib.sha256(tokens.astype('<u2').tobytes()).hexdigest()
  enc=ledger.call('tx_probability_pass',key,lambda:codec.encode_endpoints(tokens,endpoints))
  for m,K in endpoints:
   boundary();stem=f'{key}_m{m}_k{K}';bits=enc[m,K]['bits']
   # Save TX output before RX; receiver functions only receive actual bits/profile.
   with (source_out/f'm{m}_k{K}_bits.npy').open('xb') as f:np.save(f,bits,allow_pickle=False)
   family='EC_VAR_PARTIAL' if K else 'EC_VAR_WHOLE'
   rx=ledger.call('independent_source_decode',stem,lambda:codec.decode(family,bits,m,K))
   with torch.no_grad():image=ledger.call('predict_overwrite_advance',stem,lambda:hp['render_received'](native,rx['received_tokens'],m,K))
   # Source truth is consulted only after the independent output is fixed.
   assert np.array_equal(rx['received_tokens'],tokens[:OFFSETS[m]+K]),'Independent lossless roundtrip differs'
   z=captured.pop('latent');assert tuple(z.shape)==(1,32,16,16) and torch.isfinite(z).all()
   latentdesc=write_torch(source_out/f'm{m}_k{K}_latent.pt',dict(latent=z,source_id=sid,m=m,K=K,registration_sha256=regsha),torch)
   pred=torch.from_numpy(image)[None].cuda();truth=target[None].cuda()
   with torch.no_grad():
    lpv=ledger.call('lpips_distance',stem,lambda:float(lp(pred*2-1,truth*2-1).item()))
    msv=ledger.call('ms_ssim',stem,lambda:float(ms_ssim(pred,truth,data_range=1,size_average=True).item()))
   mse=float(np.mean((image.astype(np.float64)-target.numpy().astype(np.float64))**2))
   item=dict(source_id=sid,source_index=row['source_index'],split='train',m=m,K=K,token_count=OFFSETS[m]+K,terminated_bits=int(len(bits)),finish_bits=int(enc[m,K]['flush_bits']),bits_per_pixel=float(len(bits)/65536),mse=mse,psnr_db=float(-10*np.log10(mse)),lpips_alex=lpv,ms_ssim=msv,status='LOSSLESS_SOURCE_RX_COMPLETE',source_pixels_sha256=pixelsha,full_tokens_uint16_sha256=toksha,latent_path=latentdesc['path'],latent_sha256=latentdesc['sha256'],image_array_sha256=hashlib.sha256(image.tobytes()).hexdigest())
   save(source_out/f'm{m}_k{K}.json',item);rows.append(item);latents.append(z[0]);targets.append(target)
  save(out/f'progress_{len(rows):03d}.json',dict(completed_sources=len(rows)//2,completed_endpoints=len(rows),elapsed_seconds=time.perf_counter()-started,source_io_seconds=source_io))
  print(json.dumps(dict(completed_sources=len(rows)//2,endpoint_rows=len(rows),elapsed_seconds=time.perf_counter()-started)),flush=True)
 cache_seconds=sum(c['elapsed_seconds'] for c in ledger.calls if c['kind'] in ['tx_probability_pass','independent_source_decode','predict_overwrite_advance','lpips_distance','ms_ssim'])
 with (out/'source_rd.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(out/'cache_completion.json',dict(sources=32,endpoints=64,source_io_seconds=source_io,scientific_seconds=cache_seconds,rows=pin(out/'source_rd.csv'),training_started=False))
 # Explicitly bounded throughput pilot; this checkpoint is not a selected method.
 decoder.train().requires_grad_(True);opt=torch.optim.AdamW(decoder.parameters(),lr=1e-5,weight_decay=.01)
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
 gen=torch.Generator().manual_seed(reg['seed']);order=torch.randperm(len(latents),generator=gen);cursor=0;epochs=0;updates=[]
 data=torch.stack(latents);targetdata=torch.stack(targets);step=0
 def checkpoint(reason):
  return write_torch(out/f'benchmark_step_{step:04d}.pt',dict(model_name='nest_multiprefix_unconditional_throughput_pilot_v1',decoder=decoder.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=None,global_step=step,processed_samples=step*4,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=dict(name=np.random.get_state()[0],keys=np.random.get_state()[1].tolist(),position=np.random.get_state()[2],has_gauss=np.random.get_state()[3],cached_gaussian=np.random.get_state()[4]),data_order=order,cursor=cursor,epochs=epochs,order_rng=gen.get_state(),config=reg,calibration=None,calibration_status='NOT_RUN_THROUGHPUT_ONLY',reason=reason),torch)
 initial_checkpoint=checkpoint('bounded_pilot_initial_state')
 last_save=time.perf_counter()
 try:
  for next_step in range(1,51):
   boundary()
   if cursor+4>len(order):order=torch.randperm(len(latents),generator=gen);cursor=0;epochs+=1
   ids=order[cursor:cursor+4];cursor+=4
   def update():
    torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
    z=data[ids].cuda();target=targetdata[ids].cuda();pred=decoder(z)
    mse=(pred-target).square().mean();pl=lp(pred*2-1,target*2-1).mean();loss=mse+.1*pl
    assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.0);assert torch.isfinite(grad)
    opt.step();scheduler.step();torch.cuda.synchronize()
    return dict(step=next_step,samples=4,loss=float(loss.detach()),mse=float(mse.detach()),lpips=float(pl.detach()),seconds=time.perf_counter()-t)
   updates.append(ledger.call('decoder_update',str(next_step),update));step=next_step
   if step%10==0:print(json.dumps(updates[-1]),flush=True)
   if time.perf_counter()-last_save>600 and step<50:checkpoint('periodic_safe_update_boundary');last_save=time.perf_counter()
 except (InterruptedError,TimeoutError):
  if step>0 and not (out/f'benchmark_step_{step:04d}.pt').exists():checkpoint('safe_boundary_stop')
  save(out/'paused.json',dict(completed_updates=step,automatic_restart=False));raise
 except BaseException:
  # Training failure is retained; no automatic scientific retry.
  raise
 t=time.perf_counter();cp=checkpoint('bounded_50_update_completion');save_seconds=time.perf_counter()-t
 assert ns['state_sha256'](vae)==states['vae'] and ns['state_sha256'](var)==states['var']
 assert ns['state_sha256'](decoder)!=states['decoder']
 for d in reg['new_sources']+reg['frozen_sources']:checked(d)
 save(out/'training_updates.json',updates)
 save(out/'completion.json',dict(status='STAGE0_BENCHMARK_COMPLETE',registration_sha256=regsha,source_count=32,endpoint_rows=64,encoder_calls=0,reused_full_token_sources=32,tx_probability_passes=32,independent_source_decodes=64,real_receiver_completions=64,metric_rows=64,training_updates=50,training_samples=200,training_batch=4,calibration_calls=0,PHY_calls=0,DDP_status='NOT_RUN_SINGLE_GPU_BASELINE_FIRST',long_training='NOT_STARTED_RESOURCE_WINDOW_UNKNOWN',cache_scientific_seconds=cache_seconds,source_io_seconds=source_io,training_seconds=sum(x['seconds'] for x in updates),steady_last45_samples_per_second=180/sum(x['seconds'] for x in updates[5:]),checkpoint=cp,checkpoint_save_seconds=save_seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),total_seconds=time.perf_counter()-started,ledger=pin(out/'ledger.jsonl')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);a=p.parse_args()
 try:main(a.registration_sha256)
 except BaseException as e:
  out=L/'benchmark_v1';out.mkdir(parents=True,exist_ok=True)
  if not (out/'failure.json').exists():save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),process=process(os.getpid())))
  raise
