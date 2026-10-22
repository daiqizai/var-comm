"""One disjoint source shard; immutable per-source result commit, no retry loop."""
from rdscale_common import *
import argparse,signal,sys,traceback,csv
def main(regsha,shard):
 assert sha(O/'registration.json')==regsha;reg=read(O/'registration.json')
 for d in reg['sources']:checked(d)
 out=L/'validation_cache_v1'/f'shard_{shard:02d}';out.mkdir(parents=True,exist_ok=False)
 for k in ['tmp','inductor','triton']:(out/'runtime_cache'/k).mkdir(parents=True)
 os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'))
 import tempfile;tempfile.tempdir=os.environ['TMPDIR']
 started=time.perf_counter();stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
 import torch,numpy as np
 def boundary():
  if stop[0]:raise InterruptedError('Owned shard stop requested')
  if time.time()>=reg['absolute_deadline_unix']:raise TimeoutError('New bounded scale-up deadline')
  if time.perf_counter()-started>reg['validation_walltime_seconds']:raise TimeoutError('Bounded source shard limit')
 ledger=Ledger(out/'ledger.jsonl')
 from rdscale_runtime import load_runtime,load_lpips
 pci=reg['resources'][shard+6]['pci_bus_id']
 n,render,codec,capture,counter=ledger.call('model_load','frozen_models',lambda:load_runtime(reg,out,boundary,pci))
 lp=ledger.call('metric_load','lpips',lambda:load_lpips(reg))
 from pytorch_msssim import ms_ssim
 from PIL import Image
 endpoints=[tuple(x) for x in reg['endpoints']];selected=reg['validation_sources'][shard::2];allrows=[];source_seconds=[];io_seconds=0
 for slot,row in enumerate(selected):
  boundary();t=time.perf_counter();rgb=torch.load(checked(row['rgb_shard']),map_location='cpu',weights_only=True);old=torch.load(checked(row['latent_shard']),map_location='cpu',weights_only=True)
  j=row['within_shard'];sid=row['source_id'];assert rgb['image_ids'][j]==old['image_ids'][j]==sid
  pixels=rgb['targets_u8'][j].contiguous();tokens=old['full_tokens'][j].numpy().astype(np.int64);target=pixels.float()/255
  assert int(old['source_indices'][j])==row['source_index'];del rgb,old
  io_seconds+=time.perf_counter()-t;key=f"{row['split']}_i{row['source_index']:05d}";sd=out/key;sd.mkdir();t=time.perf_counter()
  streams=ledger.call('tx_probability_pass',key,lambda:codec.encode_endpoints(tokens,endpoints))
  wire={f'm{m}_k{k}':streams[m,k]['bits'] for m,k in endpoints}
  with (sd/'wire.npz').open('xb') as f:np.savez(f,**wire)
  rows=[];zs=[];rxarrays={}
  for m,k in endpoints:
   boundary();ek=f'{key}_m{m}_k{k}';family='EC_VAR_PARTIAL' if k else 'EC_VAR_WHOLE';bits=wire[f'm{m}_k{k}']
   rx=ledger.call('independent_source_decode',ek,lambda:codec.decode(family,bits,m,k))
   with torch.no_grad():im=ledger.call('predict_overwrite_advance',ek,lambda:render(n,rx['received_tokens'],m,k))
   assert np.array_equal(rx['received_tokens'],tokens[:OFFSETS[m]+k]),'Post-output lossless token audit'
   z=capture.pop('latent');zs.append(z[0]);rxarrays[f'm{m}_k{k}']=rx['received_tokens'].astype(np.uint16)
   a=torch.from_numpy(im)[None].cuda();b=target[None].cuda()
   with torch.no_grad():
    lpv=ledger.call('lpips_distance',ek,lambda:float(lp(a*2-1,b*2-1).item()))
    msv=ledger.call('ms_ssim',ek,lambda:float(ms_ssim(a,b,data_range=1,size_average=True).item()))
   mse=float(np.mean((im.astype(np.float64)-target.numpy().astype(np.float64))**2))
   rows.append(dict(source_id=sid,source_index=row['source_index'],split=row['split'],m=m,K=k,token_count=OFFSETS[m]+k,terminated_bits=int(len(bits)),finish_bits=int(streams[m,k]['flush_bits']),bits_per_pixel=len(bits)/65536,mse=mse,psnr_db=float(-10*np.log10(mse)),lpips_alex=lpv,ms_ssim=msv,source_pixels_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),source_tokens_uint16_sha256=hashlib.sha256(tokens.astype('<u2').tobytes()).hexdigest(),image_array_sha256=hashlib.sha256(im.tobytes()).hexdigest(),status='LOSSLESS_SOURCE_RX_COMPLETE'))
   if shard==0 and slot==0:
    Image.fromarray(np.rint(np.clip(im,0,1).transpose(1,2,0)*255).astype(np.uint8)).save(sd/f'preview_m{m}_k{k}.png')
  if shard==0 and slot==0:Image.fromarray(pixels.numpy().transpose(1,2,0)).save(sd/'preview_original.png')
  with (sd/'received_tokens.npz').open('xb')as f:np.savez(f,**rxarrays)
  latent=write_torch(sd/'latents.pt',dict(source_id=sid,source_index=row['source_index'],split=row['split'],endpoints=endpoints,latents=torch.stack(zs),registration_sha256=regsha),torch)
  save(sd/'completion.json',dict(status='SOURCE_SHARD_COMMITTED',rows=rows,latents=latent,wire=pin(sd/'wire.npz'),received=pin(sd/'received_tokens.npz'),rgb_shard=row['rgb_shard'],source_seconds=time.perf_counter()-t))
  allrows.extend(rows);source_seconds.append(time.perf_counter()-t)
  print(json.dumps(dict(shard=shard,completed_sources=slot+1,target_sources=len(selected),rows=len(allrows),elapsed_seconds=time.perf_counter()-started,source_seconds_mean=sum(source_seconds)/len(source_seconds))),flush=True)
 for d in reg['sources']:checked(d)
 with (out/'source_rd.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 save(out/'completion.json',dict(status='SEED_CACHE_SHARD_COMPLETE',shard=shard,sources=len(selected),endpoint_rows=len(allrows),source_seconds=sum(source_seconds),io_seconds=io_seconds,elapsed_seconds=time.perf_counter()-started,native_counts=counter.counts,ledger=pin(out/'ledger.jsonl'),rows=pin(out/'source_rd.csv'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),encoder_calls=0,PHY_calls=0,training_updates=0,registration_sha256=regsha))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
 try:main(a.registration_sha256,a.shard)
 except BaseException as e:
  out=L/'validation_cache_v1'/f'shard_{a.shard:02d}';out.mkdir(parents=True,exist_ok=True)
  if not(out/'failure.json').exists():save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),process=process(os.getpid())))
  raise
