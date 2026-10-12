"""Finite Swin200k/HiFi update. No training, selection, scoring, or bootstrap.

The receiver is passed only the observed frame plus public N/SNR/seed. One
qualification on calibration source0 precedes exactly16 full posterior calls.
The eight previous final-example calls are verified and reused, never repeated.
"""
from pathlib import Path
import argparse,dataclasses,fcntl,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/baseline-strength-20261011'))
import baseline_runtime as b
import run_continue as training
import numpy as np
import torch
from PIL import Image
OUT=ROOT/'results/hifi_final_fixed16_20261012'
ASSETS=b.TMP/'hifi_final_fixed16_v1'
VENDOR=b.TMP/'hifi_final_fixed16_assets_v1/diffcom_code'
ADM=b.MODELTEAM/'liberai-checkpoint/liulu/var_comm/hifi_final_fixed16_20261012/256x256_diffusion_uncond.pt'
INDICES=[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
PRIOR=ROOT/'results/figure_redraw_20261012/final_examples_v1'

def ash(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def pin(p):return dict(path=str(p),sha256=b.sha(p))
def checked(v):
 assert b.sha(v['path'])==v['sha256'],v['path'];return Path(v['path'])

def prepare():
 assert not (OUT/'registration.json').exists()
 sources=b.read(b.RESULT/'timing/fixed16_source_bindings_20261012.json')['records'];assert [r['source_index'] for r in sources]==INDICES
 for r in sources:
  checked(r['archive']);checked(r['checkpoint'])
 assert b.read(PRIOR/'actual_wait.json')['returncode']==0
 prior=b.read(PRIOR/'receipts.json');assert len(prior)==8
 for v in prior:checked(v['archive']);checked(v['png'])
 cp=b.read(b.RESULT/'automatic_followthrough_v1/registration.json')['checkpoints']
 for n in ['P100000','Swin200000']:checked(cp[n])
 assert b.sha(ADM)=='a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1'
 files=[Path(__file__).resolve(),Path(b.__file__),Path(training.__file__),ROOT/'experiments/external-comparison-20261004/hifi_qualification.py',ROOT/'experiments/external-comparison-20261004/hifi_swin_operator.py',ROOT/'experiments/external-comparison-20261004/hifi_swin_sampler.py',ROOT/'experiments/external-comparison-20261004/hifi_swin_schedule.py']
 files+=list((b.TMP/'vendor/SwinJSCC').rglob('*.py'))+[p for p in VENDOR.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')]
 binds={str(p):b.sha(p) for p in files if '__pycache__' not in p.parts}
 for p in (ROOT/'experiments/external-comparison-20261004').glob('swin_*.py'):binds[str(p)]=b.sha(p)
 b.write(OUT/'registration.json',dict(version='HIFI_FINAL_FIXED16_V1',sources=sources,source_indices=INDICES,N=1024,snr_db=13,channel_seed=2001,posterior_seed=23,schedule_mode='actual_data_cbr',step_limit=None,checkpoints={k:cp[k] for k in ['P100000','Swin200000']},adm=pin(ADM),vendor=str(VENDOR),code_bindings=binds,prior_receipts=pin(PRIOR/'receipts.json'),prior=prior,budget=dict(new_P=12,new_Swin=12,new_HiFi=16,reused_P=4,reused_Swin=4,qualification_calibration_sources=1,qualification_HiFi_probe_steps=0),training_updates=0,metric_calls=0,bootstrap_calls=0,source_role='original_fixed16_development_display_set',receiver_inputs=['actual_observed_frame','public_N','public_SNR','predetermined_posterior_seed'],quality_search=False,timing_comparison=False,allocator_limit_GiB=16))
 print('REGISTERED_16_FIXED_HIFI_AND_24_MISSING_BASELINE_FRAMES',flush=True)

def save_image(name,row,image,wave=None,observed=None,**extra):
 assert image.shape==(3,256,256) and image.dtype==np.float32 and np.isfinite(image).all()
 assert image.min()>=0 and image.max()<=1
 i=row['source_index'];display=np.rint(image.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
 a=ASSETS/f'{name}_{i:04d}.npz';assert not a.exists()
 arrays=dict(reconstruction=image,display=display)
 if wave is not None:arrays.update(wave=wave,observed=observed)
 np.savez_compressed(a,**arrays);p=a.with_suffix('.png');Image.fromarray(display).save(p)
 return dict(method=name,source_index=i,source_id=row['source_id'],source_pixels_sha256=row['preprocessing_id'],N=1024,snr_db=13,noise_seed=2001,reconstruction_sha256=ash(image),display_sha256=ash(display),archive=pin(a),png=pin(p),**extra)

def run():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'owner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'launch.json').exists(),'One attempt only; inspect failure before any recovery'
 reg=b.read(OUT/'registration.json')
 for p,h in reg['code_bindings'].items():assert b.sha(p)==h,p
 checked(reg['adm']);checked(reg['prior_receipts'])
 for cp in reg['checkpoints'].values():checked(cp)
 training.resource_check(startup=True);ASSETS.mkdir(exist_ok=False)
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
 b.write(OUT/'launch.json',dict(pid=os.getpid(),gpu=os.environ['BASELINE_GPU_ID'],time=time.time(),registration_sha256=b.sha(OUT/'registration.json')))
 numerical=b.configure();receipts=[];pending=None;start=time.time()
 def reserve(case):
  nonlocal pending
  assert not (OUT/'STOP').exists();training.resource_check();pending=case;b.write(OUT/'pending_case.json',case)
 def finish(v):
  nonlocal pending
  receipts.append(v);b.write(OUT/'receipts.json',receipts);(OUT/'pending_case.json').unlink(missing_ok=True);pending=None
  b.write(OUT/'status.json',dict(status='RUNNING',rows=len(receipts),hifi_done=sum(r['method']=='HiFiSwin200000' for r in receipts),elapsed_seconds=time.time()-start,last_method=v['method'],last_source=v['source_index']))
  print('FRAME_COMPLETE',v['method'],v['source_index'],flush=True)
 try:
  sources=reg['sources'];pixels={}
  for r in sources:
   with np.load(checked(r['archive']),allow_pickle=False) as z:x=z['pixels'].copy()
   assert x.shape==(3,256,256) and x.dtype==np.uint8 and hashlib.sha256(x.tobytes()).hexdigest()==r['preprocessing_id'];pixels[r['source_index']]=x
  for branch,name in [('P','P100000'),('Swin','Swin200000')]:
   reserve(dict(stage='model_load',branch=branch));s=training.load(branch);cp=reg['checkpoints'][name]
   payload=torch.load(cp['path'],map_location='cpu',weights_only=True);s['model'].load_state_dict(payload['model'],strict=True);del payload
   model=s['model'].eval().requires_grad_(False);(OUT/'pending_case.json').unlink();pending=None
   if branch=='P':
    del s['lp'];torch.cuda.empty_cache()
    from models.vqvae import VQVAE
    vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,test_mode=True,share_quant_resi=4)
    weight=b.RT/'assets/model_seed_v1/external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth'
    assert b.sha(weight)=='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c'
    vae.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);vae.cuda().eval().requires_grad_(False)
   for r in sources:
    i=r['source_index'];old=[v for v in reg['prior'] if v['method']==name and v['source_index']==i]
    if old:
     v=dict(old[0],reuse=True);checked(v['archive']);checked(v['png']);assert v['checkpoint_sha256']==cp['sha256'];finish(v);continue
    reserve(dict(stage='baseline',method=name,source_index=i))
    with torch.no_grad():
     x=torch.as_tensor(pixels[i][None],device='cuda',dtype=torch.float32)
     if branch=='P':
      F=vae.quant_conv(vae.encoder(x/127.5-1));wave=model.transmit(F)[0].cpu().numpy()
      observed=wave+s['ns']['seeded_noise']('VAR-CONTINUOUS-1024|'+r['source_id'],2001,(1024,2)).astype(np.float32)*np.float32(10**(-13/20))
      received=model.receive(torch.as_tensor(observed[None],device='cuda'),torch.tensor([13],device='cuda',dtype=torch.float32));image=s['dc'](received)[0].cpu().numpy();link={}
     else:
      wave=s['sr'].transmit(model,x/255,1024,13);observed=wave+s['sp'].standard_noise(r['source_id'],2001,1024,13)/10**(13/20)
      image,context=s['sr'].receive(model,observed,1024,13);link=dict(rx_context=dataclasses.asdict(context),header_accepted=context.accepted)
    finish(save_image(name,r,image,wave,observed,checkpoint_sha256=cp['sha256'],wave_sha256=ash(wave),observation_sha256=ash(observed),reuse=False,**link))
   if branch=='P':
    del vae,F,received,model,s;torch.cuda.empty_cache();continue
   # All HiFi forward/confirming modules are the very same loaded Swin200k model.
   from hifi_swin_sampler import FrozenHiFiSwinReceiver
   from hifi_swin_operator import SwinHiFiOperator,strict_decoded_context,qualify_native_parity
   from hifi_qualification import finite_difference
   reserve(dict(stage='qualification',source_role='calibration',index=0))
   pop=b.Population('calibration',limit=1);cal=pop.batch([0]);w=s['sr'].transmit(model,cal,1024,13);y=w+s['sp'].standard_noise(pop.ids[0],2001,1024,13)/10**(13/20)
   base,context=s['sr'].receive(model,y,1024,13);assert context.accepted
   parity=qualify_native_parity(model.native,cal,13,6)
   op=SwinHiFiOperator(model.native,strict_decoded_context(1024,13,dict(power=context.power,indices=context.indices,rate=context.channels),crc_accepted=True));measurement=op.measurement(y[:768])
   np.testing.assert_allclose(measurement['x_mse'][0].cpu().numpy(),base,rtol=2e-5,atol=3e-6)
   derivatives=[finite_difference(op,cal,measurement,n) for n in ['waveform','confirming_rgb']]
   b.write(OUT/'qualification.json',dict(status='PASS_FINAL_SWIN_OPERATOR',checkpoint_sha256=cp['sha256'],calibration_source=pop.ids[0],source_bindings=pop.bindings,native_parity=parity,derivatives=derivatives,confirming_decoder_same_as_Swin=True,training_updates=0,ADM_probe_steps=0))
   del pop,cal,w,y,base,op,measurement;(OUT/'pending_case.json').unlink();pending=None
   reserve(dict(stage='ADM_load'));receiver=FrozenHiFiSwinReceiver(model.native,VENDOR,ADM,schedule_mode='actual_data_cbr');(OUT/'pending_case.json').unlink();pending=None
   for r in sources:
    i=r['source_index'];baseline=next(v for v in receipts if v['method']=='Swin200000' and v['source_index']==i)
    with np.load(checked(baseline['archive']),allow_pickle=False) as z:observed=z['observed'].copy();reference=z['reconstruction'].copy()
    assert ash(observed)==baseline['observation_sha256']
    # Source pixels, TX waveform, TX mask and TX gain never enter the HiFi API.
    reserve(dict(stage='full_HiFi_posterior',source_index=i,observed_sha256=ash(observed)))
    image,info=receiver.receive_frame(observed,1024,13,sampling_seed=23,step_limit=None)
    assert info['header_accepted']==baseline.get('header_accepted',True)
    if info['header_accepted']:assert info['complete_author_schedule'] and not info['diagnostic_probe']
    finish(save_image('HiFiSwin200000',r,image,checkpoint_sha256=cp['sha256'],paired_Swin_archive=baseline['archive'],observation_sha256=ash(observed),sampler_receipt=info,reuse=False,timing_qualified=False))
   del receiver,model,s;torch.cuda.empty_cache()
  assert len(receipts)==48 and len({(v['method'],v['source_index']) for v in receipts})==48
  for p,h in reg['code_bindings'].items():assert b.sha(p)==h,p
  b.write(OUT/'completion.json',dict(status='COMPLETE_FIXED16_FINAL_HIFI',rows=48,HiFi_recoveries=16,new_baseline_frames=24,reused_baseline_frames=8,numerical=numerical,elapsed_seconds=time.time()-start,receipts_sha256=b.sha(OUT/'receipts.json'),training_updates=0,metric_calls=0,bootstrap_calls=0,timing_qualified=False))
  b.write(OUT/'status.json',dict(status='COMPLETE',rows=48,hifi_done=16))
 except BaseException:
  b.write(OUT/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',pending=pending,rows=len(receipts),traceback=traceback.format_exc()));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args();prepare() if args.mode=='prepare' else run()
