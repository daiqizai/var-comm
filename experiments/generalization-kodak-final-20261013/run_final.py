"""Frozen Kodak24 final-checkpoint supplement: 432 fresh frames, no training.

The only receiver inputs are its actual observation and public N/SNR. Source
pixels and TX auxiliary values are retained solely for post-receive auditing.
No old baseline observations or reconstructions enter this implementation.
"""
from pathlib import Path
import argparse,dataclasses,fcntl,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/baseline-strength-20261011'))
import baseline_runtime as b
import run_continue as loading
import numpy as np
import torch
from PIL import Image
OUT=ROOT/'results/generalization_kodak_final_20261013'
OLD=ROOT/'results/generalization_kodak_20261009'
ASSETS=b.TMP/'kodak_final_20261013_v1'
SNRS=[4,10,19];SEEDS=[2001,2002,2003]
METRIC_ID='8313f062d2435be361add61aceb9848ccb8844c1f3c9c4743f662480057489e2'

def ash(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=b.sha(p))
def checked(v):
 assert b.sha(v['path'])==v['sha256'],v['path'];return Path(v['path'])
def verify_old(snapshot):
 for p,h in snapshot.items():assert b.sha(p)==h,p

def prepare():
 assert not (OUT/'registration.json').exists(),'Do not replace a registered attempt'
 protocol=b.read(OLD/'protocol.json');manifest=b.read(OLD/'evidence/dataset_manifest.json')
 assert b.sha(OLD/'evidence/dataset_manifest.json')==protocol['dataset_manifest_sha256']
 assert protocol['N_complex_channel_uses']==1024 and protocol['snr_db']==SNRS and protocol['noise_seeds']==SEEDS
 assert protocol['statistics']['bootstrap_seed']==2026100901 and protocol['statistics']['bootstrap_draws']==10000
 sources=[]
 for record in manifest['records']:
  r=dict(record);p=ASSETS/'crops'/Path(record['png']['path']).name
  assert b.sha(p)==record['png']['sha256'];r['png']=pin(p)
  with Image.open(p) as image:
   assert image.mode=='RGB' and image.size==(256,256)
   pixels=np.ascontiguousarray(np.asarray(image,dtype=np.uint8).transpose(2,0,1))
  assert hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixels_chw_uint8_sha256']
  sources.append(r)
 assert len(sources)==24 and [r['source_index'] for r in sources]==list(range(24))
 snapshot={str(p):b.sha(p) for p in OLD.rglob('*') if p.is_file()}
 prior=b.read(b.RESULT/'automatic_followthrough_v1/registration.json')
 cp={k:prior['checkpoints'][k] for k in ['P100000','Swin200000']}
 for v in cp.values():checked(v)
 assert cp['P100000']['sha256']=='059888fec9fb1d3548e2b3c34d681030d3df2882163c26d5b59c69c96dcdc4d6'
 assert cp['Swin200000']['sha256']=='572d7acaceed2712d885ebd351f65ef6042f7716082d81f63bf45fbf8aadbd45'
 old_score=b.read(OLD/'metrics_v1/completion.json')
 assert old_score['original_metric_backend_identity']==METRIC_ID
 files=[Path(__file__).resolve(),Path(b.__file__),Path(loading.__file__),ROOT/'experiments/generalization_kodak_20261009/neural_adapter.py',ROOT/'experiments/generalization_kodak_20261009/score.py',ROOT/'experiments/entropy-partial-mismatch-20261010/scripts/h800_ep_pilot_metrics_v1.py']
 files+=list((ROOT/'experiments/external-comparison-20261004').glob('swin_*.py'))
 files+=list((b.TMP/'vendor/SwinJSCC').rglob('*.py'))
 files+=list((b.RT/'assets/var_upstream_candidate_v1').rglob('*.py'))
 files+=list((b.RT/'assets/var_native_source_seed_v1/VAR').rglob('*.py'))
 files+=list(Path(__file__).parent.glob('*.py'))
 model_paths=[b.CP/'parents/p/registration.json',next((b.CP/'parents/p').glob('*.pt')),b.CP/'parents/swin/model_080000.pt',b.RT/'assets/model_seed_v1/external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth',b.RT/'assets/model_seed_v1/VAR_COMM/outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_A_v1/checkpoints/update_0038000_1789635001035591298.pt']
 # Identity-only bindings; no source model or optimizer is updated.
 files+=model_paths
 _,pins=b.p_math()
 binds={str(p):b.sha(p) for p in files if '__pycache__' not in p.parts};binds.update(pins)
 b.write(OUT/'registration.json',dict(schema='KODAK24_FINAL_BASELINES_V1',protocol=pin(OLD/'protocol.json'),source_manifest=pin(OLD/'evidence/dataset_manifest.json'),sources=sources,source_count=24,N=1024,snr_db=SNRS,noise_seeds=SEEDS,methods=['P100K','SWIN200K'],frames=432,frames_per_method=216,batch_size=1,checkpoints=cp,code_bindings=binds,old_snapshot=snapshot,old_score_completion=pin(OLD/'metrics_v1/completion.json'),old_score_rows=pin(OLD/'metrics_v1/rows.csv'),original_rows=pin(OLD/'metrics_v1/rows.csv'),original_metric_identity=METRIC_ID,metric_config=prior['metric_config'],deadline_unix=time.time()+4*3600,stop_files=[str(OUT/'STOP'),str(ROOT/'STOP')],training_updates=0,calibration_calls=0,policy_selection=False,noise_rule={'P100K':'original seeded_noise(VAR-CONTINUOUS-1024|source_id,seed,(1024,2)) float32 times float32 10**(-snr/20)','SWIN200K':'original standard_noise(source_id,seed,1024,snr) float64 divided by 10**(snr/20)'},source_role='all24 previously fixed Kodak center256 crops',fresh_transmission_required=True,old_receiver_results_used=False,header_decode_cap=216,model_load_cap=2,visual_encode_cap=432,metric_score_cap=432,bootstrap_seed=2026100901,bootstrap_draws=10000,timing_qualified=False))
 print('REGISTERED_432_FRESH_FRAMES',flush=True)

def run():
 lock=(OUT/'inference.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'launch_inference.json').exists(),'Preserve existing attempt; no duplicate or retry'
 reg=b.read(OUT/'registration.json');verify_old(reg['old_snapshot'])
 for p,h in reg['code_bindings'].items():assert b.sha(p)==h,p
 for v in reg['checkpoints'].values():checked(v)
 loading.resource_check(startup=True)
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);os.nice(10)
 numerical=b.configure();rows=[];pending=None;started=time.time()
 runtime=ASSETS/'frames';runtime.mkdir(exist_ok=False)
 b.write(OUT/'launch_inference.json',dict(pid=os.getpid(),gpu=os.environ['BASELINE_GPU_ID'],registration_sha256=b.sha(OUT/'registration.json'),time=time.time()))
 def reserve(case):
  nonlocal pending
  assert time.time()<reg['deadline_unix'] and not any(Path(p).exists() for p in reg['stop_files'])
  loading.resource_check();pending=case;b.write(OUT/'pending_inference_case.json',case)
 def clear():
  nonlocal pending
  (OUT/'pending_inference_case.json').unlink(missing_ok=True);pending=None
 try:
  pixels={}
  for r in reg['sources']:
   with Image.open(checked(r['png'])) as image:x=np.ascontiguousarray(np.asarray(image,dtype=np.uint8).transpose(2,0,1))
   assert hashlib.sha256(x.tobytes()).hexdigest()==r['pixels_chw_uint8_sha256'];pixels[r['source_index']]=x
  for branch,name,key in [('P','P100K','P100000'),('Swin','SWIN200K','Swin200000')]:
   reserve(dict(stage='model_load',method=name));s=loading.load(branch);cp=reg['checkpoints'][key]
   payload=torch.load(cp['path'],map_location='cpu',weights_only=True);s['model'].load_state_dict(payload['model'],strict=True);del payload
   model=s['model'].eval().requires_grad_(False)
   if branch=='P':
    del s['lp'];del s['payload'];torch.cuda.empty_cache()
    from models.vqvae import VQVAE
    vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,test_mode=True,share_quant_resi=4)
    weight=b.RT/'assets/model_seed_v1/external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth'
    assert b.sha(weight)=='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c'
    vae.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);vae.cuda().eval().requires_grad_(False)
   clear()
   for r in reg['sources']:
    i=r['source_index'];sid=r['source_id']
    for snr in SNRS:
     for seed in SEEDS:
      case=dict(stage='fresh_frame',method=name,source_index=i,snr_db=snr,noise_seed=seed);reserve(case)
      with torch.inference_mode():
       x=torch.as_tensor(pixels[i][None],device='cuda',dtype=torch.float32)
       if branch=='P':
        F=vae.quant_conv(vae.encoder(x/127.5-1));wave=model.transmit(F)[0].cpu().numpy()
        noise=s['ns']['seeded_noise']('VAR-CONTINUOUS-1024|'+sid,seed,(1024,2)).astype(np.float32)
        observed=wave+noise*np.float32(10**(-snr/20))
        received=model.receive(torch.as_tensor(observed[None],device='cuda'),torch.tensor([snr],device='cuda',dtype=torch.float32));rgb=s['dc'](received)[0].cpu().numpy()
        status='RECEIVED_OK';context=None;packet_count=0
       else:
        data,power,indices=model.encode_data(x/255,snr,s['sp'].layout(1024)['channels'])
        tx_context=dict(power=float(np.float32(float(power[0]))),indices=indices[0].cpu().tolist())
        wave=s['sp'].transmit_frame(data[0].cpu().numpy(),float(power[0]),tuple(tx_context['indices']),1024)
        noise=s['sp'].standard_noise(sid,seed,1024,snr)
        observed=wave+noise/10**(snr/20)
        rgb,rx=s['sr'].receive(model,observed,1024,snr);context=dataclasses.asdict(rx)
        context_matches=bool(rx.accepted and rx.power==tx_context['power'] and list(rx.indices)==tx_context['indices'])
        status='HEADER_REJECT_GRAY' if not rx.accepted else ('RECEIVED_OK' if context_matches else 'HEADER_WRONG_ACCEPT');packet_count=1
        context=dict(actual_context=context,transmitter_audit_after_reconstruction=tx_context,received_context_matches_transmitted=context_matches)
      assert wave.shape==observed.shape==(1024,2) and rgb.shape==(3,256,256) and rgb.dtype==np.float32
      assert np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
      path=runtime/f'{name}_s{snr:02d}_i{i:02d}_n{seed}.npz';assert not path.exists()
      np.savez_compressed(path,rgb=rgb,wave=wave,observed=observed,standard_noise=noise)
      row=dict(method=name,source_index=i,source_id=sid,snr_db=snr,noise_seed=seed,N=1024,population='Kodak24',preprocessing_id=r['preprocessing_id'],pixels_chw_uint8_sha256=r['pixels_chw_uint8_sha256'],status=status,gray=status=='HEADER_REJECT_GRAY',checkpoint_sha256=cp['sha256'],reconstruction=dict(**pin(path),key='rgb'),wave_sha256=ash(wave),observation_sha256=ash(observed),standard_noise_sha256=ash(noise),reconstruction_sha256=ash(rgb),actual_energy=float(np.square(wave.astype(np.float64)).sum()),receiver_context=context,packet_decode_count=packet_count,fresh_TX=True,old_observation_used=False,source_or_hidden_TX_information_to_receiver=False)
      rows.append(row);b.write(OUT/'frames.json',rows);clear()
      if len(rows)%9==0:
       b.write(OUT/'status.json',dict(status='INFERENCE_RUNNING',frames=len(rows),method=name,source_index=i,elapsed_seconds=time.time()-started));print('FRAME_GRID_PROGRESS',len(rows),name,i,flush=True)
   if branch=='P':del vae,F,received
   del model,s;torch.cuda.empty_cache()
  assert len(rows)==432 and len({(v['method'],v['source_index'],v['snr_db'],v['noise_seed']) for v in rows})==432
  assert sum(r['packet_decode_count'] for r in rows)==216
  verify_old(reg['old_snapshot'])
  for p,h in reg['code_bindings'].items():assert b.sha(p)==h,p
  b.write(OUT/'inference_completion.json',dict(status='COMPLETE_432_FRESH_KODAK_FRAMES',frames=432,source_count=24,noise_count=3,methods=['P100K','SWIN200K'],N=1024,snr_db=SNRS,frames_sha256=b.sha(OUT/'frames.json'),fresh_transmitter_calls=432,new_channel_observations=432,receiver_calls=432,header_decode_calls=216,model_loads=2,failed_frames=sum(r['status']!='RECEIVED_OK' for r in rows),gray_frames=sum(r['gray'] for r in rows),numerical=numerical,old_results_unchanged=True,training_updates=0,selection_calls=0,elapsed_seconds=time.time()-started))
  b.write(OUT/'status.json',dict(status='INFERENCE_COMPLETE_SCORING_PENDING',frames=432))
 except BaseException:
  b.write(OUT/'inference_failure.json',dict(status='FAILED_PRESERVE_NO_AUTOMATIC_RETRY',pending=pending,frames=len(rows),traceback=traceback.format_exc()));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);a=parser.parse_args();prepare() if a.mode=='prepare' else run()
