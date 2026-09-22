from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch,yaml
from latent_enhancement.runtime import digest,model_paths,perceptual_model,settings
from latent_enhancement.latent import complete_latent,enhancement_noise,original_rgb,ContinuousDecoder
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.model import build_arms,render_received
from latent_enhancement_eval.runner import arithmetic_receive_budget,arithmetic_transmit_budget,load_targets,raw_receive_budget,raw_transmit_budget
from var_comm.next_scale_prior import load_models
from var_comm.progressive import split_prefix
from var_comm.quality import load_quality_models,quality_metrics
from var_comm.study import paired_interval,seeded_noise
from var_comm.whole_entropy import encode_prefixes

PROJECT=Path(__file__).resolve().parents[5];VAR_COMM=PROJECT;ROOT=VAR_COMM/'outputs/VAR-LATENT-ENHANCEMENT-20260917';EXP=VAR_COMM/'experiments/var-latent-enhancement-20260917'

def _selected_checkpoint(name):
 """Resolve a selected arm checkpoint and verify its recorded identity."""
 base=ROOT/'followup/decoder_adaptation_v1'; candidates=[base/f'selected_{name}.json',base/f'{name}.json',base/'selected.json']
 record=None
 for path in candidates:
  if path.exists():
   value=json.loads(path.read_text());record=value.get(name,value) if isinstance(value,dict) else None
   if isinstance(record,dict) and ('checkpoint' in record or 'step' in record): break
  record=None
 completion=base/'completion.json'
 if record is None and completion.exists():
  value=json.loads(completion.read_text());record=value.get('selected_checkpoints',{}).get(name) or value.get('selection',{}).get(name)
 if not isinstance(record,dict): raise FileNotFoundError(f'no selected record for adaptation arm {name}')
 checkpoint=record.get('checkpoint')
 if checkpoint is None:
  step=record.get('step')
  if step is None: raise KeyError(f'selected record for {name} has no checkpoint or step')
  checkpoint=str(base/'checkpoints'/f'step_{int(step):05d}.pt')
 checkpoint=Path(checkpoint)
 if not checkpoint.is_absolute(): checkpoint=(VAR_COMM/checkpoint).resolve()
 if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
 actual=digest(checkpoint);expected=record.get('checkpoint_sha256') or record.get('sha256')
 if expected is None: raise RuntimeError(f'selected record for {name} lacks checkpoint SHA')
 if actual!=expected: raise RuntimeError(f'checkpoint SHA mismatch for {name}: {actual} != {expected}')
 record=dict(record);record.update({'checkpoint':str(checkpoint),'checkpoint_sha256':actual})
 return record
def write_rows(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'followup/adaptation_development_v1'));args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);decoder_old=load_decoder(vae,device);scale=scale_statistics(device);recipe=settings()['stage_B'];
 selected_control=_selected_checkpoint('communication_continuation_Dc_frozen');selected_adapt=_selected_checkpoint('communication_plus_Dc_adaptation');ck_control=torch.load(selected_control['checkpoint'],map_location='cpu',weights_only=True);ck_adapt=torch.load(selected_adapt['checkpoint'],map_location='cpu',weights_only=True);arms=build_arms((32,16,16),scale,recipe).to(device);control=arms['enhancement1024'];control.load_state_dict(ck_control['control'],strict=True);adapt_arm=build_arms((32,16,16),scale,recipe).to(device)['enhancement1024'];adapt_arm.load_state_dict(ck_adapt['adapt'],strict=True);dec_adapt=ContinuousDecoder(vae).to(device);dec_adapt.load_state_dict(ck_adapt['decoder_adapt'],strict=True);control.eval().requires_grad_(False);adapt_arm.eval().requires_grad_(False);dec_adapt.eval().requires_grad_(False)
 quality=yaml.safe_load((VAR_COMM/'configs/progressive_channel.yaml').read_text())['quality'];perceptual,dino,_=load_quality_models(quality,device);targets=load_targets();methods=['m8_1024_control','m8_1024_decoder_adapt','raw_N4084_m8_Dc_adapted','raw_N4084_m9_Dc_adapted'];aliases={'m8_Dc_adapted':'raw_N4084_m8_Dc_adapted'};rows=[]
 for done,record in enumerate(targets):
  image=torch.from_numpy(record['pixels'][None].astype(np.float32)/127.5-1).to(device);label=int(record['target']['class_index']);source=split_prefix(record['tokens'],10)
  with torch.no_grad(): F=vae.quant_conv(vae.encoder(image));Fb=complete_latent(vae,var,source[:8],label,device);wave_control=control.encoder(F-Fb,Fb)[0].cpu().numpy();wave_adapt=adapt_arm.encoder(F-Fb,Fb)[0].cpu().numpy();raw4084m8=raw_transmit_budget(source,label,8,4084)[0];raw4084m9=raw_transmit_budget(source,label,9,4084)[0]
  imgs={m:[] for m in methods};meta={m:[] for m in methods}
  for snr in [1,4,7,13,19]:
   for seed in [2001,2002,2003]:
    base,ledger=raw_transmit_budget(source,label,8,3060);base_rx=base+seeded_noise(record['target']['image_id'],seed,base.shape)/np.sqrt(10**(snr/10));phy=raw_receive_budget(base_rx,snr,3060)
    if phy['label'] is None:base_latent=torch.zeros_like(Fb);status=torch.zeros((1,3),device=device)
    else:base_latent=complete_latent(vae,var,phy['prefix'],phy['label'],device);status=torch.tensor([[1.,float(phy['body_crc_accepted']),float(phy['mode'])]],device=device)
    for name,arm,decoder,wave in [('m8_1024_control',control,decoder_old,wave_control),('m8_1024_decoder_adapt',adapt_arm,dec_adapt,wave_adapt)]:
     with torch.no_grad():obs=torch.as_tensor(wave[None],device=device,dtype=torch.float32)+torch.as_tensor(enhancement_noise(record['target']['image_id'],seed,1024)[None],device=device,dtype=torch.float32)/np.sqrt(10**(snr/10));latent=arm.receiver(obs,base_latent,status.new_tensor([snr]),status);im=render_received(decoder,latent,status)[0].cpu().numpy()
     selected_sha=selected_control['checkpoint_sha256'] if name=='m8_1024_control' else selected_adapt['checkpoint_sha256']
     imgs[name].append(im);meta[name].append({'method':name,'checkpoint_sha256':selected_sha,'source_index':record['index'],'image_id':record['target']['image_id'],'snr_db':snr,'seed':seed,'N':4084,'E':8168,'header_ok':int(status[0,0]),'body_crc_ok':int(status[0,1])})
    # New Dc digital renders under exact 4084 received waveforms.
    for name,signal,family,mode in [('raw_N4084_m8_Dc_adapted',raw4084m8,'raw',8),('raw_N4084_m9_Dc_adapted',raw4084m9,'raw',9)]:
     received=signal+seeded_noise(record['target']['image_id'],seed,signal.shape)/np.sqrt(10**(snr/10));p=raw_receive_budget(received,snr,4084)
     if p['label'] is None:im=np.full((3,256,256),.5,np.float32)
     else:im=dec_adapt(complete_latent(vae,var,p['prefix'],p['label'],device))[0].cpu().numpy()
     imgs[name].append(im);meta[name].append({'method':name,'source_index':record['index'],'image_id':record['target']['image_id'],'snr_db':snr,'seed':seed,'N':4084,'E':8168,'header_ok':int(p['header']['accepted']),'body_crc_ok':int(p['body_crc_accepted'])})
  allim=[]
  for m in methods:allim.extend(imgs[m])
  q,_,_=quality_metrics(record['pixels'].astype(np.float32)/255.,allim,perceptual,dino,device);offset=0
  for m in methods:
   for j,row in enumerate(meta[m]):row.update(q[offset+j]);rows.append(row)
   offset+=len(meta[m])
  if done%10==0:print(f'adaptation development {done+1}/100',flush=True)
 write_rows(out/'per_frame.csv',rows); summaries=[]
 for m in methods:
  v=[r for r in rows if r['method']==m];summaries.append({'method':m,'PSNR':float(np.mean([r['psnr_db'] for r in v])),'LPIPS':float(np.mean([r['lpips_alex'] for r in v])),'DINO':float(np.mean([r['dino_cosine'] for r in v]))})
 write_rows(out/'summary.csv',summaries)
 base={m:np.array([[float(r['psnr_db']),float(r['lpips_alex']),float(r['dino_cosine'])] for r in rows if r['method']==m]).reshape(100,15,3).mean(1) for m in methods};pairs=[]
 for m in methods[1:]:
  ref=base['m8_1024_control'];cur=base[m];pairs.append({'method':m,'reference':'m8_1024_control','delta_psnr':paired_interval(cur[:,0]-ref[:,0],2026091901,10000),'delta_lpips':paired_interval(cur[:,1]-ref[:,1],2026091902,10000),'delta_dino':paired_interval(cur[:,2]-ref[:,2],2026091903,10000)})
 (out/'paired.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'status':'ADAPTATION_DEVELOPMENT_COMPLETE','rows':len(rows),'sources':100,'new_holdout_used':False,'selected_checkpoints':{'communication_continuation_Dc_frozen':selected_control,'communication_plus_Dc_adaptation':selected_adapt},'aliases':aliases,'canonical_methods':methods},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
