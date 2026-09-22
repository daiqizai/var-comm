from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch,yaml
from latent_enhancement.latent import complete_latent,enhancement_noise
from latent_enhancement.runtime import digest,model_paths,settings
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.model import build_arms,render_received
from latent_enhancement_eval.runner import load_targets,raw_receive_budget,raw_transmit_budget
from var_comm.next_scale_prior import load_models
from var_comm.progressive import split_prefix
from var_comm.quality import load_quality_models,quality_metrics
from var_comm.study import seeded_noise

PROJECT=Path(__file__).resolve().parents[5]; VAR_COMM=PROJECT; EXP=VAR_COMM/'experiments/var-latent-enhancement-20260917'; OUT_ROOT=VAR_COMM/'outputs/VAR-LATENT-ENHANCEMENT-20260917';

def write_rows(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r));
    with Path(path).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
  p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT_ROOT/'followup/allocation_v1'));args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
  config=json.loads((EXP/'followup/config.json').read_text());device=torch.device('cuda:0');
  vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device);recipe=settings()['stage_B'];
  arms=build_arms((32,16,16),scale,recipe).to(device);selected=json.loads((OUT_ROOT/'stage_B_v1/training/selected_enhancement512.json').read_text());checkpoint=Path(selected['checkpoint']);
  if digest(checkpoint)!=selected['checkpoint_sha256']:raise RuntimeError('512 checkpoint changed')
  arms.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['arms'],strict=True);arm=arms['enhancement512'].eval().requires_grad_(False)
  quality=yaml.safe_load((VAR_COMM/'configs/progressive_channel.yaml').read_text())['quality'];perceptual,dino,_=load_quality_models(quality,device);targets=load_targets();all_rows=[]
  for done,record in enumerate(targets):
    image=torch.from_numpy(record['pixels'][None].astype(np.float32)/127.5-1).to(device);label=int(record['target']['class_index']);source=split_prefix(record['tokens'],10)
    with torch.no_grad():F=vae.quant_conv(vae.encoder(image));Fb=complete_latent(vae,var,source[:8],label,device);wave=arm.encoder(F-Fb,Fb)[0].cpu().numpy()
    images=[];meta=[]
    for snr in config['snrs_db']:
      for seed in config['noise_seeds']['development']:
        base,_=raw_transmit_budget(source,label,8,3572);received=base+seeded_noise(record['target']['image_id'],seed,base.shape)/np.sqrt(10**(snr/10));phy=raw_receive_budget(received,snr,3572)
        if phy['label'] is None:base_latent=torch.zeros_like(Fb);status=torch.zeros((1,3),device=device)
        else:
          base_latent=complete_latent(vae,var,phy['prefix'],phy['label'],device);status=torch.tensor([[1.,float(phy['body_crc_accepted']),float(phy['mode'])]],device=device)
        with torch.no_grad():
          added=wave+enhancement_noise(record['target']['image_id'],seed,512)/np.sqrt(10**(snr/10));corrected=arm.receiver(torch.as_tensor(added[None],device=device,dtype=torch.float32),base_latent,torch.tensor([snr],device=device),status);image_hat=render_received(decoder,corrected,status)[0].cpu().numpy()
        images.append(image_hat);meta.append({'method':'m8_plus_latent_512_fold_N4084','source_index':record['index'],'image_id':record['target']['image_id'],'snr_db':snr,'seed':seed,'N':4084,'E':8168,'header_ok':int(phy['header']['accepted']),'body_crc_ok':int(phy['body_crc_accepted']),'received_mode':phy['mode'] if phy['mode'] is not None else '','accepted_correct':int(phy['header']['accepted'] and phy['body_crc_accepted'] and phy['label']==label and phy['mode']==8 and len(phy['prefix'])==8 and all(np.array_equal(a,b) for a,b in zip(phy['prefix'],source[:8])))})
    metric,_,_=quality_metrics(record['pixels'].astype(np.float32)/255.,images,perceptual,dino,device)
    for row,m in zip(meta,metric):row.update(m)
    image_dir=out/'images'/f"{record['index']:03d}";image_dir.mkdir(parents=True,exist_ok=True)
    write_rows(image_dir/'per_frame.csv',meta);all_rows.extend(meta)
    if done%10==0:print(f'allocation fold {done+1}/100',flush=True)
  write_rows(out/'per_frame.csv',all_rows);json.dump({'status':'ALLOCATION_FOLD_COMPLETE','sources':100,'rows':len(all_rows),'N':4084,'digital_data_uses':3504,'continuous_uses':512,'new_holdout_used':False},(out/'completion.json').open('w'),ensure_ascii=False,indent=2);print('complete',len(all_rows))
if __name__=='__main__':main()
