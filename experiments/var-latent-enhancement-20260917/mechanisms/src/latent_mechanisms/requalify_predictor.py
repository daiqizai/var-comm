"""Evaluate existing selected predictor checkpoints; no optimizer or training."""
import argparse,copy,json,csv
from pathlib import Path
import numpy as np
import torch,yaml
from .predictor_innovation import ROOT,EXP,VAR,CONFIG,read,selected_record,load_shared_predictor,matched_batch,calibrate_v2
from latent_followup.run_identity import selection_record,checked_checkpoint,register_run
from latent_enhancement.runtime import configure,model_paths,settings,write_json,digest,perceptual_model,require_available
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.data import MatchedPopulation
from latent_enhancement_b.model import build_arms,render_received
from latent_enhancement.latent import complete_latent,enhancement_noise
from latent_enhancement_eval.runner import load_targets,raw_transmit_budget,raw_receive_budget
from var_comm.next_scale_prior import load_models
from var_comm.progressive import split_prefix
from var_comm.study import seeded_noise
from var_comm.quality import load_quality_models,quality_metrics

def write_rows(path,rows):
    with Path(path).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--from-run',required=True);p.add_argument('--output',required=True);p.add_argument('--reuse-calibration');args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    configure();require_available();device=torch.device('cuda:0');cfg=read(CONFIG);recipe=settings()['stage_B'];scale=scale_statistics(device)
    origin=Path(args.from_run);names=('original_residual_control','predicted_innovation_candidate');paths=[origin/f'selected_{n}.json' for n in names]
    registration=register_run(out,CONFIG,paths+[ROOT/'stage_B_v1/training/selected_receiver_only_refiner.json'])
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);perceptual=perceptual_model(device)
    predictor,predictor_id=load_shared_predictor(ROOT/'stage_B_v1/training/selected_receiver_only_refiner.json',(32,16,16),scale,recipe,device)
    arms={};selected={}
    for n,path in zip(names,paths):
        r,c,sha=selected_record(path);selected[n]=selection_record(n,r,c,r['parent_step']);payload=torch.load(c,map_location='cpu',weights_only=True)
        arm=build_arms((32,16,16),scale,recipe)['enhancement1024'].to(device);arm.load_state_dict(payload[selected[n]['arm_key']],strict=True);arms[n]=arm.eval().requires_grad_(False)
        write_json(out/f'selected_{n}.json',selected[n])
    summaries={}
    if args.reuse_calibration:
        old=Path(args.reuse_calibration);prior=read(old/'registration.json')['source_bindings'];current=read(out/'registration.json')['source_bindings']
        for path,sha in prior.items():
            if Path(path).resolve()!=Path(__file__).resolve() and current.get(path)!=sha:raise RuntimeError('calibration dependency changed')
        reuse={}
        for n in names:
            if read(old/f'selected_{n}.json')['checkpoint_sha256']!=selected[n]['checkpoint_sha256']:raise RuntimeError('calibration checkpoint changed')
            path=old/'selected_calibration'/f"{n}_full_{selected[n]['step']:05d}.json";v=read(path)
            keys={(r['source_index'],r['snr_db'],r['seed']) for r in v['rows'][n]}
            expected={(i,s,k) for i in range(1000) for s in (1.,4.,7.,13.,19.) for k in (4101,4102,4103)}
            if keys!=expected or len(v['rows'][n])!=len(expected):raise RuntimeError('incomplete saved calibration')
            summaries[n]=v['summary'][n];write_json(out/'selected_calibration'/path.name,v);reuse[n]={'path':str(path),'sha256':digest(path)}
        write_json(out/'calibration_reuse.json',reuse)
    else:
        cal=MatchedPopulation('calibration')
        for n,m in arms.items():
            checked_checkpoint(selected[n],VAR);summaries[n]=calibrate_v2({n:m},predictor,decoder,perceptual,cal,scale,out,selected[n]['step'],cfg,device,phase='selected_calibration')[n];print('selected calibration',n,summaries[n],flush=True)
        del cal
    del perceptual
    qcfg=yaml.safe_load((VAR/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,_=load_quality_models(qcfg,device);rows=[]
    with torch.no_grad():
        for i,rec in enumerate(load_targets()):
            source=split_prefix(rec['tokens'],10);label=int(rec['target']['class_index']);image=torch.from_numpy(rec['pixels'][None].astype(np.float32)/127.5-1).to(device);f=vae.quant_conv(vae.encoder(image));tx=complete_latent(vae,var,source[:8],label,device);signal,_=raw_transmit_budget(source,label,8,3060);images=[];records=[]
            for snr in (1.,4.,7.,13.,19.):
                for seed in (2001,2002,2003):
                    y=signal+seeded_noise(rec['target']['image_id'],seed,signal.shape)/np.sqrt(10**(snr/10));phy=raw_receive_budget(y,snr,3060);ok=phy['label'] is not None
                    rx=complete_latent(vae,var,phy['prefix'],phy['label'],device) if ok else torch.zeros_like(tx);status=torch.tensor([[float(ok),float(phy['body_crc_accepted']),float(phy['mode'] or 0)]],device=device)
                    b={'F':f,'Fb_TX':tx,'Fb_RX':rx,'rx_status':status,'snr_db':torch.tensor([snr],device=device),'standard_noise':torch.as_tensor(enhancement_noise(rec['target']['image_id'],seed,1024)[None],device=device,dtype=torch.float32)}
                    for n,m in arms.items():
                        latent,wave,_,_=matched_batch(m,predictor,b,cfg['predictor_nominal_snr_db'],n==names[1]);images.append(render_received(decoder,latent,status)[0].cpu().numpy());records.append({'method':n,'source_index':i,'image_id':rec['target']['image_id'],'snr_db':snr,'seed':seed,'N':4084,'E':8168,'header_ok':ok,'body_crc_ok':bool(phy['body_crc_accepted']),'checkpoint_sha256':selected[n]['checkpoint_sha256']})
            metrics,_,_=quality_metrics(rec['pixels'].astype(np.float32)/255.,images,lp,dino,device);rows.extend({**r,**m} for r,m in zip(records,metrics));write_rows(out/'partial.csv',rows);print('predictor development',i+1,flush=True)
    keys={(r['method'],r['source_index'],r['snr_db'],r['seed']) for r in rows};assert len(keys)==len(rows)==3000
    write_rows(out/'per_frame.csv',rows);write_json(out/'completion.json',{'status':'EXISTING_SELECTED_CHECKPOINTS_REEVALUATED','registration_sha256':registration,'selected':selected,'selected_calibration':summaries,'development_rows':len(rows),'per_frame_sha256':digest(out/'per_frame.csv'),'updates_this_run':0,'historical_training_source_identity':'NOT_ESTABLISHED_FROM_OLD_COMPLETION','interface':'encoder [residual,Ptx]; receiver begins at Prx','new_holdout_used':False})
if __name__=='__main__':main()
