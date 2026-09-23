import argparse,json
from pathlib import Path
import numpy as np,torch,yaml
from latent_enhancement.runtime import configure,require_available,model_paths,write_json,digest,snapshot
from latent_enhancement.latent import complete_latent,enhancement_noise
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.model import render_received
from latent_followup.timing_clean import load_arm,OUT_ROOT
from latent_mechanisms.requalify_predictor import write_rows
from latent_enhancement_eval.runner import load_targets,raw_transmit_budget,raw_receive_budget
from var_comm.next_scale_prior import load_models
from var_comm.progressive import split_prefix
from var_comm.study import seeded_noise
from var_comm.quality import load_quality_models,quality_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    configure();require_available();device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device);arm=load_arm('enhancement1024',(32,16,16),scale,device)
    root=OUT_ROOT.parents[1];qcfg=yaml.safe_load((root/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,_=load_quality_models(qcfg,device);rows=[]
    with torch.no_grad():
        for i,r in enumerate(load_targets()):
            source=split_prefix(r['tokens'],10);label=int(r['target']['class_index']);image=torch.from_numpy(r['pixels'][None].astype(np.float32)/127.5-1).to(device);f=vae.quant_conv(vae.encoder(image));tx=complete_latent(vae,var,source[:8],label,device);signal,_=raw_transmit_budget(source,label,8,3060);wave=arm.encoder(f-tx,tx);reference=decoder(f)[0].cpu().numpy();images=[];records=[]
            for snr in (1.,4.,7.,13.,19.):
                for seed in (2001,2002,2003):
                    y=signal+seeded_noise(r['target']['image_id'],seed,signal.shape)/np.sqrt(10**(snr/10));phy=raw_receive_budget(y,snr,3060);ok=phy['label'] is not None;rx=complete_latent(vae,var,phy['prefix'],phy['label'],device) if ok else torch.zeros_like(tx)
                    status=torch.tensor([[float(ok),float(phy['body_crc_accepted']),float(phy['mode'] or 0)]],device=device);snr_tensor=torch.tensor([snr],device=device);noise=torch.as_tensor(enhancement_noise(r['target']['image_id'],seed,1024)[None],device=device,dtype=torch.float32)
                    for base_name,base in [('actual_RX',rx),('oracle_TX',tx)]:
                        for noise_name,multiplier in [('AWGN',1.),('zero_noise',0.)]:
                            z=arm.receiver(wave+noise*multiplier/np.sqrt(10**(snr/10)),base,snr_tensor,status);images.append(render_received(decoder,z,status)[0].cpu().numpy());records.append({'method':base_name+'_'+noise_name,'source_index':i,'image_id':r['target']['image_id'],'snr_db':snr,'seed':seed,'N':4084,'E':8168,'diagnostic_only':base_name!='actual_RX' or noise_name!='AWGN','header_ok':ok,'normalized_latent':float(((z-f)/scale[None,:,None,None]).square().mean())})
                    images.append(reference);records.append({'method':'Dc_F_representation_reference','source_index':i,'image_id':r['target']['image_id'],'snr_db':snr,'seed':seed,'N':'not_a_wireless_method','E':'not_a_wireless_method','diagnostic_only':True,'header_ok':ok,'normalized_latent':0.})
            metrics,_,_=quality_metrics(r['pixels'].astype(np.float32)/255.,images,lp,dino,device);rows.extend({**r,**m} for r,m in zip(records,metrics));write_rows(out/'partial.csv',rows);print('diagnosis',i+1,flush=True)
    assert len(rows)==7500 and len({(r['method'],r['source_index'],r['snr_db'],r['seed']) for r in rows})==7500
    write_rows(out/'per_frame.csv',rows);write_json(out/'completion.json',{'status':'FIXED_1024_DIAGNOSIS_COMPLETE','rows':7500,'source_snapshot':snapshot([__file__]),'checkpoint':json.loads((OUT_ROOT/'stage_B_v1/training/selected_enhancement1024.json').read_text()),'per_frame_sha256':digest(out/'per_frame.csv'),'neural_SNR_and_status_fixed':True,'privileged_inputs_diagnostic_only':True,'new_holdout_used':False})
if __name__=='__main__':main()
