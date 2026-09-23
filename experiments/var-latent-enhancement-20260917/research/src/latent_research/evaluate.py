"""Actual selected-model development and CPU RGB endpoint cost measurement."""
import argparse,time,json
from pathlib import Path
import numpy as np,torch,yaml
from .train import ROOT,PROJECT,CONFIG,read,make_models,NAMES
from .models import prefix_latent
from latent_enhancement.runtime import configure,require_available,model_paths,settings,write_json,digest,snapshot,verify_snapshot
from latent_enhancement.latent import complete_latent,enhancement_noise
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.model import render_received
from latent_followup.run_identity import checked_checkpoint
from latent_enhancement_eval.runner import load_targets,raw_transmit_budget,raw_receive_budget
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.study import seeded_noise
from var_comm.quality import load_quality_models,quality_metrics
from latent_mechanisms.requalify_predictor import write_rows

@torch.no_grad()
def execute(record,name,model,snr,seed,vae,var,decoder,device):
    torch.cuda.synchronize();t=time.perf_counter();image=torch.from_numpy(record['pixels'][None].astype(np.float32)/127.5-1).to(device);f=vae.quant_conv(vae.encoder(image))
    if name=='pure_continuous':wave=model.transmit(f)[0].cpu().numpy();signal=wave
    else:
        scales=vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=False);source=[x[0].cpu().numpy() for x in scales];label=int(record['target']['class_index']);base,_=raw_transmit_budget(source,label,8,3060)
        if not np.array_equal(np.concatenate(source[:8]),np.asarray(record['tokens'])[:255]):raise RuntimeError('actual encoded m8 prefix differs from frozen development source')
        if name=='light_tx':condition=prefix_latent(vae,scales);wave=model.encoder(f,condition)[0].cpu().numpy()
        else:condition=complete_latent(vae,var,source[:8],label,device);wave=model.encoder(f-condition,condition)[0].cpu().numpy()
        signal=np.concatenate((base,wave))
    torch.cuda.synchronize();tx_ms=(time.perf_counter()-t)*1000
    if signal.shape!=(4084,2) or abs(float(np.square(signal,dtype=np.float64).sum())-8168)>0.01:raise RuntimeError('actual N/E mismatch')
    if name=='pure_continuous':observed=signal+seeded_noise('VAR-CONTINUOUS-4084|'+record['target']['image_id'],seed,signal.shape)/np.sqrt(10**(snr/10))
    else:observed=np.concatenate((base+seeded_noise(record['target']['image_id'],seed,base.shape)/np.sqrt(10**(snr/10)),wave+enhancement_noise(record['target']['image_id'],seed,1024)/np.sqrt(10**(snr/10))))
    t=time.perf_counter();st=torch.tensor([snr],device=device)
    if name=='pure_continuous':
        z=model.receive(torch.as_tensor(observed[None],device=device,dtype=torch.float32),st);pred=decoder(z);header=1;body='not_applicable'
    else:
        phy=raw_receive_budget(observed[:3060],snr,3060);header=int(phy['label'] is not None);body=int(phy['body_crc_accepted']);base_rx=complete_latent(vae,var,phy['prefix'],phy['label'],device) if header else torch.zeros((1,32,16,16),device=device)
        status=torch.tensor([[header,body,float(phy['mode'] or 0)]],device=device);z=model.receiver(torch.as_tensor(observed[3060:][None],device=device,dtype=torch.float32),base_rx,st,status);pred=render_received(decoder,z,status)
    image=pred[0].cpu().numpy();torch.cuda.synchronize();rx_ms=(time.perf_counter()-t)*1000
    return image,{'tx_ms':tx_ms,'rx_ms':rx_ms,'total_ms':tx_ms+rx_ms,'header_ok':header,'body_crc_ok':body,'normalized_latent':float(((z-f)/model.scale if name=='pure_continuous' else (z-f)/model.encoder.scale).square().mean())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--output',required=True);a=p.parse_args();training=Path(a.training);done=read(training/'completion.json');out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    registration=read(training/'registration.json');verify_snapshot(registration['bindings'])
    configure();require_available();device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device);r=read(ROOT/'stage_B_v1/training/selected_enhancement1024.json');parent=torch.load(checked_checkpoint(r,PROJECT),map_location='cpu',weights_only=True);models=make_models(scale,settings()['stage_B'],parent,device);del parent
    if state_sha256(decoder)!=registration['decoder_state_sha256']:raise RuntimeError('selected evaluation Decoder changed')
    selected={}
    for n in NAMES:
        rec=read(training/f'selected_{n}.json');path=checked_checkpoint(rec,PROJECT);payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['registration_sha256']!=done['registration_sha256']:raise RuntimeError('selected registration mismatch')
        if rec!=done['selected'][n] or rec['arm_key']!=n:raise RuntimeError('selected record drifted after training completion')
        prefix=n+'.';models[n].load_state_dict({k[len(prefix):]:v for k,v in payload['models'].items() if k.startswith(prefix)},strict=True);models[n].eval().requires_grad_(False);selected[n]=rec
    write_json(out/'registration.json',{'source_snapshot':snapshot([__file__,Path(__file__).with_name('models.py'),CONFIG]),'selected':selected,'training_completion_sha256':digest(training/'completion.json'),'hardware':{'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'cuda':torch.version.cuda,'cpu_threads':torch.get_num_threads()},'quality_timing_contract':'same execute function; CPU RGB -> CPU waveform and noisy CPU waveform -> CPU RGB; one noise per segment outside RX'})
    qcfg=yaml.safe_load((PROJECT/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,_=load_quality_models(qcfg,device);targets=load_targets();rows=[];quality_reference={};timing_sources=(0,11,22,33,44,55,66,77,88,99)
    for i,r in enumerate(targets):
        images=[];records=[]
        for snr in (1.,4.,7.,13.,19.):
            for seed in (2001,2002,2003):
                for n in NAMES:
                    img,info=execute(r,n,models[n],snr,seed,vae,var,decoder,device);images.append(img);records.append({'method':n,'source_index':i,'image_id':r['target']['image_id'],'snr_db':snr,'seed':seed,'N':4084,'E':8168,'checkpoint_sha256':selected[n]['checkpoint_sha256'],'decoder_sha256':registration['decoder_state_sha256'],'protocol_id':n+':N4084:E8168','normalized_latent':info['normalized_latent'],'header_ok':info['header_ok'],'body_crc_ok':info['body_crc_ok']})
        for img,rec in zip(images,records):
            if i in timing_sources and rec['seed']==2001:quality_reference[(i,rec['snr_db'],rec['method'])]=img.copy()
        metrics,_,_=quality_metrics(r['pixels'].astype(np.float32)/255.,images,lp,dino,device);rows.extend({**r,**m} for r,m in zip(records,metrics));write_rows(out/'partial.csv',rows);print('new methods development',i+1,flush=True)
    assert len(rows)==4500 and len({(r['method'],r['source_index'],r['snr_db'],r['seed']) for r in rows})==4500
    write_rows(out/'per_frame.csv',rows)
    warm=[]
    for n in NAMES:
        for repeat in range(5):
            _,v=execute(targets[0],n,models[n],13.,2001,vae,var,decoder,device);warm.append({'method':n,'repeat':repeat,**{k:v[k] for k in ('tx_ms','rx_ms','total_ms')}})
    timings=[];consistency=[]
    for i in timing_sources:
        for snr in (1.,4.,7.,13.,19.):
            for repeat in range(2):
                shift=(i+repeat)%3
                for pos,n in enumerate(NAMES[shift:]+NAMES[:shift]):
                    img,v=execute(targets[i],n,models[n],snr,2001,vae,var,decoder,device);timings.append({'method':n,'source_index':i,'snr_db':snr,'seed':2001,'repeat':repeat,'position':pos,**{k:v[k] for k in ('tx_ms','rx_ms','total_ms')}})
                    error=float(np.max(np.abs(img-quality_reference[(i,snr,n)])))
                    if error>2e-5:raise RuntimeError('quality/timing output mismatch')
                    consistency.append({'method':n,'source_index':i,'snr_db':snr,'repeat':repeat,'max_abs_error':error,'tolerance':2e-5})
    write_rows(out/'quality_timing_consistency.csv',consistency)
    write_rows(out/'timing.csv',timings);write_rows(out/'warmup.csv',warm)
    write_json(out/'completion.json',{'status':'SELECTED_NEW_METHODS_DEVELOPMENT_AND_TIMING_COMPLETE','rows':4500,'timed_calls':len(timings),'registration_sha256':digest(out/'registration.json'),'per_frame_sha256':digest(out/'per_frame.csv'),'no_model_selection_on_development':True,'new_holdout_used':False})
if __name__=='__main__':main()
