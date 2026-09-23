"""Freeze existing equal-N endpoints at one explicit numerical protocol."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch,yaml
from .train import ROOT,PROJECT,EXP,read
from latent_enhancement.runtime import configure,require_available,model_paths,perceptual_model,image_losses,write_json,digest,snapshot
from latent_enhancement.latent import complete_latent,enhancement_noise
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.data import MatchedPopulation
from latent_enhancement_b.model import render_received
from latent_followup.timing_clean import load_arm
from latent_mechanisms.requalify_predictor import write_rows
from latent_enhancement_eval.runner import raw_transmit_budget,raw_receive_budget,load_targets
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.study import seeded_noise
from var_comm.progressive import split_prefix
from var_comm.quality import load_quality_models,quality_metrics
from latent_followup.digital_policy_matrix import evaluate_source,load_source,aggregate_candidates,freeze_policies

def read_csv(p):return list(csv.DictReader(Path(p).open()))

def folded_receive(base,wave,tx,image_id,snr,seed,vae,var,arm,device,cache):
    if base.shape!=(3572,2) or wave.shape!=(1,512,2):raise RuntimeError('folded budget mismatch')
    if abs(float(np.square(base,dtype=np.float64).sum()+wave.square().sum().item())-8168)>.01:raise RuntimeError('folded energy mismatch')
    received=base+seeded_noise(image_id,seed,base.shape)/np.sqrt(10**(snr/10))
    phy=raw_receive_budget(received,snr,3572);ok=phy['label'] is not None
    if ok:
        key=(phy['label'],tuple(np.concatenate(phy['prefix']).tolist()))
        if key not in cache:cache[key]=complete_latent(vae,var,phy['prefix'],phy['label'],device)
        rx=cache[key]
    else:rx=torch.zeros_like(tx)
    status=torch.tensor([[float(ok),float(phy['body_crc_accepted']),float(phy['mode'] or 0)]],device=device)
    noise=torch.as_tensor(enhancement_noise(image_id,seed,512)[None],device=device,dtype=torch.float32)
    return arm.receiver(wave+noise/np.sqrt(10**(snr/10)),rx,torch.tensor([snr],device=device),status),status

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    digital_root=ROOT/'followup/research_20260923_digital_strict';done=read(digital_root/'completion.json')
    digital_path=digital_root/'calibration.csv'
    if digest(digital_path)!=done['calibration_sha256']:raise RuntimeError('fresh digital calibration hash changed')
    configure();require_available();device=torch.device('cuda:0')
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);scale=scale_statistics(device)
    lp=perceptual_model(device);pop=MatchedPopulation('calibration')
    arms={n:load_arm(n,(32,16,16),scale,device) for n in ('enhancement512','enhancement1024')}
    digital_reg=read(digital_root/'registration.json')
    if digital_reg['decoder_state_sha256']!=state_sha256(decoder) or digital_reg['precision']!={'matmul_tf32':False,'cudnn_tf32':False}:
        raise RuntimeError('digital numerical scope mismatch')
    digital_rows=read_csv(digital_path);validation=[]
    with torch.no_grad():
        for index in (0,11):
            actual=evaluate_source(load_source(index),vae,var,decoder,lp,device,4084,[4101],[1.,13.],renderers=('Dc',))
            cached={(r['family'],int(r['mode']),float(r['snr_db']),int(r['seed'])):r for r in digital_rows if int(r['image_index'])==index}
            for r in actual:
                old=cached[(r['family'],r['mode'],r['snr_db'],r['seed'])]
                errors={k:abs(float(r[k])-float(old[k])) for k in ('psnr_db','lpips')}
                validation.append({'source_index':index,'family':r['family'],'mode':r['mode'],'snr_db':r['snr_db'],'seed':r['seed'],**errors})
    accepted=all(r['psnr_db']<=1e-4 and r['lpips']<=1e-5 for r in validation)
    write_json(out/'digital_reuse_validation.json',{'status':'PASS' if accepted else 'FAIL','scope':'24 checks of newly recomputed strict-precision candidates',
        'decoder_state_sha256':state_sha256(decoder),'digital_source_sha256':digest(digital_path),'absolute_errors':validation})
    if not accepted:raise RuntimeError('new digital cache reuse validation failed')
    rows=[]
    with torch.no_grad():
        for i in range(len(pop)):
            f=pop.source.values['F'][i:i+1].to(device);tx=pop.source.values['Fb_TX'][i:i+1].to(device)
            target=pop.source.images[i:i+1].to(device).float()/255;label=int(pop.source.labels[i])
            scales=vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=False);source=[s[0].cpu().numpy() for s in scales]
            base,_=raw_transmit_budget(source,label,8,3572);wave=arms['enhancement512'].encoder(f-tx,tx);cache={}
            for si,snr in enumerate(pop.snrs.tolist()):
                for ni,seed in enumerate(pop.seeds):
                    ids=torch.tensor([i]);b=pop.batch(ids,torch.tensor([si]),torch.tensor([ni]),[seed],device)
                    z,_=arms['enhancement1024'].receive_training_sample(b['F'],b['Fb_TX'],b['Fb_RX'],b['snr_db'],b['rx_status'],b['standard_noise'])
                    mse,per=image_losses(render_received(decoder,z,b['rx_status']),target,lp)
                    rows.append({'method':'m8_plus_latent_1024','source_index':i,'snr_db':snr,'seed':seed,'mse':float(mse[0]),'lpips':float(per[0]),'utility':float(mse[0]+.1*per[0])})
                    z,status=folded_receive(base,wave,tx,pop.identifiers[i],snr,seed,vae,var,arms['enhancement512'],device,cache)
                    mse,per=image_losses(render_received(decoder,z,status),target,lp)
                    rows.append({'method':'m8_plus_latent_512_fold_N4084','source_index':i,'snr_db':snr,'seed':seed,'mse':float(mse[0]),'lpips':float(per[0]),'utility':float(mse[0]+.1*per[0])})
            if i%20==0:print('allocation calibration',i,flush=True)
    for r in digital_rows:
        if r['image_id']!=pop.identifiers[int(r['image_index'])]:raise RuntimeError('digital calibration source identity mismatch')
        mse=10**(-float(r['psnr_db'])/10);per=float(r['lpips'])
        rows.append({'method':f"{r['family']}_N4084_m{r['mode']}_Dc",'source_index':int(r['image_index']),'snr_db':float(r['snr_db']),'seed':int(r['seed']),'mse':mse,'lpips':per,'utility':mse+.1*per})
    methods=sorted({r['method'] for r in rows})
    intended={f'{family}_N4084_m{mode}_Dc' for family in ('raw','arithmetic') for mode in (7,8,9)}|{'m8_plus_latent_1024','m8_plus_latent_512_fold_N4084'}
    if set(methods)!=intended or not all(np.isfinite(r['utility']) for r in rows):raise RuntimeError('invalid policy scope or utility')
    expected={(i,s,n) for i in range(1000) for s in (1.,4.,7.,13.,19.) for n in (4101,4102,4103)}
    for m in methods:
        selected=[r for r in rows if r['method']==m]
        if len(selected)!=len(expected) or {(r['source_index'],r['snr_db'],r['seed']) for r in selected}!=expected:raise RuntimeError('calibration coverage')
    actions={};scores=[]
    for snr in (1.,4.,7.,13.,19.):
        values={m:float(np.mean([r['utility'] for r in rows if r['method']==m and r['snr_db']==snr])) for m in methods}
        actions[str(snr)]=min(values,key=lambda m:(values[m],m));scores.extend({'snr_db':snr,'method':m,'utility':v} for m,v in values.items())
    (out/'calibration_by_method').mkdir()
    for m in methods:write_rows(out/'calibration_by_method'/f'{m}.csv',[r for r in rows if r['method']==m])
    write_rows(out/'calibration.csv',rows);write_rows(out/'calibration_scores.csv',scores)
    policy={'status':'FROZEN_ON_CALIBRATION_ONLY','actions':actions,'criterion':'per-frame MSE+0.1LPIPS mean','N':4084,'E':8168,
        'decoder_state_sha256':state_sha256(decoder),'candidates':methods,'selected_arms':{n:read(ROOT/f'stage_B_v1/training/selected_{n}.json') for n in arms},
        'precision':digital_reg['precision'],'digital_calibration_sha256':digest(digital_path),'calibration_sha256':digest(out/'calibration.csv'),
        'lookup_not_a_novel_resource_algorithm':True,'source_snapshot':snapshot([__file__])}
    write_json(out/'frozen_policy.json',policy)
    typed=[{**r,'raw_payload_bits':int(r['raw_payload_bits']),'actual_payload_bits':int(r['actual_payload_bits'])} for r in digital_rows]
    config=read(EXP/'followup/config.json');config={**config,'budgets':[4084],'renderers':['Dc']}
    adaptive=freeze_policies(aggregate_candidates(typed),config)
    write_json(out/'digital_adaptive_policies.json',{'actions':adaptive,'calibration_sha256':digest(digital_path),'rule_config':config})
    # Policies are now frozen. Only now read development outputs or infer folded development.
    devpath=digital_root/'development.csv'
    if digest(devpath)!=done['development_sha256']:raise RuntimeError('digital development hash changed')
    lookup={}
    for r in read_csv(devpath):
        name=f"{r['family']}_N4084_m{r['mode']}_Dc";i=int(r['image_index']);snr=float(r['snr_db']);seed=int(r['seed'])
        if (name,i,snr,seed) in lookup:raise RuntimeError('duplicate digital development key')
        lookup[(name,i,snr,seed)]={'image_id':r['image_id'],'psnr_db':float(r['psnr_db']),'lpips_alex':float(r['lpips']),'dino_cosine':float(r['dino_cosine'])}
    diagpath=ROOT/'followup/research_20260923_diagnostics/per_frame.csv'
    for r in read_csv(diagpath):
        if r['method']=='actual_RX_AWGN':lookup[('m8_plus_latent_1024',int(r['source_index']),float(r['snr_db']),int(r['seed']))]=r
    qcfg=yaml.safe_load((PROJECT/'configs/progressive_channel.yaml').read_text())['quality'];_,dino,_=load_quality_models(qcfg,device)
    folded=[]
    with torch.no_grad():
        for i,r in enumerate(load_targets()):
            source=split_prefix(r['tokens'],10);label=int(r['target']['class_index']);image_id=r['target']['image_id']
            image=torch.from_numpy(r['pixels'][None].astype(np.float32)/127.5-1).to(device);f=vae.quant_conv(vae.encoder(image))
            tx=complete_latent(vae,var,source[:8],label,device);base,_=raw_transmit_budget(source,label,8,3572);wave=arms['enhancement512'].encoder(f-tx,tx);cache={}
            images=[];records=[]
            for snr in (1.,4.,7.,13.,19.):
                for seed in (2001,2002,2003):
                    z,status=folded_receive(base,wave,tx,image_id,snr,seed,vae,var,arms['enhancement512'],device,cache)
                    images.append(render_received(decoder,z,status)[0].cpu().numpy())
                    records.append({'method':'m8_plus_latent_512_fold_N4084','source_index':i,'image_id':image_id,'snr_db':snr,'seed':seed,'N':4084,'E':8168})
            values,_,_=quality_metrics(r['pixels'].astype(np.float32)/255,images,lp,dino,device)
            for rec,metric in zip(records,values):
                item={**rec,**metric};folded.append(item);lookup[(rec['method'],i,rec['snr_db'],rec['seed'])]=item
            if i%10==0:print('folded strict development',i,flush=True)
    write_rows(out/'folded_development.csv',folded)
    expected_dev={(i,s,n) for i in range(100) for s in (1.,4.,7.,13.,19.) for n in (2001,2002,2003)}
    images={}
    for m in methods:
        if {key[1:] for key in lookup if key[0]==m}!=expected_dev:raise RuntimeError('development endpoint coverage')
    for key,r in lookup.items():
        if images.setdefault(key[1],r['image_id'])!=r['image_id']:raise RuntimeError('development identity mismatch')
    result=[];digital_result=[]
    for i,snr,seed in sorted(expected_dev):
        name=actions[str(snr)];r=lookup[(name,i,snr,seed)]
        result.append({'method':'calibration_frozen_resource_lookup','selected_endpoint':name,'source_index':i,'image_id':r['image_id'],'snr_db':snr,'seed':seed,'N':4084,'E':8168,**{k:float(r[k]) for k in ('psnr_db','lpips_alex','dino_cosine')}})
        for family in ('raw','arithmetic'):
            mode=adaptive[family]['4084']['Dc'][str(snr)]['quality'];name=f'{family}_N4084_m{mode}_Dc';r=lookup[(name,i,snr,seed)]
            digital_result.append({'method':family+'_adaptive_m789_Dc','selected_mode':mode,'source_index':i,'image_id':r['image_id'],'snr_db':snr,'seed':seed,'N':4084,'E':8168,**{k:float(r[k]) for k in ('psnr_db','lpips_alex','dino_cosine')}})
    write_rows(out/'per_frame.csv',result);write_rows(out/'digital_adaptive_per_frame.csv',digital_result)
    write_json(out/'completion.json',{'status':'STRICT_PRECISION_EXISTING_ENDPOINT_LOOKUP_EVALUATED','rows':1500,'digital_adaptive_rows':3000,
        'policy_sha256':digest(out/'frozen_policy.json'),'development_sources':{str(p):digest(p) for p in (devpath,diagpath)},
        'new_holdout_used':False,'not_full_m10_or_new_decoder_digital_adaptation':True})
if __name__=='__main__':main()
