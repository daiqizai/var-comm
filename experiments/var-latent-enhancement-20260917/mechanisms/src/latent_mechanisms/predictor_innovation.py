from __future__ import annotations
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from latent_enhancement.runtime import image_losses, model_paths, perceptual_model, save_torch, settings
from latent_enhancement_b.common import load_decoder, scale_statistics
from latent_enhancement_b.data import MatchedPopulation
from latent_enhancement_b.model import build_arms, latent_errors, render_received
from latent_enhancement.training import PairedOrder
from var_comm.next_scale_prior import load_models

PROJECT=Path(__file__).resolve().parents[5];VAR=PROJECT;ROOT=VAR/'outputs/VAR-LATENT-ENHANCEMENT-20260917';EXP=VAR/'experiments/var-latent-enhancement-20260917';CONFIG=EXP/'mechanisms/predictor_innovation_v2_config.json'
def read(p): return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def write_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def selected_record(p):
 r=read(p);c=Path(r['checkpoint']);c=(VAR/c).resolve() if not c.is_absolute() else c
 if not c.exists():raise FileNotFoundError(c)
 if digest(c)!=r['checkpoint_sha256']:raise RuntimeError(f'selected checkpoint changed: {c}')
 return r,c,digest(c)
def load_shared_predictor(p,shape,scale,recipe,device):
 r,c,sha=selected_record(p);s=torch.load(c,map_location='cpu',weights_only=True);a=build_arms(shape,scale,recipe).to(device);a.load_state_dict(s['arms'],strict=True);return a['receiver_only_refiner'].eval().requires_grad_(False),{'record':r,'checkpoint':str(c),'checkpoint_sha256':sha}
def shared_predictor(predictor,base,snr,device):
 status=torch.tensor([1.,1.,8.],device=device,dtype=base.dtype).expand(len(base),3);nominal=torch.full((len(base),),float(snr),device=device,dtype=base.dtype);return predictor.receiver(None,base,nominal,status)
def matched_batch(model,predictor,batch,nominal_snr,innovation):
 with torch.no_grad():ptx=shared_predictor(predictor,batch['Fb_TX'],nominal_snr,batch['Fb_TX'].device);prx=shared_predictor(predictor,batch['Fb_RX'],nominal_snr,batch['Fb_RX'].device)
 residual=batch['F']-(ptx if innovation else batch['Fb_TX']);latent,wave=model.receive_training_residual_sample(residual,ptx,prx,batch['snr_db'],batch['rx_status'],batch['standard_noise']);return latent,wave,ptx,prx
@torch.no_grad()
def calibrate_v2(arms,predictor,decoder,perceptual,population,scale,out,step,cfg,device,phase='calibration'):
 started=time.perf_counter();rows={n:[] for n in arms};micro=int(cfg['microbatch_size'])
 for start in range(0,len(population),micro):
  ids=torch.arange(start,min(start+micro,len(population)))
  for si in range(len(population.snrs)):
   for ni in range(len(population.seeds)):
    seeds=torch.full((len(ids),),int(population.seeds[ni]),dtype=torch.long);b=population.batch(ids,torch.full((len(ids),),si,dtype=torch.long),torch.full((len(ids),),ni,dtype=torch.long),seeds,device)
    for name,m in arms.items():
     z,_,_,_=matched_batch(m,predictor,b,cfg['predictor_nominal_snr_db'],name=='predicted_innovation_candidate');pred=render_received(decoder,z,b['rx_status']);mse,lp=image_losses(pred,b['target'],perceptual);aux=latent_errors(z,b['F'],scale,b['rx_status'])
     for j in range(len(ids)):rows[name].append({'source_index':int(ids[j]),'snr_db':float(b['snr_db'][j]),'seed':int(seeds[j]),'mse':float(mse[j]),'lpips':float(lp[j]),'normalized_latent':float(aux[j])})
 summary={}
 for n,v in rows.items():
  a=np.asarray([[x['mse'],x['lpips'],x['normalized_latent']] for x in v],dtype=np.float64)
  if not np.isfinite(a).all():raise RuntimeError(f'nonfinite calibration metrics: {n}')
  summary[n]={'mse':float(a[:,0].mean()),'lpips':float(a[:,1].mean()),'normalized_latent':float(a[:,2].mean()),'utility':float((a[:,0]+.1*a[:,1]+.01*a[:,2]).mean()),'rows':len(v),'snrs_db':population.snrs.tolist(),'seeds':list(population.seeds)}
 write_json(Path(out)/phase/f'full_{step:05d}.json',{'step':step,'phase':phase,'rows':rows,'summary':summary,'elapsed_seconds':time.perf_counter()-started});return summary
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'followup/predictor_innovation_v2'));a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cfg=read(CONFIG);device=torch.device('cuda:0');vae,_=load_models(model_paths(),device);torch.cuda.empty_cache();recipe=settings()['stage_B'];train=MatchedPopulation('train');cal=MatchedPopulation('calibration');scale=scale_statistics(device);shape=train.shape
 parent_record,parent_path,parent_sha=selected_record(ROOT/'stage_B_v1/training/selected_enhancement1024.json');parent=torch.load(parent_path,map_location='cpu',weights_only=True);arms=build_arms(shape,scale,recipe).to(device);arms.load_state_dict(parent['arms'],strict=True);control=arms['enhancement1024'];candidate=copy.deepcopy(control);decoder=load_decoder(vae,device).eval().requires_grad_(False);perceptual=perceptual_model(device);predictor,predictor_id=load_shared_predictor(ROOT/'stage_B_v1/training/selected_receiver_only_refiner.json',shape,scale,recipe,device)
 oc=torch.optim.AdamW(control.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);ox=torch.optim.AdamW(candidate.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);parent_opt=parent.get('optimizers',{}).get('enhancement1024')
 if parent_opt is not None:oc.load_state_dict(copy.deepcopy(parent_opt));ox.load_state_dict(copy.deepcopy(parent_opt))
 order=PairedOrder(len(train),int(cfg['data_seed']));rng=torch.Generator().manual_seed(int(cfg['channel_seed']));state={'step':0,'last_full_step':-1,'selection':{'original_residual_control':{'step':0,'utility':None},'predicted_innovation_candidate':{'step':0,'utility':None}},'plateau_checks':{'original_residual_control':0,'predicted_innovation_candidate':0},'update_seconds':0.,'calibration_seconds':0.,'calibration_calls':0,'parent_step':int(parent_record.get('step',0)),'parent_checkpoint_sha256':parent_sha,'predictor_checkpoint_sha256':predictor_id['checkpoint_sha256']}
 latest=out/'latest.json'
 if latest.exists():
  r=read(latest);ck=torch.load(r['path'],map_location='cpu',weights_only=True);control.load_state_dict(ck['control'],strict=True);candidate.load_state_dict(ck['candidate'],strict=True);oc.load_state_dict(copy.deepcopy(ck['opt_control']));ox.load_state_dict(copy.deepcopy(ck['opt_candidate']));order.load_state_dict(ck['order']);rng.set_state(ck['channel_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);state=ck['state']
 def checkpoint(reason):
  path=out/'checkpoints'/f"step_{state['step']:05d}.pt";path.parent.mkdir(parents=True,exist_ok=True);save_torch(path,{'control':control.state_dict(),'candidate':candidate.state_dict(),'opt_control':oc.state_dict(),'opt_candidate':ox.state_dict(),'order':order.state_dict(),'channel_rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'state':state,'parent_checkpoint_sha256':parent_sha,'predictor_checkpoint_sha256':predictor_id['checkpoint_sha256']});write_json(out/'latest.json',{'path':str(path),'sha256':digest(path),'step':state['step'],'reason':reason,'state':state})
  for n,k in (('original_residual_control','control'),('predicted_innovation_candidate','candidate')):
   if state['selection'][n]['step']==state['step']:write_json(out/f'selected_{n}.json',{**state['selection'][n],'checkpoint':str(path),'checkpoint_sha256':digest(path),'arm_key':k,'parent_step':state['parent_step'],'incremental_step':state['step'],'total_step':state['parent_step']+state['step']})
 def full_calibration():
  start=time.perf_counter();summary=calibrate_v2({'original_residual_control':control.eval(),'predicted_innovation_candidate':candidate.eval()},predictor,decoder,perceptual,cal,scale,out,state['step'],cfg,device)
  for n,s in summary.items():
   old=state['selection'][n]['utility'];improved=old is None or s['utility']<old*(1-float(cfg['relative_improvement']));state['plateau_checks'][n]=0 if improved else state['plateau_checks'][n]+1
   if old is None or s['utility']<old:state['selection'][n]={'step':state['step'],'utility':s['utility']}
  state['last_full_step']=state['step'];state['calibration_calls']+=1;state['calibration_seconds']+=time.perf_counter()-start;checkpoint('full_calibration')
 if state['step']==0:full_calibration()
 while state['step']<int(cfg['updates']['maximum']) and not(state['step']>=int(cfg['updates']['minimum']) and all(v>=int(cfg['plateau_checks']) for v in state['plateau_checks'].values())):
  ids=order.next(int(recipe['logical_batch_size']));si=torch.randint(len(train.snrs),(len(ids),),generator=rng);ni=torch.randint(len(train.seeds),(len(ids),),generator=rng);seeds=torch.full_like(ids,int(recipe['channel_seed'])+state['step']);batch=train.batch(ids,si,ni,seeds,device);start=time.perf_counter()
  for m,opt,innovation in ((control,oc,False),(candidate,ox,True)):
   opt.zero_grad(set_to_none=True);z,_,_,_=matched_batch(m,predictor,batch,cfg['predictor_nominal_snr_db'],innovation);pred=render_received(decoder,z,batch['rx_status']);mse,lp=image_losses(pred,batch['target'],perceptual);aux=latent_errors(z,batch['F'],scale,batch['rx_status']);((mse+recipe['LPIPS_weight']*lp+recipe['normalized_latent_weight']*aux).mean()).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),recipe['gradient_clip_norm'],error_if_nonfinite=True);opt.step()
  state['step']+=1;state['update_seconds']+=time.perf_counter()-start
  if state['step']%int(cfg['updates']['full_calibration_interval'])==0:full_calibration()
  elif state['step']%int(cfg['updates']['checkpoint_interval'])==0:checkpoint('regular')
 if state['last_full_step']!=state['step']:full_calibration()
 selected={}
 for n,r in state['selection'].items():
  path=out/'checkpoints'/f"step_{int(r['step']):05d}.pt";selected[n]={**r,'checkpoint':str(path),'checkpoint_sha256':digest(path),'parent_step':state['parent_step'],'incremental_step':int(r['step']),'total_step':state['parent_step']+int(r['step'])}
 selected_evaluation={}
 for n,r in selected.items():
  payload=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
  arm=copy.deepcopy(control).to(device)
  arm.load_state_dict(payload[r['arm_key']],strict=True)
  selected_evaluation[n]={'checkpoint':r['checkpoint'],'checkpoint_sha256':digest(r['checkpoint']),'parent_step':r['parent_step'],'incremental_step':r['incremental_step'],'total_step':r['total_step'],'summary':calibrate_v2({n:arm.eval()},predictor,decoder,perceptual,cal,scale,out,r['step'],cfg,device,phase='selected_evaluation')[n]}
 write_json(out/'completion.json',{'status':'MATCHED_PREDICTOR_INNOVATION_COMPLETE','algorithm':'matched_tx_shared_P_rx_shared_P_two_arm_training','updates':state['step'],'parent_step':state['parent_step'],'selected_checkpoints':selected,'selected_checkpoint_evaluation':selected_evaluation,'evaluation_population':'calibration_existing_actual_rx_cache','predictor':predictor_id,'predictor_nominal_snr_db':cfg['predictor_nominal_snr_db'],'calibration_calls':state['calibration_calls'],'calibration_seconds':state['calibration_seconds'],'new_holdout_used':False,'old_exploration':str(ROOT/'followup/predictor_innovation_v1')})
if __name__=='__main__':main()
